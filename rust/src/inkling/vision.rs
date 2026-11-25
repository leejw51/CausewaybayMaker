//! Inkling 2's image encoder: a vision transformer over 20-pixel patches with a learned
//! position table (interpolated to the image's patch grid), two-dimensional rotary phases in
//! every attention layer and per-layer output scales. Each 2x2 block of patch outputs is merged
//! and projected into one text-width embedding, so an image costs one token per 40x40 pixels.
//! Those embeddings take the place of the image-placeholder tokens in the prompt.
//!
//! Images are not resized to a fixed shape: the pixel grid is padded up to a multiple of 40.
//! Large images are scaled down first (`MAX_SIDE`); Ollama's runner feeds them at full size.
//!
//! The encoder works in bf16 throughout, as the reference does. Three small Metal kernels
//! (`Fused`) replace the chains of elementwise ops around its matrix products (the rotation of
//! queries and keys, the GELU, the scaled residual sums), which otherwise take as long as the
//! products themselves: a 1,024-pixel image encodes in under 60 ms on an M5 Max.
//!
//! Ported from Ollama's MLX runner (ollama/ollama, mlxrunner/model/inkling2/vision.go, MIT
//! License).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::concatenate;
use mlx_rs::{Array, Dtype, fast};
use serde_json::Value;

use super::f32a;
use super::ollama::Manifest;
use super::ops::{Kernel, Launch};

/// Images with a longer side than this many pixels are scaled down to it (26 x 26 tokens for a
/// square image; the encoder's attention grows with the square of the patch count).
pub const MAX_SIDE: u32 = 1024;
/// The placeholder token whose embeddings an image replaces.
pub const IMAGE_TOKEN: &str = "<|unused_200054|>";

/// CLIP's normalization, which the pixels go through (rounded to bf16) before the patch
/// encoder maps them back to [-1, 1], as in the reference.
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Pixels of a decoded image, padded to whole tokens of `side` pixels: transparency is
/// flattened onto a checkerboard, and the padding holds the value -1 of 255.
fn pixels(rgba: &image::RgbaImage, side: i32) -> Result<Pixels> {
    let (w, h) = (rgba.width() as i32, rgba.height() as i32);
    if w == 0 || h == 0 {
        bail!("the image is empty");
    }
    let (rows, cols) = ((h - 1) / side + 1, (w - 1) / side + 1);
    let (height, width) = (rows * side, cols * side);
    let mut data = vec![0f32; (height * width * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            // Outside the image the reference uses the value -1 of 255.
            let mut rgb = [-1i32; 3];
            if x < w && y < h {
                let p = rgba.get_pixel(x as u32, y as u32).0;
                let bg = if ((x / 8) + (y / 8)) % 2 == 1 { 153 } else { 204 };
                let alpha = p[3] as i32;
                for c in 0..3 {
                    rgb[c] = (p[c] as i32 * alpha + bg * (255 - alpha) + 127) / 255;
                }
            }
            for c in 0..3 {
                data[(3 * (y * width + x)) as usize + c] = (rgb[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(Pixels { data, rows, cols, side })
}

/// `MAX_SIDE`, or INKLING_IMAGE_SIDE pixels (0: no limit, as in Ollama, where a 3,000-pixel
/// photo is 5,625 tokens).
fn max_side() -> u32 {
    match std::env::var("INKLING_IMAGE_SIDE").ok().and_then(|v| v.parse::<u32>().ok()) {
        Some(0) => u32::MAX,
        Some(side) => side,
        None => MAX_SIDE,
    }
}

/// Decode an image file, upright, its longer side at most `max_side` pixels.
fn open(path: &Path, max_side: u32) -> Result<image::RgbaImage> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(scaled(img, max_side).to_rgba8())
}

/// `img`, scaled down (keeping its shape) if a side is longer than `max_side`.
fn scaled(img: image::DynamicImage, max_side: u32) -> image::DynamicImage {
    if img.width().max(img.height()) > max_side { img.resize(max_side, max_side, image::imageops::FilterType::Lanczos3) } else { img }
}

struct Config {
    patch: i32,
    merge: i32,
    width: i32,
    layers: usize,
    heads: i32,
    grid: i32,
    eps: f32,
}

impl Config {
    fn parse(v: &Value) -> Result<Self> {
        let int = |k: &str| v[k].as_i64().with_context(|| format!("vision config has no {k}"));
        let c = Self {
            patch: int("native_patch_size")? as i32,
            merge: int("adapter_merge_size")? as i32,
            width: int("hidden_size")? as i32,
            layers: int("num_hidden_layers")? as usize,
            heads: int("num_attention_heads")? as i32,
            grid: int("posemb_grid_size")? as i32,
            eps: v["layer_norm_eps"].as_f64().context("vision config has no layer_norm_eps")? as f32,
        };
        if c.patch <= 0 || c.merge <= 0 || c.heads <= 0 || c.width % c.heads != 0 || (c.width / c.heads) % 4 != 0 || v["projector_bias"].as_bool() == Some(true) {
            bail!("unsupported Inkling 2 vision configuration");
        }
        Ok(c)
    }
}

struct Linear {
    w: Array,
    b: Option<Array>,
}

impl Linear {
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = x.matmul(self.w.t())?;
        Ok(match &self.b {
            Some(b) => y.add(b)?,
            None => y,
        })
    }
}

struct Norm {
    w: Array,
    b: Array,
}

struct Layer {
    norm: Norm,
    ff_norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
    attn_scale: Array,
    ff_scale: Array,
}

pub struct Vision {
    cfg: Config,
    patch: Linear,
    /// The class token with its position offset, [1, W].
    class: Array,
    positions: Array,
    pre_norm: Norm,
    layers: Vec<Layer>,
    project: Linear,
    out: Linear,
    fused: Fused,
}

/// A decoded image, padded to whole tokens: CLIP-normalized pixels [rows * side, cols * side, 3].
pub struct Pixels {
    data: Vec<f32>,
    pub rows: i32,
    pub cols: i32,
    side: i32,
}

impl Pixels {
    /// Tokens the image takes in the prompt.
    pub fn tokens(&self) -> usize {
        (self.rows * self.cols) as usize
    }
}

impl Vision {
    /// Load the encoder's tensors (`files`: tensor name -> safetensors path).
    pub fn load(m: &Manifest) -> Result<Self> {
        let cfg = Config::parse(&m.json("config.json")?["vision_config"])?;
        let p = "model.visual.";
        let mut map: HashMap<String, Array> = HashMap::new();
        let files: Vec<&PathBuf> = m.files.iter().filter(|(name, _)| name.starts_with(p)).map(|(_, path)| path).collect();
        for path in files {
            map.extend(Array::load_safetensors(m.safetensors(path)?)?);
        }
        let mut take = |name: &str| map.remove(&format!("{p}{name}")).with_context(|| format!("Inkling tensor {p}{name} is missing"));
        let e = "backbone.embeddings.";
        let patch = Linear { w: take(&format!("{e}patch_embedding.weight"))?, b: None };
        let class = take(&format!("{e}class_embedding"))?.add(take(&format!("{e}cls_position_embedding"))?)?.reshape(&[1, cfg.width])?;
        let positions = take(&format!("{e}position_embedding"))?;
        let pre_norm = Norm { w: take("backbone.ln_pre.weight")?, b: take("backbone.ln_pre.bias")? };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("backbone.layers.{i}.");
            let mut linear = |name: &str| -> Result<Linear> { Ok(Linear { w: take(&format!("{l}{name}.weight"))?, b: Some(take(&format!("{l}{name}.bias"))?) }) };
            let (q, k, v, o) = (linear("self_attn.q_proj")?, linear("self_attn.k_proj")?, linear("self_attn.v_proj")?, linear("self_attn.out_proj")?);
            let (up, down) = (linear("mlp.c_fc")?, linear("mlp.c_proj")?);
            layers.push(Layer {
                norm: Norm { w: take(&format!("{l}ln_1.weight"))?, b: take(&format!("{l}ln_1.bias"))? },
                ff_norm: Norm { w: take(&format!("{l}ln_2.weight"))?, b: take(&format!("{l}ln_2.bias"))? },
                q,
                k,
                v,
                o,
                up,
                down,
                attn_scale: take(&format!("{l}ls_1.gamma"))?,
                ff_scale: take(&format!("{l}ls_2.gamma"))?,
            });
        }
        let project = Linear { w: take("proj_in.weight")?, b: None };
        let out = Linear { w: take("proj_out.weight")?, b: None };
        if patch.w.shape() != [cfg.width, cfg.patch * cfg.patch * 3] || positions.shape() != [cfg.grid * cfg.grid, cfg.width] || project.w.dim(1) != cfg.merge * cfg.merge * cfg.width {
            bail!("Inkling 2 vision tensors have unexpected shapes");
        }
        Ok(Self { cfg, patch, class, positions, pre_norm, layers, project, out, fused: Fused::new() })
    }

    /// Pixels of one prompt token along each axis.
    fn side(&self) -> i32 {
        self.cfg.patch * self.cfg.merge
    }

    /// Run the encoder once on a blank image of 2 x 2 tokens: reads the weights in and compiles
    /// the kernels.
    pub fn warm(&self) -> Result<()> {
        let side = self.side();
        let blank = image::RgbaImage::from_pixel(2 * side as u32, 2 * side as u32, image::Rgba([127, 127, 127, 255]));
        self.encode(&pixels(&blank, side)?)?;
        Ok(())
    }

    /// Read an image file: turn it upright (EXIF orientation), scale it down if it is large,
    /// flatten transparency onto a checkerboard, and pad to whole tokens.
    pub fn read(&self, path: &Path) -> Result<Pixels> {
        let context = || format!("cannot read image {}", path.display());
        pixels(&open(path, max_side()).with_context(context)?, self.side()).with_context(context)
    }

    /// One embedding per prompt token: [rows * cols, text width] bf16.
    pub fn encode(&self, pixels: &Pixels) -> Result<Array> {
        let c = &self.cfg;
        let (rows, cols) = (pixels.rows, pixels.cols);
        let (prows, pcols) = (rows * c.merge, cols * c.merge);
        let bf = Dtype::Bfloat16;
        let data = Array::from_slice(&pixels.data, &[rows * pixels.side, cols * pixels.side, 3]).as_dtype(bf)?;
        let (mean, std) = (Array::from_slice(&MEAN, &[3]).as_dtype(bf)?, Array::from_slice(&STD, &[3]).as_dtype(bf)?);
        let data = data.multiply(&std)?.add(&mean)?.multiply(Array::from_f32(2.0).as_dtype(bf)?)?.subtract(Array::from_f32(1.0).as_dtype(bf)?)?;
        let patches = data.reshape(&[prows, c.patch, pcols, c.patch, 3])?.transpose_axes(&[0, 2, 1, 3, 4])?.reshape(&[prows * pcols, c.patch * c.patch * 3])?;
        let mut h = self.patch.forward(&patches)?.add(self.positions(prows, pcols)?)?;
        h = concatenate(&[&self.class, &h], 0)?;
        h = fast::layer_norm(&h, &self.pre_norm.w, &self.pre_norm.b, c.eps)?;
        let (cos, sin) = phases(prows, pcols, c.width / c.heads)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, c, &self.fused)?;
        }
        // Drop the class token and merge each merge x merge block of patches.
        let merged = h
            .index((1.., ..))
            .reshape(&[rows, c.merge, cols, c.merge, c.width])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[rows * cols, c.merge * c.merge * c.width])?;
        let out = self.out.forward(&self.fused.gelu(&merged, &self.project)?)?;
        out.eval()?;
        Ok(out)
    }

    /// The position table, bilinearly interpolated to a rows x cols patch grid: [rows * cols, W].
    fn positions(&self, rows: i32, cols: i32) -> Result<Array> {
        let size = self.cfg.grid;
        let (mut indices, mut weights) = (Vec::new(), Vec::new());
        let at = |i: i32, n: i32| ((i as f64 + 0.5) * size as f64 / n as f64 - 0.5).clamp(0.0, (size - 1) as f64);
        for row in 0..rows {
            for col in 0..cols {
                let (y, x) = (at(row, rows), at(col, cols));
                let (y0, x0) = (y as i32, x as i32);
                let (y1, x1) = ((y0 + 1).min(size - 1), (x0 + 1).min(size - 1));
                let (dy, dx) = ((y - y0 as f64) as f32, (x - x0 as f64) as f32);
                indices.extend([y0 * size + x0, y0 * size + x1, y1 * size + x0, y1 * size + x1]);
                weights.extend([(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx]);
            }
        }
        let n = rows * cols;
        let selected = f32a(&self.positions.take_axis(Array::from_slice(&indices, &[n, 4]), 0)?)?;
        Ok(selected.multiply(Array::from_slice(&weights, &[n, 4, 1]))?.sum_axes(&[1], false)?.as_dtype(self.positions.dtype())?)
    }
}

/// Rotary phases for the class token (at the origin) and each patch (column and row, from 1):
/// cos and sin, [count, head_dim / 2], the first half of the frequencies turning with x and
/// the second with y.
fn phases(rows: i32, cols: i32, head_dim: i32) -> Result<(Array, Array)> {
    let count = rows * cols + 1;
    let (mut x, mut y) = (vec![0f32; count as usize], vec![0f32; count as usize]);
    for i in 0..(count - 1) {
        x[i as usize + 1] = (i % cols + 1) as f32;
        y[i as usize + 1] = (i / cols + 1) as f32;
    }
    let n = head_dim / 4;
    let freq: Vec<f32> = (0..n).map(|i| 10000f64.powf(-4.0 * i as f64 / head_dim as f64) as f32).collect();
    let freq = Array::from_slice(&freq, &[1, n]);
    let angles = concatenate(&[Array::from_slice(&x, &[count, 1]).multiply(&freq)?, Array::from_slice(&y, &[count, 1]).multiply(&freq)?], -1)?;
    Ok((angles.cos()?, angles.sin()?))
}

// Rotate each (even, odd) pair of x [count, heads * DIM] by its row's phase in f32, and write
// the result head-major: [1, heads, count, DIM]. One thread per pair.
const ROTATE_SOURCE: &str = r#"
#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
const uint half_dim = DIM / 2;
const uint count = x_shape[0];
uint pair = i % half_dim;
uint head = (i / half_dim) % HEADS;
uint row = i / (half_dim * HEADS);
uint src = (row * HEADS + head) * DIM + 2 * pair;
uint dst = (head * count + row) * DIM + 2 * pair;
float even = float(x[src]);
float odd = float(x[src + 1]);
float c = cs[row * half_dim + pair];
float s = sn[row * half_dim + pair];
out[dst] = static_cast<T>(even * c - odd * s);
out[dst + 1] = static_cast<T>(even * s + odd * c);
"#;

// erf to within an ulp in f32 (the polynomial MLX's own kernels use).
const ERF_HEADER: &str = r#"
inline float erf32(float a) {
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f) {
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f);
        r = metal::fma(r, t, -6.34846687e-1f);
        r = metal::fma(r, t, -1.28717512e-1f);
        r = metal::fma(r, t, -t);
        r = 1.0f - metal::exp(r);
        r = metal::copysign(r, a);
    } else {
        r = -5.96761703e-4f;
        r = metal::fma(r, s, 4.99119423e-3f);
        r = metal::fma(r, s, -2.67681349e-2f);
        r = metal::fma(r, s, 1.12819925e-1f);
        r = metal::fma(r, s, -3.76125336e-1f);
        r = metal::fma(r, s, 1.28379166e-1f);
        r = metal::fma(r, a, a);
    }
    return r;
}
"#;

// gelu(y + bias) for a product y [N, C], rounded like the separate ops (the sum, then the
// activation). BIAS = 0: no bias.
const GELU_SOURCE: &str = r#"
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
float v = float(y[i]);
if (BIAS) v = float(static_cast<T>(v + float(b[i % y_shape[1]])));
out[i] = static_cast<T>(v * (1.0f + erf32(v * 0.70710678118654752f)) * 0.5f);
"#;

// h + (y + bias) * scale for a product y [N, C], rounded like the separate ops.
const RESIDUAL_SOURCE: &str = r#"
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
uint c = i % y_shape[1];
float v = float(static_cast<T>(float(y[i]) + float(b[c])));
v = float(static_cast<T>(v * float(scale[c])));
out[i] = static_cast<T>(float(h[i]) + v);
"#;

/// One thread per element of `like`: the kernel's single output, in `like`'s shape and dtype.
fn elementwise(kernel: &Kernel, inputs: &[&Array], like: &Array, ints: &[(&str, i32)]) -> Result<Array> {
    let n = like.size() as i32;
    Ok(kernel
        .run(Launch { inputs, ints, dtypes: &[("T", like.dtype())], grid: (n, 1, 1), group: (n.clamp(1, 256), 1, 1), outputs: &[(like.shape(), like.dtype())] })?
        .pop()
        .unwrap())
}

/// The encoder's fused kernels.
struct Fused {
    rotate: Kernel,
    gelu: Kernel,
    residual: Kernel,
}

impl Fused {
    fn new() -> Self {
        Self {
            rotate: Kernel::new("inkling_vision_rotate", &["x", "cs", "sn"], &["out"], "", ROTATE_SOURCE),
            gelu: Kernel::new("inkling_vision_gelu", &["y", "b"], &["out"], ERF_HEADER, GELU_SOURCE),
            residual: Kernel::new("inkling_vision_residual", &["h", "y", "b", "scale"], &["out"], "", RESIDUAL_SOURCE),
        }
    }

    /// gelu(x @ w^T + b)
    fn gelu(&self, x: &Array, l: &Linear) -> Result<Array> {
        let y = x.matmul(l.w.t())?;
        match &l.b {
            Some(b) => elementwise(&self.gelu, &[&y, b], &y, &[("BIAS", 1)]),
            None => elementwise(&self.gelu, &[&y, &y], &y, &[("BIAS", 0)]),
        }
    }

    /// h + (x @ w^T + b) * scale
    fn residual(&self, h: &Array, x: &Array, l: &Linear, scale: &Array) -> Result<Array> {
        let y = x.matmul(l.w.t())?;
        elementwise(&self.residual, &[h, &y, l.b.as_ref().context("this layer has a bias")?, scale], h, &[])
    }
}

/// Rotate each (even, odd) pair of x [count, heads * dim] by its row's phase (cos and sin
/// [count, dim / 2]), in f32: [1, heads, count, dim] in x's dtype.
fn rotate(kernel: &Kernel, x: &Array, cos: &Array, sin: &Array, heads: i32) -> Result<Array> {
    let count = x.dim(0);
    let dim = x.dim(1) / heads;
    let threads = count * heads * dim / 2;
    Ok(kernel
        .run(Launch {
            inputs: &[x, cos, sin],
            ints: &[("DIM", dim), ("HEADS", heads)],
            dtypes: &[("T", x.dtype())],
            grid: (threads, 1, 1),
            group: (threads.clamp(1, 256), 1, 1),
            outputs: &[(&[1, heads, count, dim], x.dtype())],
        })?
        .pop()
        .unwrap())
}

impl Layer {
    fn forward(&self, h: &Array, cos: &Array, sin: &Array, c: &Config, fused: &Fused) -> Result<Array> {
        let (count, width) = (h.dim(0), h.dim(1));
        let dim = width / c.heads;
        let n = fast::layer_norm(h, &self.norm.w, &self.norm.b, c.eps)?;
        // [count, heads, dim] -> [1, heads, count, dim]
        let by_head = |x: Array| -> Result<Array> { Ok(x.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
        let q = rotate(&fused.rotate, &self.q.forward(&n)?, cos, sin, c.heads)?;
        let k = rotate(&fused.rotate, &self.k.forward(&n)?, cos, sin, c.heads)?;
        let v = by_head(self.v.forward(&n)?.reshape(&[count, c.heads, dim])?)?;
        let a = fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / (dim as f32).sqrt(), None, None)?;
        let a = a.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[count, width])?;
        let h = fused.residual(h, &a, &self.o, &self.attn_scale)?;
        let up = fused.gelu(&fast::layer_norm(&h, &self.ff_norm.w, &self.ff_norm.b, c.eps)?, &self.up)?;
        fused.residual(&h, &up, &self.down, &self.ff_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tolerance * (1.0 + y.abs()), "element {i}: {x} vs {y}");
        }
    }

    #[test]
    fn an_image_is_padded_to_whole_tokens() {
        // 50 x 90 pixels with 40-pixel tokens: 2 columns, 3 rows.
        let img = image::RgbaImage::from_pixel(50, 90, image::Rgba([255, 0, 51, 255]));
        let p = pixels(&img, 40).unwrap();
        assert_eq!((p.cols, p.rows, p.tokens()), (2, 3, 6));
        assert_eq!(p.data.len(), 80 * 120 * 3);
        let at = |x: usize, y: usize, c: usize| p.data[3 * (y * 80 + x) + c];
        let norm = |v: f32, c: usize| (v / 255.0 - MEAN[c]) / STD[c];
        for c in 0..3 {
            assert_eq!(at(49, 89, c), norm([255.0, 0.0, 51.0][c], c), "inside the image");
            assert_eq!(at(50, 0, c), norm(-1.0, c), "right of the image");
            assert_eq!(at(0, 90, c), norm(-1.0, c), "below the image");
        }
    }

    #[test]
    fn transparency_shows_a_checkerboard() {
        let img = image::RgbaImage::from_pixel(40, 40, image::Rgba([10, 20, 30, 0]));
        let p = pixels(&img, 40).unwrap();
        let red = |x: usize, y: usize| p.data[3 * (y * 40 + x)];
        let norm = |v: f32| (v / 255.0 - MEAN[0]) / STD[0];
        assert_eq!(red(0, 0), norm(204.0));
        assert_eq!(red(8, 0), norm(153.0));
        assert_eq!(red(8, 8), norm(204.0));
    }

    #[test]
    fn large_images_are_scaled_down_in_proportion() {
        let small = scaled(image::DynamicImage::new_rgb8(600, 400), 1024);
        assert_eq!((small.width(), small.height()), (600, 400));
        let wide = scaled(image::DynamicImage::new_rgb8(3000, 1500), 1024);
        assert_eq!((wide.width(), wide.height()), (1024, 512));
        let tall = scaled(image::DynamicImage::new_rgb8(1000, 4000), 1024);
        assert_eq!((tall.width(), tall.height()), (256, 1024));
        let kept = scaled(image::DynamicImage::new_rgb8(3000, 1500), u32::MAX);
        assert_eq!((kept.width(), kept.height()), (3000, 1500));
    }

    #[test]
    fn an_image_file_is_read_and_a_missing_one_is_an_error() {
        let dir = std::env::temp_dir().join(format!("inkling-vision-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wide.png");
        image::RgbaImage::from_pixel(2000, 500, image::Rgba([1, 2, 3, 255])).save(&path).unwrap();
        let img = open(&path, 1024).unwrap();
        assert_eq!((img.width(), img.height()), (1024, 256));
        assert!(open(&dir.join("missing.png"), 1024).is_err());
        std::fs::write(dir.join("not-an-image.png"), b"hello").unwrap();
        assert!(open(&dir.join("not-an-image.png"), 1024).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(dir.join("not-an-image.png")).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_photo_is_turned_upright_by_its_exif_orientation() {
        // A 40 x 20 JPEG, red on the left and blue on the right, tagged "rotate 90 degrees
        // clockwise to display" (orientation 6): upright it is 20 x 40, red on top.
        let mut img = image::RgbImage::new(40, 20);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = if x < 20 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) };
        }
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg).unwrap();
        // An APP1 segment right after the start marker: "Exif", a little-endian TIFF header and
        // one directory entry, tag 0x0112 (orientation), a short, value 6.
        let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
        let mut tagged = jpeg[..2].to_vec();
        tagged.extend([0xff, 0xe1]);
        tagged.extend(((exif.len() + 2) as u16).to_be_bytes());
        tagged.append(&mut exif);
        tagged.extend(&jpeg[2..]);
        let dir = std::env::temp_dir().join(format!("inkling-exif-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (plain, turned) = (dir.join("plain.jpg"), dir.join("turned.jpg"));
        std::fs::write(&plain, &jpeg).unwrap();
        std::fs::write(&turned, &tagged).unwrap();
        let as_stored = open(&plain, 1024).unwrap();
        assert_eq!((as_stored.width(), as_stored.height()), (40, 20));
        let upright = open(&turned, 1024).unwrap();
        assert_eq!((upright.width(), upright.height()), (20, 40));
        let (top, bottom) = (upright.get_pixel(10, 5).0, upright.get_pixel(10, 35).0);
        assert!(top[0] > 200 && top[2] < 60, "the top is red: {top:?}");
        assert!(bottom[2] > 200 && bottom[0] < 60, "the bottom is blue: {bottom:?}");
        std::fs::remove_file(&plain).unwrap();
        std::fs::remove_file(&turned).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn the_class_token_is_not_rotated_and_patches_count_from_one() {
        let (cos, sin) = phases(2, 3, 8).unwrap();
        assert_eq!(cos.shape(), &[7, 4]);
        let (cos, sin) = (values(&cos), values(&sin));
        close(&cos[..4], &[1.0; 4], 0.0);
        close(&sin[..4], &[0.0; 4], 0.0);
        // The last patch is at column 3, row 2; the first two frequencies (1 and 0.01) turn
        // with x, the last two with y.
        let last = &sin[6 * 4..];
        close(last, &[3f32.sin(), 0.03f32.sin(), 2f32.sin(), 0.02f32.sin()], 1e-5);
    }

    #[test]
    fn the_rotation_kernel_matches_the_plain_formula() {
        let fused = Fused::new();
        let (count, heads, dim) = (5, 3, 8);
        let x: Vec<f32> = (0..count * heads * dim).map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0).collect();
        let (cos, sin) = phases(2, 2, dim).unwrap();
        let input = Array::from_slice(&x, &[count, heads * dim]);
        let out = rotate(&fused.rotate, &input, &cos, &sin, heads).unwrap();
        assert_eq!(out.shape(), &[1, heads, count, dim]);
        let (out, cos, sin) = (values(&out), values(&cos), values(&sin));
        for row in 0..count as usize {
            for head in 0..heads as usize {
                for pair in 0..(dim / 2) as usize {
                    let src = (row * heads as usize + head) * dim as usize + 2 * pair;
                    let dst = (head * count as usize + row) * dim as usize + 2 * pair;
                    let (c, s) = (cos[row * (dim / 2) as usize + pair], sin[row * (dim / 2) as usize + pair]);
                    close(&out[dst..dst + 2], &[x[src] * c - x[src + 1] * s, x[src] * s + x[src + 1] * c], 1e-5);
                }
            }
        }
    }

    #[test]
    fn the_fused_kernels_match_the_separate_operations() {
        let fused = Fused::new();
        let (n, c_in, c_out) = (6, 4, 5);
        let x = Array::from_slice(&(0..n * c_in).map(|i| ((i * 29 % 83) as f32 - 40.0) / 13.0).collect::<Vec<_>>(), &[n, c_in]);
        let w = Array::from_slice(&(0..c_out * c_in).map(|i| ((i * 17 % 61) as f32 - 30.0) / 40.0).collect::<Vec<_>>(), &[c_out, c_in]);
        let b = Array::from_slice(&[0.5f32, -0.25, 0.0, 1.5, -2.0], &[c_out]);
        let scale = Array::from_slice(&[1.0f32, 0.5, -0.5, 2.0, 0.125], &[c_out]);
        let h = Array::from_slice(&(0..n * c_out).map(|i| (i as f32) / 7.0 - 2.0).collect::<Vec<_>>(), &[n, c_out]);
        let linear = Linear { w, b: Some(b) };
        let y = linear.forward(&x).unwrap();

        // gelu(v) = v * (1 + erf(v / sqrt(2))) / 2, on both sides of the polynomial's switch.
        let expected = values(&mlx_rs::nn::gelu(&y).unwrap());
        close(&values(&fused.gelu(&x, &linear).unwrap()), &expected, 1e-5);
        assert!(values(&y).iter().any(|v| v.abs() > 1.0) && values(&y).iter().any(|v| v.abs() < 0.9));
        let no_bias = Linear { w: linear.w.clone(), b: None };
        let expected = values(&mlx_rs::nn::gelu(no_bias.forward(&x).unwrap()).unwrap());
        close(&values(&fused.gelu(&x, &no_bias).unwrap()), &expected, 1e-5);

        let expected = values(&h.add(y.multiply(&scale).unwrap()).unwrap());
        close(&values(&fused.residual(&h, &x, &linear, &scale).unwrap()), &expected, 1e-5);
    }
}
