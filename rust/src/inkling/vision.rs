//! Inkling 2's image encoder: a vision transformer over 20-pixel patches with a learned
//! position table (interpolated to the image's patch grid), two-dimensional rotary phases in
//! every attention layer and per-layer output scales. Each 2x2 block of patch outputs is merged
//! and projected into one text-width embedding, so an image costs one token per 40x40 pixels.
//! Those embeddings take the place of the image-placeholder tokens in the prompt.
//!
//! Images are not resized to a fixed shape: the pixel grid is padded up to a multiple of 40.
//! Large images are scaled down first (`MAX_SIDE`), which Ollama's runner leaves to its caller.
//!
//! Ported from Ollama's MLX runner (ollama/ollama, mlxrunner/model/inkling2/vision.go, MIT
//! License).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate, stack};
use mlx_rs::{Array, Dtype, fast, nn};
use serde_json::Value;

use super::f32a;
use super::ollama::Manifest;

/// Images with a longer side than this many pixels are scaled down to it (26 x 26 tokens for a
/// square image; the encoder's attention grows with the square of the patch count).
pub const MAX_SIDE: u32 = 1024;
/// The placeholder token whose embeddings an image replaces.
pub const IMAGE_TOKEN: &str = "<|unused_200054|>";

/// CLIP's normalization, which the pixels go through (rounded to bf16) before the patch
/// encoder maps them back to [-1, 1], as in the reference.
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

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
        Ok(Self { cfg, patch, class, positions, pre_norm, layers, project, out })
    }

    /// Pixels of one prompt token along each axis.
    fn side(&self) -> i32 {
        self.cfg.patch * self.cfg.merge
    }

    /// Read an image file: scale it down if it is large, flatten transparency onto a
    /// checkerboard, and pad to whole tokens.
    pub fn read(&self, path: &Path) -> Result<Pixels> {
        let mut img = image::open(path).with_context(|| format!("cannot read image {}", path.display()))?;
        if img.width().max(img.height()) > MAX_SIDE {
            img = img.resize(MAX_SIDE, MAX_SIDE, image::imageops::FilterType::Lanczos3);
        }
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width() as i32, rgba.height() as i32);
        if w == 0 || h == 0 {
            bail!("image {} is empty", path.display());
        }
        let side = self.side();
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
            h = layer.forward(&h, &cos, &sin, c)?;
        }
        // Drop the class token and merge each merge x merge block of patches.
        let merged = h
            .index((1.., ..))
            .reshape(&[rows, c.merge, cols, c.merge, c.width])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[rows * cols, c.merge * c.merge * c.width])?;
        let out = self.out.forward(&nn::gelu(self.project.forward(&merged)?)?)?;
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
/// cos and sin, [count, 1, head_dim / 2], the first half of the frequencies turning with x and
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
    Ok((angles.cos()?.expand_dims(1)?, angles.sin()?.expand_dims(1)?))
}

/// Rotate each (even, odd) pair of x [count, heads, dim] by its phase, in f32.
fn rotate(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let (count, heads, dim) = (x.dim(0), x.dim(1), x.dim(2));
    let pairs = f32a(x)?.reshape(&[count, heads, dim / 2, 2])?;
    let (even, odd) = (pairs.index((.., .., .., 0)), pairs.index((.., .., .., 1)));
    let real = even.multiply(cos)?.subtract(odd.multiply(sin)?)?;
    let imag = even.multiply(sin)?.add(odd.multiply(cos)?)?;
    Ok(stack(&[real, imag], -1)?.reshape(&[count, heads, dim])?.as_dtype(x.dtype())?)
}

impl Layer {
    fn forward(&self, h: &Array, cos: &Array, sin: &Array, c: &Config) -> Result<Array> {
        let (count, width) = (h.dim(0), h.dim(1));
        let dim = width / c.heads;
        let n = fast::layer_norm(h, &self.norm.w, &self.norm.b, c.eps)?;
        // [count, heads, dim] -> [1, heads, count, dim]
        let by_head = |x: Array| -> Result<Array> { Ok(x.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
        let q = by_head(rotate(&self.q.forward(&n)?.reshape(&[count, c.heads, dim])?, cos, sin)?)?;
        let k = by_head(rotate(&self.k.forward(&n)?.reshape(&[count, c.heads, dim])?, cos, sin)?)?;
        let v = by_head(self.v.forward(&n)?.reshape(&[count, c.heads, dim])?)?;
        let a = fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / (dim as f32).sqrt(), None, None)?;
        let a = a.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[count, width])?;
        let h = h.add(self.o.forward(&a)?.multiply(&self.attn_scale)?)?;
        let f = self.down.forward(&nn::gelu(self.up.forward(&fast::layer_norm(&h, &self.ff_norm.w, &self.ff_norm.b, c.eps)?)?)?)?;
        Ok(h.add(f.multiply(&self.ff_scale)?)?)
    }
}
