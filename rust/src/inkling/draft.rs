//! Inkling 2's draft network (`Inkling2ForConditionalGenerationMTP`): three small transformer
//! layers that propose the next few tokens in one pass, for the target model to verify together.
//!
//! The draft reads the target's own states. For every committed token it takes the target's
//! residual at a few visits plus its output row (the "features"), mixes them per draft layer and
//! stores them as cross-attention keys and values. To propose, it runs a block of the current
//! token followed by mask embeddings, each row attending to that context and to the whole
//! block. A small recurrent head then picks the tokens left to right, correcting each row's
//! scores with the tokens already chosen.
//!
//! Ported from Ollama's MLX runner (ollama/ollama, mlxrunner/model/inkling2/draft.go, MIT
//! License).

use anyhow::{Context, Result, bail};
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::{broadcast_to, concatenate, quantize, quantized_matmul, sigmoid, softmax_axis, tanh, zeros_dtype};
use mlx_rs::{Array, Dtype, fast, nn, random};
use serde_json::Value;

use super::ollama::Manifest;
use super::{Model, f32a, linear, model_rms, softcap};

/// Bits of the draft's projections and of its copy of the vocabulary head (MLX affine
/// quantization, as Ollama does on Metal). Proposals only steer verification, so the draft
/// trades weight precision for bytes read per round. Zero keeps the published bf16 weights.
const BITS: i32 = 8;
const VOCABULARY_BITS: i32 = 4;
const GROUP: i32 = 64;
/// Feature rows held before they are written into the draft's context.
const FLUSH_ROWS: i32 = 256;
/// Rows of the vocabulary head folded at a time while loading.
const FOLD_ROWS: i32 = 32768;

struct Config {
    width: i32,
    layers: usize,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    local: Vec<usize>,
    window: i32,
    theta: f32,
    eps: f32,
    block: i32,
    state: i32,
    taps: Vec<usize>,
}

impl Config {
    fn parse(root: &Value, target: &Model) -> Result<Self> {
        let int = |k: &str| root[k].as_i64().with_context(|| format!("draft config has no {k}"));
        let list = |k: &str| -> Vec<usize> { root[k].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default() };
        let c = Self {
            width: int("hidden_size")? as i32,
            layers: int("num_hidden_layers")? as usize,
            heads: int("num_attention_heads")? as i32,
            kv_heads: int("num_key_value_heads")? as i32,
            head_dim: int("head_dim")? as i32,
            local: list("local_layer_ids"),
            window: int("sliding_window_size")? as i32,
            theta: root["rope_theta"].as_f64().context("draft config has no rope_theta")? as f32,
            eps: root["rms_norm_eps"].as_f64().context("draft config has no rms_norm_eps")? as f32,
            block: int("block_size")? as i32,
            state: int("markov_rank")? as i32,
            taps: list("target_layer_ids"),
        };
        let supported = root["architectures"][0].as_str() == Some("Inkling2ForConditionalGenerationMTP")
            && root["markov_head_type"].as_str() == Some("rnn_hidden")
            && root["q_bias"].as_bool() != Some(true)
            && root["o_bias"].as_bool() != Some(true)
            && c.width == target.cfg.hidden
            && c.layers > 0
            && c.heads % c.kv_heads == 0
            && c.block >= 2
            && c.taps.windows(2).all(|w| w[0] < w[1])
            && c.taps.last() == Some(&target.plan.len());
        if !supported {
            bail!("unsupported Inkling 2 draft configuration");
        }
        Ok(c)
    }
}

/// One or several projections stacked as the rows of one matrix, so the input is read once.
struct Projection {
    w: Array,
    /// Affine scales and biases when quantized.
    quant: Option<(Array, Array)>,
    bits: i32,
}

impl Projection {
    fn new(bits: i32, parts: &[Array]) -> Result<Self> {
        let w = if parts.len() == 1 { parts[0].clone() } else { concatenate(parts, 0)? };
        // Affine groups span whole input columns; keep a width they cannot tile.
        if bits > 0 && w.dim(1) % GROUP == 0 {
            let (w, scales, biases) = quantize(&w, GROUP, bits)?;
            return Ok(Self { w, quant: Some((scales, biases)), bits });
        }
        Ok(Self { w, quant: None, bits: 0 })
    }

    /// x [.., in] -> [.., out]
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut shape = x.shape().to_vec();
        let flat = x.reshape(&[-1, x.dim(-1)])?;
        let y = match &self.quant {
            Some((scales, biases)) => quantized_matmul(&flat, &self.w, scales, biases, true, GROUP, self.bits)?,
            None => linear(&flat, &self.w)?,
        };
        *shape.last_mut().unwrap() = y.dim(-1);
        Ok(y.reshape(&shape)?)
    }

    fn arrays(&self) -> impl Iterator<Item = &Array> {
        std::iter::once(&self.w).chain(self.quant.iter().flat_map(|(s, b)| [s, b]))
    }
}

struct Layer {
    local: bool,
    input_norm: Array,
    ff_norm: Array,
    context_norm: Array,
    query_norm: Array,
    key_norm: Array,
    context_key_norm: Array,
    /// Query, key and value rows.
    qkv: Projection,
    /// Key and value rows for the context features.
    context_kv: Projection,
    output: Projection,
    gate_up: Projection,
    down: Projection,
}

pub struct Draft {
    cfg: Config,
    layers: Vec<Layer>,
    /// Softmax of the tap weights, [layers + 1, taps] f32: one mixture per layer, then the head's.
    fuse: Array,
    mask: Array,
    norm: Array,
    head_norm: Array,
    /// Embedding of the previous token for the recurrent head, [vocab, state].
    previous: Array,
    recurrent_w: Array,
    recurrent_b: Array,
    /// The head's latent projection folded into the target's vocabulary head, [vocab, state].
    folded: Projection,
    folded_bias: Array,
    /// The draft's own (coarser) copy of the vocabulary head.
    vocabulary: Projection,
    logit_cap: f32,
    /// Measured milliseconds per round by depth (`speculate::calibrate`), the starting point
    /// of each conversation's depth choice.
    pub(crate) round_costs: Vec<f64>,
}

/// The draft's per-conversation state.
pub struct Cache {
    /// Per layer: rotated context keys and values [1, KV, n, D]; local layers keep a window.
    context: Vec<Option<(Array, Array)>>,
    /// The head's mixture of the latest committed row, [1, W].
    feature: Option<Array>,
    /// Context rows written: the position of the next one.
    count: i32,
    /// Feature rows [1, t, taps * W] not yet written.
    queued: Vec<Array>,
    queued_rows: i32,
    /// Depth statistics learned over the conversation.
    pub(crate) depth: super::speculate::Learned,
}

impl Cache {
    /// Queue the features of newly fed rows.
    pub(crate) fn queue(&mut self, features: Array) {
        self.queued_rows += features.dim(1);
        self.queued.push(features);
    }

    /// Keep only the first `keep` rows of the latest queued forward.
    pub(crate) fn keep_latest(&mut self, keep: i32) -> Result<()> {
        let last = self.queued.pop().context("no queued features to take back")?;
        self.queued_rows -= last.dim(1) - keep;
        self.queued.push(last.index((.., ..keep, ..)));
        Ok(())
    }

    /// Committed rows, written or queued.
    pub(crate) fn rows(&self) -> i32 {
        self.count + self.queued_rows
    }

    pub(crate) fn state(&self) -> Vec<&Array> {
        let mut all: Vec<&Array> = self.context.iter().flatten().flat_map(|(k, v)| [k, v]).collect();
        all.extend(self.feature.iter());
        all.extend(self.queued.iter());
        all
    }
}

/// A proposal: `tokens` [n] int32, and for sampled decoding the distribution each was drawn
/// from, [n, vocab] f32 (greedy proposals are point masses and carry none).
pub struct Proposal {
    pub tokens: Array,
    pub probs: Option<Array>,
}

impl Draft {
    pub fn load(m: &Manifest, target: &Model) -> Result<Self> {
        let cfg = Config::parse(&m.json("draft/config.json")?, target)?;
        let p = "draft.";
        let take = |name: &str| -> Result<Array> {
            let key = format!("{p}{name}");
            let mut map = Array::load_safetensors(m.safetensors(m.file(&key)?)?)?;
            map.remove(&key).with_context(|| format!("Inkling tensor {key} is missing"))
        };
        let f32w = |name: &str| -> Result<Array> { f32a(&take(name)?) };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("layers.{i}.");
            let a = format!("{l}self_attn.");
            layers.push(Layer {
                local: cfg.local.contains(&i),
                input_norm: f32w(&format!("{l}input_layernorm.weight"))?,
                ff_norm: f32w(&format!("{l}post_attention_layernorm.weight"))?,
                context_norm: f32w(&format!("{l}dflare_norm.weight"))?,
                query_norm: f32w(&format!("{a}q_norm.weight"))?,
                key_norm: f32w(&format!("{a}k_norm.weight"))?,
                context_key_norm: f32w(&format!("{a}k_norm_ctx.weight"))?,
                qkv: Projection::new(BITS, &[take(&format!("{a}q_proj.weight"))?, take(&format!("{a}k_proj.weight"))?, take(&format!("{a}v_proj.weight"))?])?,
                context_kv: Projection::new(BITS, &[take(&format!("{a}k_proj_ctx.weight"))?, take(&format!("{a}v_proj_ctx.weight"))?])?,
                output: Projection::new(BITS, &[take(&format!("{a}o_proj.weight"))?])?,
                gate_up: Projection::new(BITS, &[take(&format!("{l}mlp.gate_proj.weight"))?, take(&format!("{l}mlp.up_proj.weight"))?])?,
                down: Projection::new(BITS, &[take(&format!("{l}mlp.down_proj.weight"))?])?,
            });
        }
        // The mixes use the bf16 softmax of the tap weights, in f32.
        let fuse = f32a(&softmax_axis(&f32a(&take("fuse_weight")?)?, -1, true)?.as_dtype(Dtype::Bfloat16)?)?;
        if fuse.shape() != [cfg.layers as i32 + 1, cfg.taps.len() as i32] {
            bail!("Inkling draft: unexpected tap weights");
        }
        let h = "markov_head.";
        // Fold the latent projection into the target's head once, accumulating in f32 (a slice
        // of the head at a time, so the f32 copy stays small).
        let (latent_w, latent_b) = (f32w(&format!("{h}hidden_out.weight"))?, f32w(&format!("{h}hidden_out.bias"))?);
        let vocab = target.unembed.dim(0);
        let (mut folded, mut folded_bias) = (Vec::new(), Vec::new());
        for start in (0..vocab).step_by(FOLD_ROWS as usize) {
            let head = f32a(&target.unembed.index((start..(start + FOLD_ROWS).min(vocab), ..)))?;
            let w = head.matmul(&latent_w)?.as_dtype(Dtype::Bfloat16)?;
            let b = head.matmul(&latent_b)?.as_dtype(Dtype::Bfloat16)?;
            mlx_rs::transforms::eval([&w, &b])?;
            folded.push(w);
            folded_bias.push(b);
        }
        let (folded, folded_bias) = (concatenate(&folded, 0)?, concatenate(&folded_bias, 0)?);
        let draft = Self {
            layers,
            fuse,
            mask: take("mask_embed")?,
            norm: f32w("norm.weight")?,
            head_norm: f32w(&format!("{h}dflare_norm.weight"))?,
            previous: take(&format!("{h}markov_w1.weight"))?,
            recurrent_w: take(&format!("{h}rnn_proj.weight"))?,
            recurrent_b: take(&format!("{h}rnn_proj.bias"))?,
            folded: Projection::new(BITS, &[folded])?,
            folded_bias,
            vocabulary: Projection::new(VOCABULARY_BITS, &[target.unembed.clone()])?,
            logit_cap: target.cfg.logit_cap,
            round_costs: Vec::new(),
            cfg,
        };
        if draft.recurrent_w.shape() != [3 * draft.cfg.state, 2 * (draft.cfg.state + draft.cfg.width)] {
            bail!("Inkling draft: unexpected recurrent head");
        }
        let mut all: Vec<&Array> = vec![&draft.fuse, &draft.mask, &draft.norm, &draft.head_norm, &draft.previous, &draft.recurrent_w, &draft.recurrent_b, &draft.folded_bias];
        all.extend(draft.folded.arrays().chain(draft.vocabulary.arrays()));
        for l in &draft.layers {
            all.extend([&l.input_norm, &l.ff_norm, &l.context_norm, &l.query_norm, &l.key_norm, &l.context_key_norm]);
            all.extend(l.qkv.arrays().chain(l.context_kv.arrays()).chain(l.output.arrays()).chain(l.gate_up.arrays()).chain(l.down.arrays()));
        }
        mlx_rs::transforms::eval(all)?;
        Ok(draft)
    }

    /// The target visits whose incoming residual the draft reads (the last entry, one past the
    /// final visit, stands for the target's output row).
    pub(crate) fn taps(&self) -> &[usize] {
        &self.cfg.taps[..self.cfg.taps.len() - 1]
    }

    /// The most tokens one proposal holds.
    pub fn limit(&self) -> usize {
        self.cfg.block as usize - 1
    }

    pub fn make_cache(&self) -> Cache {
        Cache { context: self.layers.iter().map(|_| None).collect(), feature: None, count: 0, queued: Vec::new(), queued_rows: 0, depth: super::speculate::Learned::seeded(&self.round_costs) }
    }

    /// Keys normalized and rotated to their positions, and values, both arranged by head:
    /// [1, n, KV * D] -> [1, KV, n, D].
    fn rotate(&self, key: &Array, value: &Array, norm: &Array, position: i32) -> Result<(Array, Array)> {
        let c = &self.cfg;
        let n = key.dim(1);
        let k = model_rms(&key.reshape(&[1, n, c.kv_heads, c.head_dim])?, norm, c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = value.reshape(&[1, n, c.kv_heads, c.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        Ok((fast::rope(&k, c.head_dim, false, c.theta, 1.0, position, None)?, v))
    }

    /// Write the queued feature rows into the context.
    fn flush(&self, cache: &mut Cache) -> Result<()> {
        if cache.queued.is_empty() {
            return Ok(());
        }
        let c = &self.cfg;
        let features = if cache.queued.len() == 1 { cache.queued.pop().unwrap() } else { concatenate(&cache.queued, 1)? };
        cache.queued.clear();
        let n = features.dim(1);
        let kvw = c.kv_heads * c.head_dim;
        // One f32 product with the tap weights forms every layer's mixture and the head's.
        let mixed = self.fuse.matmul(&features.reshape(&[1, n, c.taps.len() as i32, c.width])?)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let mix = model_rms(&mixed.index((.., .., i as i32, ..)), &layer.context_norm, c.eps)?;
            let kv = layer.context_kv.forward(&mix)?;
            let (mut k, mut v) = self.rotate(&kv.index((.., .., ..kvw)), &kv.index((.., .., kvw..)), &layer.context_key_norm, cache.count)?;
            if let Some((pk, pv)) = &cache.context[i] {
                k = concatenate(&[pk, &k], 2)?;
                v = concatenate(&[pv, &v], 2)?;
            }
            // A local layer's block sees the window before its first row.
            let extra = k.dim(2) - c.window;
            if layer.local && extra > 0 {
                k = k.index((.., .., extra.., ..));
                v = v.index((.., .., extra.., ..));
            }
            cache.context[i] = Some((k, v));
        }
        let head = model_rms(&mixed.index((.., n - 1, c.layers as i32, ..)), &self.head_norm, c.eps)?;
        cache.feature = Some(head);
        cache.count += n;
        cache.queued_rows = 0;
        Ok(())
    }

    /// Write queued rows once enough have gathered (keeps a long prefill or a stretch of
    /// plain decoding from holding every feature row).
    pub(crate) fn flush_if_full(&self, cache: &mut Cache) -> Result<()> {
        if cache.queued_rows >= FLUSH_ROWS {
            self.flush(cache)?;
            // Nothing reads the context until the next proposal: evaluate it now, so it stops
            // holding the feature rows.
            mlx_rs::transforms::async_eval(cache.state())?;
        }
        Ok(())
    }

    /// Propose the `n` tokens after `current` ([1] int32, the token after the committed rows).
    /// `temp` <= 0 proposes the best-scoring token at each position.
    pub fn propose(&self, target: &Model, cache: &mut Cache, current: &Array, n: usize, temp: f32) -> Result<Proposal> {
        let c = &self.cfg;
        self.flush(cache)?;
        let feature = cache.feature.clone().context("the draft network has no context yet")?;
        // Changing the number of masks changes what a bidirectional block predicts, so always
        // run the whole block and use a prefix.
        let rows = c.block;
        let position = cache.count;
        let anchor = target.embed(&current.reshape(&[1, 1])?)?;
        let masks = broadcast_to(&self.mask, &[1, rows - 1, c.width])?;
        let mut x = concatenate(&[anchor, masks], 1)?;
        let (qw, kvw) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        for (i, layer) in self.layers.iter().enumerate() {
            let qkv = layer.qkv.forward(&model_rms(&x, &layer.input_norm, c.eps)?)?;
            let q = model_rms(&qkv.index((.., .., ..qw)).reshape(&[1, rows, c.heads, c.head_dim])?, &layer.query_norm, c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let q = fast::rope(&q, c.head_dim, false, c.theta, 1.0, position, None)?;
            let (mut k, mut v) = self.rotate(&qkv.index((.., .., qw..qw + kvw)), &qkv.index((.., .., qw + kvw..)), &layer.key_norm, position)?;
            if let Some((ck, cv)) = &cache.context[i] {
                k = concatenate(&[ck, &k], 2)?;
                v = concatenate(&[cv, &v], 2)?;
            }
            // No mask: every row of the block sees the context and the whole block.
            let attended = fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
            let attention = layer.output.forward(&attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, rows, qw])?)?;
            let residual = residual_sum(&x, &attention)?;
            let gate_up = layer.gate_up.forward(&model_rms(&residual, &layer.ff_norm, c.eps)?)?;
            let half = gate_up.dim(-1) / 2;
            let (gate, up) = (gate_up.index((.., .., ..half)), gate_up.index((.., .., half..)));
            x = residual_sum(&residual, &layer.down.forward(&nn::silu(&gate)?.multiply(&up)?)?)?;
        }
        // Row i predicts the token at its own position; the anchor row is unused.
        let n = (n as i32).min(rows - 1);
        let hidden = model_rms(&x, &self.norm, c.eps)?.index((0, 1..n + 1, ..)); // [n, W]
        let base = self.vocabulary.forward(&hidden)?; // [n, V]
        let mut state = zeros_dtype(&[1, c.state], Dtype::Bfloat16)?;
        let mut last = current.reshape(&[1])?;
        let (mut tokens, mut probs) = (Vec::with_capacity(n as usize), Vec::new());
        for i in 0..n {
            let previous = self.previous.take_axis(&last, 0)?;
            let input = concatenate(&[&state, &previous, &hidden.index((i..i + 1, ..)), &feature], -1)?;
            let gates = linear(&input, &self.recurrent_w)?.add(&self.recurrent_b)?;
            let keep = sigmoid(&gates.index((.., ..c.state)))?;
            let update = tanh(&gates.index((.., c.state..2 * c.state)))?;
            let latent = tanh(&gates.index((.., 2 * c.state..)))?;
            state = keep.multiply(&state)?.add(&Array::from_f32(1.0).subtract(&keep)?.multiply(&update)?)?.as_dtype(Dtype::Bfloat16)?;
            // The bf16 head rounds the sum before the f32 softcap.
            let corrected = base.index((i..i + 1, ..)).add(&self.folded.forward(&latent)?.add(&self.folded_bias)?)?;
            let scores = softcap(&f32a(&corrected.as_dtype(Dtype::Bfloat16)?)?, self.logit_cap)?;
            last = if temp > 0.0 {
                let scaled = scores.multiply(Array::from_f32(1.0 / temp))?;
                probs.push(softmax_axis(&scaled, -1, true)?);
                random::categorical(&scaled, -1, None, None)?.as_dtype(Dtype::Int32)?
            } else {
                argmax_axis(&scores, -1, None)?.as_dtype(Dtype::Int32)?
            };
            tokens.push(last.clone());
        }
        Ok(Proposal { tokens: concatenate(&tokens, 0)?, probs: if probs.is_empty() { None } else { Some(concatenate(&probs, 0)?) } })
    }
}

/// The reference normalizes the f32 sum but carries a bf16 residual into the next addition:
/// round the residual on reuse, and add in f32.
fn residual_sum(residual: &Array, update: &Array) -> Result<Array> {
    Ok(f32a(&residual.as_dtype(Dtype::Bfloat16)?)?.add(&f32a(update)?)?)
}
