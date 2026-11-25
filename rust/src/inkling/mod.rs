//! Inkling 2 Mini (`inkling2_loop`): a 26B-total / 5B-active routed-expert model with relative
//! position attention, short convolutions, attention over residual blocks and a looped middle
//! section (layers 6..18 run three times). The image encoder (`vision.rs`) is loaded when the
//! first image is attached. The draft
//! network for speculative decoding is in `draft.rs`, and the decoder that uses it in
//! `speculate.rs`.
//!
//! Ported from Ollama's MLX runner (ollama/ollama, mlxrunner/model/inkling2, MIT License). The
//! hot elementwise chains run as small fused Metal kernels (`ops.rs`) and attention goes through
//! MLX's fused SDPA. Weights are read from Ollama's own store; see `ollama.rs`.

pub mod draft;
pub mod ollama;
pub mod ops;
pub mod speculate;
pub mod vision;

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use mlx_rs::fast;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp, take_along_axis};
use mlx_rs::ops::{arange, concatenate, split, stack, zeros, zeros_dtype};
use mlx_rs::{Array, Dtype};
use serde_json::Value;

use ollama::Manifest;
use ops::{Kernels, Outer, gather_qmm, sdpa, slice_update};

const RESIDUAL_EPS: f32 = 1e-6;
/// From this many tokens in a forward, expert routes are sorted so each expert runs as a batch.
const SORT_ROUTES_FROM: i32 = 16;
/// From this many tokens in a forward, the sorted expert products run in bf16 (Ollama's
/// threshold for its grouped products).
const BF16_PRODUCTS_FROM: i32 = 64;
/// Most tokens per prefill chunk.
const PREFILL_CHUNK: i32 = 1024;
/// The SGLang reference's outer scale for NVFP4 activations: row amax / 1536.
const NVFP4_DENOMINATOR: i32 = 1536;
/// Rows the vocabulary kernel scores in one pass over the head.
const UNEMBED_ROWS: i32 = 8;
/// Up to this many rows in a forward (a speculative round), the router multiplies one row at a
/// time and the expert products look for experts shared between rows.
const SHARED_EXPERT_ROWS: i32 = 8;

pub struct Config {
    num_layers: usize,
    local_layers: Vec<usize>,
    block_size: usize,
    loop_start: usize,
    loop_end: usize,
    passes: usize,
    hidden: i32,
    unpadded_vocab: i32,
    head_dim: i32,
    heads: i32,
    kv_heads: i32,
    rel_dim: i32,
    rel_range: i32,
    window: i32,
    taps: i32,
    experts: i32,
    top_k: i32,
    norm_eps: f32,
    qk_clip: f32,
    rel_clip: f32,
    log_floor: f32,
    log_alpha: f32,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
    logit_cap: f32,
}

impl Config {
    fn parse(root: &Value) -> Result<Self> {
        let t = &root["text_config"];
        let int = |k: &str| t[k].as_i64().with_context(|| format!("Inkling config missing {k}"));
        let float = |k: &str| t[k].as_f64().with_context(|| format!("Inkling config missing {k}")).map(|v| v as f32);
        let lp = &t["loop_config"];
        if lp["is_varloop"].as_bool().unwrap_or(false) {
            bail!("variable Inkling loops are not supported");
        }
        if int("swa_head_dim")? != int("head_dim")?
            || int("swa_num_attention_heads")? != int("num_attention_heads")?
            || int("swa_num_key_value_heads")? != int("num_key_value_heads")?
            || int("n_shared_experts")? != 0
        {
            bail!("unsupported Inkling attention or expert configuration");
        }
        let num_layers = int("num_hidden_layers")? as usize;
        Ok(Self {
            num_layers,
            local_layers: t["local_layer_ids"]
                .as_array()
                .context("local_layer_ids")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect(),
            block_size: int("n_layers_per_attn_res_block")? as usize,
            loop_start: lp["core_start"].as_u64().unwrap_or(0) as usize,
            loop_end: lp["core_end"].as_u64().unwrap_or(0) as usize,
            passes: lp["num_passes"].as_u64().unwrap_or(1) as usize,
            hidden: int("hidden_size")? as i32,
            unpadded_vocab: int("unpadded_vocab_size")? as i32,
            head_dim: int("head_dim")? as i32,
            heads: int("num_attention_heads")? as i32,
            kv_heads: int("num_key_value_heads")? as i32,
            rel_dim: int("d_rel")? as i32,
            rel_range: int("rel_extent")? as i32,
            window: int("sliding_window_size")? as i32,
            taps: int("sconv_kernel_size")? as i32,
            experts: int("n_routed_experts")? as i32,
            top_k: int("num_experts_per_tok")? as i32,
            norm_eps: float("rms_norm_eps")?,
            qk_clip: float("qk_l2_clip_threshold")?,
            rel_clip: float("rel_l2_clip_threshold")?,
            log_floor: float("log_scaling_n_floor")?,
            log_alpha: float("log_scaling_alpha")?,
            route_scale: float("route_scale")?,
            gate_cap: float("mlp_gate_softcap")?,
            up_cap: float("mlp_up_softcap")?,
            logit_cap: float("final_logit_softcapping")?,
        })
    }

    /// (layer, pass) in execution order: layers before the loop, the loop body `passes` times,
    /// then the rest.
    fn plan(&self) -> Vec<(usize, usize)> {
        let looped = self.passes > 1 && self.loop_start < self.loop_end;
        let end = if looped { self.loop_end } else { self.num_layers };
        let mut plan: Vec<(usize, usize)> = (0..end).map(|l| (l, 0)).collect();
        if looped {
            for pass in 1..self.passes {
                plan.extend((self.loop_start..self.loop_end).map(|l| (l, pass)));
            }
            plan.extend((self.loop_end..self.num_layers).map(|l| (l, 0)));
        }
        plan
    }
}

// ---------- small helpers ----------

/// Tokens per forward when `t` tokens are fed in the fewest equal chunks of at most `limit`.
fn chunk_size(t: i32, limit: i32) -> i32 {
    let chunks = ((t + limit - 1) / limit).max(1);
    (t + chunks - 1) / chunks
}

/// INKLING_TRACE=1: evaluate `arrays` now and charge the time since the last mark to `name`
/// (a breakdown of a long forward; forcing evaluation distorts short ones).
pub(crate) fn mark(name: &'static str, arrays: &[&Array]) -> Result<()> {
    use std::sync::{Mutex, OnceLock};
    static ON: OnceLock<bool> = OnceLock::new();
    static STATE: Mutex<(Option<std::time::Instant>, Vec<(&'static str, f64)>)> = Mutex::new((None, Vec::new()));
    if !*ON.get_or_init(|| std::env::var_os("INKLING_TRACE").is_some()) {
        return Ok(());
    }
    let mut state = STATE.lock().unwrap();
    if name == "report" {
        let total: f64 = state.1.iter().map(|x| x.1).sum();
        for (n, t) in &state.1 {
            eprintln!("  {n:<12} {:8.1} ms  {:4.1}%", t * 1e3, t / total * 100.0);
        }
        eprintln!("  {:<12} {:8.1} ms", "total", total * 1e3);
        *state = (None, Vec::new());
        return Ok(());
    }
    mlx_rs::transforms::eval(arrays.iter().copied())?;
    let now = std::time::Instant::now();
    if let (Some(last), false) = (state.0, name == "start") {
        let dt = (now - last).as_secs_f64();
        match state.1.iter_mut().find(|x| x.0 == name) {
            Some(x) => x.1 += dt,
            None => state.1.push((name, dt)),
        }
    }
    state.0 = Some(now);
    Ok(())
}

fn f32a(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

/// x[..., start:end]
fn cols(x: &Array, start: i32, end: i32) -> Array {
    x.index((Ellipsis, start..end))
}

/// x @ w^T
fn linear(x: &Array, w: &Array) -> Result<Array> {
    Ok(x.matmul(w.t())?)
}

/// RMS norm over the last axis in f32 with an f32 weight, rounded to bf16.
fn model_rms(x: &Array, w: &Array, eps: f32) -> Result<Array> {
    Ok(fast::rms_norm(f32a(x)?, Some(w), eps)?.as_dtype(Dtype::Bfloat16)?)
}

fn softcap(x: &Array, cap: f32) -> Result<Array> {
    if cap <= 0.0 {
        return Ok(x.clone());
    }
    Ok(mlx_rs::ops::tanh(x.divide(scalar(cap))?)?.multiply(scalar(cap))?)
}

/// A residual-mix query, pre-scaled by 1/sqrt(width) in f32.
fn mix_query(q: &Array) -> Result<Array> {
    Ok(f32a(q)?.multiply(scalar(1.0 / (q.dim(-1) as f32).sqrt()))?)
}

/// Attention across residual blocks, independently at each token. `query` (from `mix_query`) is
/// [W] (one mixture) or [Q, W] (one per row). Returns f32 [B, T, W] or [B, T, Q, W]; with `norm`
/// (weight [W] or [Q, W], epsilon) the mixtures are RMS-normalized and rounded to bf16.
fn mix_residuals(k: &Kernels, sources: &[&Array], query: &Array, norm: Option<(&Array, f32)>) -> Result<Array> {
    let (b, t, w) = (sources[0].dim(0), sources[0].dim(1), sources[0].dim(2));
    let out = k.resmix(sources, query, norm)?;
    Ok(if query.ndim() == 1 { out.reshape(&[b, t, w])? } else { out.reshape(&[b, t, query.dim(0), w])? })
}

// ---------- layers ----------

/// Causal depthwise convolution over time, with `taps` taps and a scaled residual.
struct Conv {
    /// [taps, channels] f32, oldest tap first.
    w: Array,
}

impl Conv {
    fn state(&self, x: &Array, state: &mut Option<Array>) -> Result<Array> {
        Ok(match state.take() {
            Some(s) => s,
            None => zeros_dtype(&[x.dim(0), self.w.dim(0) - 1, self.w.dim(1)], x.dtype())?,
        })
    }

    /// `slot`: which of the visit's conv states this one is.
    fn forward(&self, k: &Kernels, x: &Array, cache: &mut VisitCache, slot: usize) -> Result<Array> {
        let prev = self.state(x, &mut cache.conv[slot])?;
        let (y, next) = k.conv(&prev, x, 0, x.dtype(), &self.w, None, None)?;
        cache.set_conv(slot, next, x.dim(1), prev, vec![(x.clone(), 0)]);
        Ok(y)
    }

    /// (residual + conv(x)) in f32, rounded to `dtype`.
    fn forward_add(&self, k: &Kernels, x: &Array, residual: &Array, dtype: Dtype, cache: &mut VisitCache, slot: usize) -> Result<Array> {
        let prev = self.state(x, &mut cache.conv[slot])?;
        let (y, next) = k.conv(&prev, x, 0, x.dtype(), &self.w, None, Some((residual, dtype)))?;
        cache.set_conv(slot, next, x.dim(1), prev, vec![(x.clone(), 0)]);
        Ok(y)
    }
}

/// What gives a conv state after part of a forward is taken back.
enum Undo {
    /// The state after each row, [.., T, taps-1, C] (passes over a few rows).
    Each(Array),
    /// The state before the forward, and the rows the forward fed (columns `offset..` of a
    /// [B, T, *] tensor; the key rows then the value rows for the key/value conv).
    Rebuild { prev: Array, rows: Vec<(Array, i32)> },
}

impl Undo {
    /// The conv state after only the first `keep` rows: the last `tail` rows fed so far.
    fn state(&self, keep: i32) -> Result<Array> {
        let (prev, rows) = match self {
            Self::Each(states) => return Ok(row_state(states, keep - 1)),
            Self::Rebuild { prev, rows } => (prev, rows),
        };
        let (tail, width) = (prev.dim(-2), prev.dim(-1));
        let one = |prev: Array, (rows, off): &(Array, i32)| -> Result<Array> {
            let fed = rows.index((.., (keep - tail).max(0)..keep, *off..*off + width)).as_dtype(prev.dtype())?;
            Ok(if keep >= tail { fed } else { concatenate(&[prev.index((.., keep.., ..)), fed], 1)? })
        };
        if rows.len() == 1 {
            return one(prev.clone(), &rows[0]);
        }
        let parts: Vec<Array> = rows.iter().enumerate().map(|(i, r)| one(prev.index(i as i32), r)).collect::<Result<_>>()?;
        Ok(stack(&parts, 0)?)
    }
}

/// Row `row` of per-row conv states [.., T, taps-1, C].
fn row_state(states: &Array, row: i32) -> Array {
    if states.ndim() == 4 { states.index((.., row, .., ..)) } else { states.index((.., .., row, .., ..)) }
}

struct Attention {
    /// Rows read from the query input: query, relative, gate, sink.
    wq: Array,
    /// Key rows, then the dynamic key-tap rows (global layers).
    wk: Array,
    wv: Array,
    /// [wk; wv] for one shared key/value input (local layers).
    wkv: Array,
    wo: Array,
    /// [d_rel, range]
    pos_proj: Array,
    q_scale: Array,
    r_scale: Array,
    /// Key and value conv taps [2, taps, kv width] f32: oldest first (local layers), or by lag
    /// with per-token dynamic taps added (global layers).
    kv_conv: Array,
    /// Sliding window (local layers) or 0.
    window: i32,
}

/// One visit's attention and convolution state.
#[derive(Default)]
pub struct VisitCache {
    /// Keys and values [2, B, KV, capacity, D], f32 (bf16-rounded); `len` positions are filled.
    kv: Option<Array>,
    len: i32,
    /// Absolute position of the first cached key.
    first_key: i32,
    /// Conv states: keys and values [2, B, taps-1, kv width]; attention output; feed-forward output.
    conv: [Option<Array>; 3],
    /// How to take back rows of the latest forward, per conv state.
    undo: [Option<Undo>; 3],
}

/// KV buffers grow in steps of this many positions.
const KV_STEP: i32 = 256;

impl VisitCache {
    /// Append kv [2, B, KV, T, D]; local layers (`window` > 0) drop keys that no query can see
    /// any more. Returns views of the filled keys and values, [B, KV, len, D] each.
    fn append(&mut self, kv: &Array, offset: i32, window: i32) -> Result<(Array, Array)> {
        let (b, kvh, t, d) = (kv.dim(1), kv.dim(2), kv.dim(3), kv.dim(4));
        if self.kv.is_none() {
            self.first_key = offset;
            self.len = 0;
        }
        let cap = self.kv.as_ref().map_or(0, |a| a.dim(3));
        // A query at position p sees keys after p - window; the first query here is at `len`.
        let keep_from = if window > 0 && self.len + t > window + KV_STEP { (self.len - window).max(0) } else { 0 };
        if self.len + t > cap || keep_from > 0 {
            let kept = self.len - keep_from;
            let new_cap = (kept + t + KV_STEP - 1) / KV_STEP * KV_STEP + KV_STEP;
            let fresh = zeros::<f32>(&[2, b, kvh, new_cap, d])?;
            self.kv = Some(match &self.kv {
                Some(old) if kept > 0 => slice_update(&fresh, &old.index((.., .., .., keep_from..self.len, ..)), &[0, 0, 0, 0, 0], &[2, b, kvh, kept, d])?,
                _ => fresh,
            });
            self.first_key += keep_from;
            self.len = kept;
        }
        let buf = slice_update(self.kv.as_ref().unwrap(), kv, &[0, 0, 0, self.len, 0], &[2, b, kvh, self.len + t, d])?;
        self.len += t;
        let keys = buf.index((0, .., .., 0..self.len, ..));
        let values = buf.index((1, .., .., 0..self.len, ..));
        self.kv = Some(buf);
        Ok((keys, values))
    }

    /// Store a conv kernel's state output after a forward of `rows` rows, and what can take
    /// rows back: the per-row states if the kernel returned them, else `prev` and the rows fed.
    fn set_conv(&mut self, slot: usize, out: Array, rows: i32, prev: Array, fed: Vec<(Array, i32)>) {
        if ops::prefix_states(rows) {
            self.conv[slot] = Some(row_state(&out, rows - 1));
            self.undo[slot] = Some(Undo::Each(out));
        } else {
            self.conv[slot] = Some(out);
            self.undo[slot] = Some(Undo::Rebuild { prev, rows: fed });
        }
    }

    /// Keep only the first `keep` of the `rows` positions the latest forward added.
    fn rollback(&mut self, keep: i32, rows: i32) -> Result<()> {
        // The buffer may have been trimmed and rebased during the forward, so count from the end.
        self.len -= rows - keep;
        for (state, undo) in self.conv.iter_mut().zip(&self.undo) {
            *state = Some(undo.as_ref().context("no forward to take back")?.state(keep)?);
        }
        Ok(())
    }
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, c: &Config, kern: &Kernels, sinks: &Array, qin: &Array, kin: &Array, vin: &Array, cache: &mut VisitCache, offset: i32) -> Result<Array> {
        let (b, t) = (qin.dim(0), qin.dim(1));
        let (h, kvh, d) = (c.heads, c.kv_heads, c.head_dim);
        let width = h * d;
        let rel = h * c.rel_dim;
        let kvw = kvh * d;
        mark("mix+norm", &[qin, kin, vin])?;
        let qp = linear(qin, &self.wq)?;
        let state = match cache.conv[0].take() {
            Some(s) => s,
            None => zeros_dtype(&[2, b, c.taps - 1, kvw], qin.dtype())?,
        };
        // Per-token dynamic taps follow the key/value columns of a global layer's projections.
        let dynamic = (self.wk.dim(0) > kvw).then_some(kvw);
        let (kv, next, rows) = if std::ptr::eq(kin, vin) {
            let both = linear(kin, &self.wkv)?;
            let n = self.wk.dim(0);
            let (kv, next) = kern.kv(&state, (&both, 0), (&both, n), &self.kv_conv, dynamic.map(|x| (x, x + n)), kvh, d, c.qk_clip, c.norm_eps)?;
            (kv, next, vec![(both.clone(), 0), (both, n)])
        } else {
            let (kp, vp) = (linear(kin, &self.wk)?, linear(vin, &self.wv)?);
            let (kv, next) = kern.kv(&state, (&kp, 0), (&vp, 0), &self.kv_conv, dynamic.map(|x| (x, x)), kvh, d, c.qk_clip, c.norm_eps)?;
            (kv, next, vec![(kp, 0), (vp, 0)])
        };
        mark("qkv proj+conv", &[&qp, &kv, &next])?;
        cache.set_conv(0, next, t, state, rows);

        // Global layers sharpen queries past log_floor tokens: 1 + alpha * ln(max(n / floor, 1)).
        let tau = if self.window == 0 && c.log_floor > 0.0 && c.log_alpha != 0.0 && (offset + t) as f32 > c.log_floor {
            let n = f32a(&arange::<i32, i32>(offset + 1, offset + t + 1, None)?)?;
            let ratio = mlx_rs::ops::maximum(n.divide(scalar(c.log_floor))?, scalar(1.0))?;
            Some(ratio.log()?.multiply(scalar(c.log_alpha))?.add(scalar(1.0))?)
        } else {
            None
        };
        let (q, r) = kern.qr(&qp, &self.q_scale, &self.r_scale, tau.as_ref(), h, d, c.qk_clip, c.rel_clip)?;
        let (keys, values) = cache.append(&kv, offset, self.window)?;
        mark("qr+append", &[&q, &r, &keys, &values])?;
        let mask = kern.attn_mask(&r, &self.pos_proj, &qp, 2 * width + rel, offset, cache.first_key, self.window, keys.dim(2))?;
        mark("attn mask", &[&mask])?;
        let attended = sdpa(&q, &keys, &values, 1.0 / (d as f32).sqrt(), &mask, sinks)?;
        mark("sdpa", &[&attended])?;
        let gated = kern.attn_gate(&attended, &qp, width + rel, qin.dtype())?;
        let out = linear(&gated, &self.wo)?;
        mark("attn out", &[&out])?;
        Ok(out)
    }
}

/// A packed expert bank: NVFP4 (with per-expert amax) or MXFP8.
struct Bank {
    w: Array,
    scales: Array,
    /// NVFP4 only: per-expert weight amax, and the bank's outer-scale denominator times 1536.
    maxima: Option<Array>,
    denominator: Option<Array>,
}

impl Bank {
    /// x: [N, ., 1, in] -> raw products [N, K, 1, out] f32, and the activation amax (NVFP4).
    /// `sorted`: `indices` ascend, so each expert's rows are multiplied as one batch.
    /// `plan` (`Kernels::shared_plan`, for a few rows): which routes share an expert, so each
    /// expert is decoded once for all the rows that use it.
    fn project(&self, k: &Kernels, x: &Array, indices: &Array, sorted: bool, plan: Option<&Array>) -> Result<(Array, Option<Array>)> {
        let (values, amax) = self.quantize(k, x)?;
        Ok((self.product(k, &values, indices, sorted, plan)?, amax))
    }

    /// The activations as the bank's products take them (f32), and their row amax (NVFP4).
    fn quantize(&self, k: &Kernels, x: &Array) -> Result<(Array, Option<Array>)> {
        Ok(if self.maxima.is_some() {
            let (values, amax) = k.nvfp4(x, NVFP4_DENOMINATOR)?;
            (values, Some(amax))
        } else {
            (k.mxfp8(x)?, None)
        })
    }

    /// Quantized activations [N, ., 1, in] times the experts `indices` picks.
    fn product(&self, k: &Kernels, values: &Array, indices: &Array, sorted: bool, plan: Option<&Array>) -> Result<Array> {
        let (group, bits, mode) = if self.maxima.is_some() { (16, 4, "nvfp4") } else { (32, 8, "mxfp8") };
        match plan {
            Some(plan) => k.shared_qmm(&f32a(values)?, &self.w, &self.scales, indices, plan, group, bits),
            None => gather_qmm(values, &self.w, &self.scales, indices, group, bits, mode, sorted),
        }
    }

    fn outer<'a>(&'a self, amax: &'a Option<Array>, per_route: bool) -> Outer<'a> {
        let nvfp4 = match (amax, &self.maxima, &self.denominator) {
            (Some(a), Some(m), Some(d)) => Some((a, m, d)),
            _ => None,
        };
        Outer { nvfp4, per_route }
    }
}

struct Experts {
    router: Array,
    bias: Array,
    gate_up: Bank,
    down: Bank,
}

impl Experts {
    fn forward(&self, c: &Config, k: &Kernels, x: &Array) -> Result<Array> {
        let (b, t, w) = (x.dim(0), x.dim(1), x.dim(2));
        let n = b * t;
        let flat = x.reshape(&[n, w])?;
        let flat32 = f32a(&flat)?;
        let logits = if n > 1 && n <= SHARED_EXPERT_ROWS {
            // MLX's product of a few rows with this small matrix is ten times slower than its
            // product of one row; the few-row kernel (no softcap) does all rows in one launch.
            k.unembed(&flat32, &self.router, 0.0)?
        } else {
            linear(&flat32, &self.router)?
        };
        let (indices, weights) = k.route(&logits, &self.bias, c.top_k, c.route_scale)?;
        mark("route", &[&indices, &weights])?;
        if n >= SORT_ROUTES_FROM {
            return Ok(self.sorted(c, k, &flat, &indices, &weights)?.reshape(&[b, t, w])?);
        }
        // The rows of a speculative round often share experts.
        let plan = if n > 1 && n <= SHARED_EXPERT_ROWS { Some(k.shared_plan(&indices)?) } else { None };
        let (paired, amax) = self.gate_up.project(k, &flat.reshape(&[n, 1, 1, w])?, &indices, false, plan.as_ref())?;
        let activated = k.swiglu(&paired, &indices, &self.gate_up.outer(&amax, false), &weights, c.gate_cap, c.up_cap)?;
        let (down, amax) = self.down.project(k, &activated, &indices, false, plan.as_ref())?;
        let summed = k.expert_sum(&down, &indices, &self.down.outer(&amax, true), None, n)?;
        Ok(summed.reshape(&[b, t, w])?)
    }

    /// The same for many tokens (prefill): routes are sorted by expert so each expert multiplies
    /// all its tokens as one batch, then put back in token order for the sum.
    fn sorted(&self, c: &Config, k: &Kernels, flat: &Array, indices: &Array, weights: &Array) -> Result<Array> {
        let (n, w) = (flat.dim(0), flat.dim(1));
        let top = c.top_k;
        let routes = n * top;
        let flat_indices = indices.reshape(&[routes])?;
        let order = mlx_rs::ops::argsort(&flat_indices)?;
        let inverse = mlx_rs::ops::argsort(&order)?;
        let rows = order.floor_divide(Array::from_int(top))?;
        let sorted_indices = flat_indices.take_axis(&order, 0)?.reshape(&[routes, 1])?;
        let sorted_weights = weights.reshape(&[routes])?.take_axis(&order, 0)?;
        // The activations are quantized once per token, then copied to the token's routes.
        // Quantized activations are exact in bf16, which the batched products run fastest in;
        // like Ollama's grouped path, from BF16_PRODUCTS_FROM tokens.
        let bf16 = n >= BF16_PRODUCTS_FROM && std::env::var_os("INKLING_F32_PRODUCTS").is_none();
        let narrow = |x: Array| -> Result<Array> { Ok(if bf16 { x.as_dtype(Dtype::Bfloat16)? } else { x }) };
        let (values, amax) = self.gate_up.quantize(k, flat)?;
        let x = narrow(values)?.take_axis(&rows, 0)?.reshape(&[routes, 1, 1, w])?;
        let amax = match amax {
            Some(a) => Some(a.reshape(&[n])?.take_axis(&rows, 0)?),
            None => None,
        };
        mark("sort", &[&x, &sorted_indices, &sorted_weights, &inverse])?;
        let paired = self.gate_up.product(k, &x, &sorted_indices, true, None)?;
        mark("gate_up qmm", &[&paired])?;
        let activated = k.swiglu(&paired, &sorted_indices, &self.gate_up.outer(&amax, true), &sorted_weights, c.gate_cap, c.up_cap)?;
        mark("swiglu", &[&activated])?;
        let (values, amax) = self.down.quantize(k, &activated)?;
        let down = self.down.product(k, &narrow(values)?, &sorted_indices, true, None)?;
        mark("down qmm", &[&down])?;
        // The sum reads each token's routes where the sort put them.
        k.expert_sum(&down, &sorted_indices, &self.down.outer(&amax, true), Some(&inverse), n)
    }
}

/// A looped layer's gates between this pass's inputs and messages from the previous pass: three
/// for the attention query, key and value inputs, one for the feed-forward input.
struct MessageGates {
    /// Per gate: the down projection [rank, 2W] (bf16) and the gain-then-forget expansion
    /// [rank, 2W] (f32).
    down: Vec<Array>,
    gates: Vec<Array>,
}

impl MessageGates {
    /// Gate `mixed` ([B, T, W] shared by the gates, or [B, T, n, W]) with rows `first..first+n`
    /// of `messages` [B, T, 4, W], then RMS-normalize with `nw` ([n or 1, W]).
    /// Returns bf16 [n, B*T, W].
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, k: &Kernels, mixed: &Array, messages: &Array, first: i32, n: i32, nw: &Array, eps: f32) -> Result<Array> {
        let pairs = k.pair(mixed, messages, first, n)?; // [n, N, 2W]
        // One single-row product per gate: MLX's matrix-vector path is far faster than a small
        // batched product.
        let expansions: Vec<Array> = (0..n)
            .map(|g| {
                let i = (first + g) as usize;
                // The down product is rounded to bf16, as in the reference.
                Ok(f32a(&linear(&pairs.index(g), &self.down[i])?)?.matmul(&self.gates[i])?)
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&Array> = expansions.iter().collect();
        k.gate_norm(mixed, messages, &refs, nw, eps, first, n)
    }
}

struct Layer {
    attn: Attention,
    /// Query, key and value input norms (f32); local layers share one.
    attn_norm: [Array; 3],
    /// The three norms stacked [3, W], for the fused mix-and-normalize.
    attn_norms: Array,
    local: bool,
    /// Residual-mix query (scaled): [W] (local), [3, W] (global), none for layer 0.
    attn_query: Option<Array>,
    attn_out: Conv,
    ffn: Experts,
    ffn_norm: Array,
    ffn_query: Array,
    ffn_out: Conv,
    /// Looped layers: message queries [4, W] (q, k, v, feed-forward; scaled) and their gates.
    msg_queries: Option<Array>,
    msg_gates: Option<MessageGates>,
}

#[derive(Clone)]
struct Residual {
    blocks: Vec<Array>,
    partial: Array,
}

impl Residual {
    fn mix(&self, k: &Kernels, query: &Array, norm: Option<(&Array, f32)>) -> Result<Array> {
        let mut sources: Vec<&Array> = self.blocks.iter().collect();
        sources.push(&self.partial);
        mix_residuals(k, &sources, query, norm)
    }
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        c: &Config,
        k: &Kernels,
        sinks: &Array,
        state: Residual,
        messages: Option<&Array>,
        cache: &mut VisitCache,
        boundary: bool,
        offset: i32,
    ) -> Result<Residual> {
        let attention = self.attention(c, k, sinks, &state, messages, cache, offset)?;
        let mut sub = if boundary {
            let mut blocks = state.blocks.clone();
            blocks.push(state.partial.clone());
            Residual { blocks, partial: self.attn_out.forward(k, &attention, cache, 1)? }
        } else {
            // Within a block the feed-forward input sees the f32 sum.
            let partial = self.attn_out.forward_add(k, &attention, &state.partial, Dtype::Float32, cache, 1)?;
            Residual { blocks: state.blocks.clone(), partial }
        };
        let ffn_in = match (messages, &self.msg_gates) {
            (Some(m), Some(g)) => {
                let mixed = sub.mix(k, &self.ffn_query, None)?;
                g.apply(k, &mixed, m, 3, 1, &self.ffn_norm, c.norm_eps)?.reshape(mixed.shape())?
            }
            _ => sub.mix(k, &self.ffn_query, Some((&self.ffn_norm, c.norm_eps)))?,
        };
        mark("attn conv+mix", &[&ffn_in])?;
        let ffn = self.ffn.forward(c, k, &ffn_in)?;
        mark("expert sum", &[&ffn])?;
        sub.partial = self.ffn_out.forward_add(k, &ffn, &sub.partial, state.partial.dtype(), cache, 2)?;
        mark("ffn conv", &[&sub.partial])?;
        Ok(sub)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, c: &Config, k: &Kernels, sinks: &Array, state: &Residual, messages: Option<&Array>, cache: &mut VisitCache, offset: i32) -> Result<Array> {
        let query = match (state.blocks.is_empty(), &self.attn_query) {
            (false, Some(q)) => q,
            _ => {
                let x = model_rms(&state.partial, &self.attn_norm[0], c.norm_eps)?;
                return self.attn.forward(c, k, sinks, &x, &x, &x, cache, offset);
            }
        };
        let gates = match (messages, &self.msg_gates) {
            (Some(m), Some(g)) => Some((m, g)),
            _ => None,
        };
        let Some((m, g)) = gates else {
            // Mix and normalize in one launch.
            if query.ndim() == 1 {
                let x = state.mix(k, query, Some((&self.attn_norm[0], c.norm_eps)))?;
                return self.attn.forward(c, k, sinks, &x, &x, &x, cache, offset);
            }
            let x = state.mix(k, query, Some((&self.attn_norms, c.norm_eps)))?; // [B, T, 3, W]
            let qkv: Vec<Array> = (0..3).map(|i| x.index((Ellipsis, i, ..))).collect();
            return self.attn.forward(c, k, sinks, &qkv[0], &qkv[1], &qkv[2], cache, offset);
        };
        let mixed = state.mix(k, query, None)?;
        let (b, t, w) = (mixed.dim(0), mixed.dim(1), mixed.dim(-1));
        let gated = g.apply(k, &mixed, m, 0, 3, &self.attn_norms, c.norm_eps)?; // [3, B*T, W]
        let qkv: Vec<Array> = (0..3).map(|i| gated.index(i).reshape(&[b, t, w])).collect::<Result<_, _>>()?;
        self.attn.forward(c, k, sinks, &qkv[0], &qkv[1], &qkv[2], cache, offset)
    }
}

// ---------- the model ----------

pub struct Cache {
    visits: Vec<VisitCache>,
    offset: i32,
    /// Rows of the latest forward (what `rollback` can take back).
    rows: i32,
    /// A generated token that has been handed out but not fed yet; the next forward starts
    /// with it.
    pub(crate) pending: Option<i32>,
    /// The draft network's state (when the model has one).
    pub draft: Option<draft::Cache>,
    /// Encoded images ([tokens, W] each) waiting for their placeholder tokens: the next
    /// forward puts them in place of each run of image tokens, in order.
    images: std::collections::VecDeque<Array>,
}

impl Cache {
    /// Tokens processed so far.
    pub fn len(&self) -> usize {
        self.offset as usize
    }

    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Images attached and not yet fed.
    pub fn images(&self) -> usize {
        self.images.len()
    }

    /// Drop the images attached and not yet fed.
    pub fn drop_images(&mut self) {
        self.images.clear();
    }

    /// Keep only the first `keep` rows of the latest forward, as if the rest had never been
    /// fed. Valid once, straight after that forward.
    pub fn rollback(&mut self, keep: i32) -> Result<()> {
        let rows = self.rows;
        if keep < 1 || keep > rows {
            bail!("cannot keep {keep} of the last {rows} rows");
        }
        if keep < rows {
            for v in &mut self.visits {
                v.rollback(keep, rows)?;
            }
            self.offset -= rows - keep;
            self.rows = keep;
            if let Some(d) = &mut self.draft {
                d.keep_latest(keep)?;
            }
        }
        Ok(())
    }

    /// For checks: where this cache's state differs from another's ("" when identical).
    pub fn debug_diff(&self, other: &Cache) -> Result<String> {
        let mut out = if self.offset == other.offset { String::new() } else { format!("offset {} vs {}; ", self.offset, other.offset) };
        let maxabs = |a: &Array, b: &Array| -> Result<f32> { Ok(f32a(a)?.subtract(&f32a(b)?)?.abs()?.max(None)?.item::<f32>()) };
        for (i, (x, y)) in self.visits.iter().zip(&other.visits).enumerate() {
            let mut notes = Vec::new();
            if x.len != y.len || x.first_key != y.first_key {
                notes.push(format!("len {} vs {}, first {} vs {}", x.len, y.len, x.first_key, y.first_key));
            } else if let (Some(a), Some(b)) = (&x.kv, &y.kv) {
                let d = maxabs(&a.index((.., .., .., 0..x.len, ..)), &b.index((.., .., .., 0..y.len, ..)))?;
                if d > 0.0 {
                    notes.push(format!("kv {d:.4}"));
                }
            }
            for s in 0..3 {
                if let (Some(a), Some(b)) = (&x.conv[s], &y.conv[s]) {
                    let d = maxabs(a, b)?;
                    if d > 0.0 {
                        notes.push(format!("conv{s} {d:.4} shapes {:?} {:?}", a.shape(), b.shape()));
                    }
                }
            }
            if !notes.is_empty() {
                out.push_str(&format!("visit {i}: {}; ", notes.join(", ")));
                if out.len() > 600 {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Evaluate the cache's state now.
    pub fn eval_state(&self) -> Result<()> {
        Ok(mlx_rs::transforms::eval(self.state())?)
    }

    /// Everything that must be evaluated for the cache to stop referring to past graphs.
    pub(crate) fn state(&self) -> Vec<&Array> {
        let mut live: Vec<&Array> = self.visits.iter().flat_map(|v| v.kv.iter().chain(v.conv.iter().flatten())).collect();
        if let Some(d) = &self.draft {
            live.extend(d.state());
        }
        live
    }
}

pub struct Model {
    pub cfg: Config,
    plan: Vec<(usize, usize)>,
    embed: Array,
    embed_norm: Array,
    out_norm: Array,
    out_query: Array,
    /// [unpadded vocab, W] bf16, multiplied with f32 sums (the checkpoint asks for f32 output).
    unembed: Array,
    layers: Vec<Layer>,
    kernels: Kernels,
    /// Zero sink logits per head: the real sinks are folded into the attention mask.
    sinks: Array,
    /// The draft network that proposes tokens for speculative decoding, if loaded.
    pub draft: Option<draft::Draft>,
    /// Where the weights are, for the image encoder, which loads on first use.
    manifest: Manifest,
    vision: std::sync::OnceLock<vision::Vision>,
    /// The id of the image-placeholder token (set by whoever owns the tokenizer).
    pub image_token: Option<i32>,
}

struct Weights(HashMap<String, Array>);

impl Weights {
    fn take(&mut self, key: &str) -> Result<Array> {
        self.0.remove(key).with_context(|| format!("Inkling tensor {key} is missing"))
    }
    fn f32(&mut self, key: &str) -> Result<Array> {
        f32a(&self.take(key)?)
    }
}

impl Model {
    pub fn load(m: &Manifest) -> Result<Self> {
        let cfg = Config::parse(&m.json("config.json")?)?;
        let mut map = HashMap::new();
        for (name, path) in &m.files {
            // Text model only: skip the vision tower and the config files (the draft network
            // loads its own tensors).
            if name.starts_with("model.llm.") {
                map.extend(Array::load_safetensors(m.safetensors(path)?)?);
            }
        }
        let mut ws = Weights(map);
        let p = "model.llm.";
        let embed = ws.take(&format!("{p}embed.weight"))?;
        let unembed = concatenate(&[ws.take(&format!("{p}unembed.weight"))?.index((0..cfg.unpadded_vocab, ..))], 0)?;
        let embed_norm = ws.f32(&format!("{p}embed_norm.weight"))?;
        let out_norm = ws.f32(&format!("{p}norm.weight"))?;
        let out_query = mix_query(&ws.take(&format!("{p}attn_res.output_query"))?)?;
        let looped = cfg.passes > 1;
        let (cross_q, cross_kv) = if looped {
            (Some(ws.take(&format!("{p}middle_cycle_cross_pass_queries"))?), Some(ws.take(&format!("{p}middle_cycle_cross_pass_kv_queries"))?))
        } else {
            (None, None)
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = format!("{p}layers.{i}.");
            let local = cfg.local_layers.contains(&i);
            let a = format!("{lp}attn.");
            // Fixed taps arrive as [channels, 1, taps]; dynamic ones as [taps, channels] by lag.
            let taps = |ws: &mut Weights, name: &str, dynamic: bool| -> Result<Array> {
                if dynamic {
                    ws.f32(&format!("{name}.static_w"))
                } else {
                    Ok(ws.f32(&format!("{name}.weight"))?.squeeze_axes(&[1])?.t())
                }
            };
            let wq = concatenate(
                &[
                    ws.take(&format!("{a}wq_du.weight"))?,
                    ws.take(&format!("{a}wr_du.weight"))?,
                    ws.take(&format!("{a}gate_attn.weight"))?,
                    ws.take(&format!("{a}attn_sink_proj.weight"))?,
                ],
                0,
            )?;
            let (wk, wv) = if local {
                (ws.take(&format!("{a}wk_dv.weight"))?, ws.take(&format!("{a}wv_dv.weight"))?)
            } else {
                (
                    concatenate(&[ws.take(&format!("{a}wk_dv.weight"))?, ws.take(&format!("{a}k_sconv.weight_proj"))?], 0)?,
                    concatenate(&[ws.take(&format!("{a}wv_dv.weight"))?, ws.take(&format!("{a}v_sconv.weight_proj"))?], 0)?,
                )
            };
            let wkv = concatenate(&[&wk, &wv], 0)?;
            let attn = Attention {
                wq,
                wk,
                wv,
                wkv,
                wo: ws.take(&format!("{a}wo_ud.weight"))?,
                pos_proj: ws.take(&format!("{a}rel_logits_proj.proj"))?,
                q_scale: ws.take(&format!("{a}q_l2_clip.scale"))?,
                r_scale: ws.take(&format!("{a}rel_l2_clip.scale"))?,
                kv_conv: stack(&[taps(&mut ws, &format!("{a}k_sconv"), !local)?, taps(&mut ws, &format!("{a}v_sconv"), !local)?], 0)?,
                window: if local { cfg.window } else { 0 },
            };
            if attn.pos_proj.dim(-1) != if local { cfg.window } else { cfg.rel_range } {
                bail!("Inkling layer {i}: unexpected relative-position range");
            }
            let attn_norm = if local {
                let n = ws.f32(&format!("{lp}attn_norm.weight"))?;
                [n.clone(), n.clone(), n]
            } else {
                [
                    ws.f32(&format!("{lp}q_input_norm.weight"))?,
                    ws.f32(&format!("{lp}k_input_norm.weight"))?,
                    ws.f32(&format!("{lp}v_input_norm.weight"))?,
                ]
            };
            let attn_query = if i > 0 { Some(mix_query(&ws.take(&format!("{p}attn_res.attn_queries.{}", i - 1))?)?) } else { None };
            let bank = |ws: &mut Weights, name: &str| -> Result<Bank> {
                let w = ws.take(name)?;
                let scales = ws.take(&format!("{name}_scale"))?;
                let maxima = ws.0.remove(&format!("{name}_amax"));
                let denominator = match ws.0.remove(&format!("{name}_outer_scale_denominator")) {
                    Some(d) => Some(f32a(&d)?.multiply(scalar(NVFP4_DENOMINATOR as f32))?),
                    None => None,
                };
                Ok(Bank { w, scales, maxima, denominator })
            };
            let ffn = Experts {
                router: ws.f32(&format!("{lp}mlp.gate.weight"))?,
                bias: match ws.0.remove(&format!("{lp}mlp.gate.bias")) {
                    Some(b) => f32a(&b)?,
                    None => zeros::<f32>(&[cfg.experts])?,
                },
                gate_up: bank(&mut ws, &format!("{lp}mlp.experts.w13_weight"))?,
                down: bank(&mut ws, &format!("{lp}mlp.experts.w2_weight"))?,
            };
            if ffn.router.dim(0) != cfg.experts {
                bail!("Inkling layer {i}: expected {} experts", cfg.experts);
            }
            let (msg_queries, msg_gates) = match (&cross_q, &cross_kv) {
                (Some(cq), Some(ckv)) if i >= cfg.loop_start && i < cfg.loop_end => {
                    let row = 2 * (i - cfg.loop_start) as i32;
                    let queries = concatenate(
                        &[cq.index((row..row + 1, ..)), ckv.index((row..row + 2, ..)), cq.index((row + 1..row + 2, ..))],
                        0,
                    )?;
                    let (mut downs, mut gates) = (Vec::new(), Vec::new());
                    for name in ["attn", "k", "v", "mlp"] {
                        let stem = format!("{lp}controllers.{name}.");
                        let gain = ws.f32(&format!("{stem}gain_weight"))?;
                        let forget = ws.f32(&format!("{stem}forget_weight"))?;
                        downs.push(ws.take(&format!("{stem}down_weight"))?);
                        gates.push(concatenate(&[gain, forget], 0)?.t());
                    }
                    let gates_struct = MessageGates { down: downs, gates };
                    (Some(mix_query(&queries)?), Some(gates_struct))
                }
                _ => (None, None),
            };
            let attn_norms = stack(&attn_norm, 0)?;
            layers.push(Layer {
                attn,
                attn_norm,
                attn_norms,
                local,
                attn_query,
                attn_out: Conv { w: taps(&mut ws, &format!("{lp}attn_sconv"), false)? },
                ffn,
                ffn_norm: ws.f32(&format!("{lp}mlp_norm.weight"))?,
                ffn_query: mix_query(&ws.take(&format!("{p}attn_res.mlp_queries.{i}"))?)?,
                ffn_out: Conv { w: taps(&mut ws, &format!("{lp}mlp_sconv"), false)? },
                msg_queries,
                msg_gates,
            });
        }
        let plan = cfg.plan();
        let sinks = zeros::<f32>(&[cfg.heads])?;
        let mut model = Self { cfg, plan, embed, embed_norm, out_norm, out_query, unembed, layers, kernels: Kernels::new(), sinks, draft: None, manifest: m.clone(), vision: Default::default(), image_token: None };
        model.eval_weights()?;
        // INKLING_DRAFT=0 decodes one token per pass, without the draft network.
        if std::env::var("INKLING_DRAFT").map_or(true, |v| v != "0") && m.files.contains_key("draft/config.json") {
            model.draft = Some(draft::Draft::load(m, &model)?);
            let costs = speculate::calibrate(&model)?;
            if let Some(d) = &mut model.draft {
                d.round_costs = costs;
            }
        }
        Ok(model)
    }

    /// Materialize every weight now, so the first prompt does not pay for loading.
    fn eval_weights(&self) -> Result<()> {
        let mut all: Vec<&Array> = vec![&self.embed, &self.embed_norm, &self.out_norm, &self.out_query, &self.unembed, &self.sinks];
        for l in &self.layers {
            let a = &l.attn;
            all.extend([&a.wq, &a.wk, &a.wv, &a.wkv, &a.wo, &a.pos_proj, &a.q_scale, &a.r_scale, &a.kv_conv]);
            all.extend(l.attn_norm.iter());
            all.push(&l.attn_norms);
            all.extend(l.attn_query.iter());
            all.extend([&l.attn_out.w, &l.ffn_out.w, &l.ffn_norm, &l.ffn_query, &l.ffn.router, &l.ffn.bias]);
            for bank in [&l.ffn.gate_up, &l.ffn.down] {
                all.extend([&bank.w, &bank.scales]);
                all.extend(bank.maxima.iter().chain(bank.denominator.iter()));
            }
            all.extend(l.msg_queries.iter());
            if let Some(g) = &l.msg_gates {
                all.extend(g.down.iter().chain(g.gates.iter()));
            }
        }
        mlx_rs::transforms::eval(all)?;
        Ok(())
    }

    /// For checks: the largest difference between the shared-expert products and MLX's own, as a
    /// fraction of the largest product, over an NVFP4 and an MXFP8 layer and both banks, with
    /// every expert used by two of four rows.
    pub fn check_shared_products(&self) -> Result<f32> {
        let (c, k) = (&self.cfg, &self.kernels);
        let mut worst = 0.0f32;
        for layer in [7, self.layers.len() - 1] {
            let e = &self.layers[layer].ffn;
            let x = mlx_rs::random::normal::<f32>(&[4, 1, 1, c.hidden], None, None, None)?.as_dtype(Dtype::Bfloat16)?;
            let half: Vec<i32> = (0..2 * c.top_k).map(|i| (i * 5 + layer as i32) % c.experts).collect();
            let indices = Array::from_slice(&[half.clone(), half].concat(), &[4, c.top_k]).as_dtype(Dtype::Uint32)?;
            let plan = k.shared_plan(&indices)?;
            let mut input = x;
            for bank in [&e.gate_up, &e.down] {
                let (mine, _) = bank.project(k, &input, &indices, false, Some(&plan))?;
                let (theirs, _) = bank.project(k, &input, &indices, false, None)?;
                let diff = mine.subtract(&theirs)?.abs()?.max(None)?.item::<f32>() / theirs.abs()?.max(None)?.item::<f32>();
                worst = worst.max(diff);
                // The down bank takes one input per route, half as wide as a gate/up product.
                input = theirs.index((.., .., .., ..theirs.dim(-1) / 2)).multiply(scalar(1e-6))?.as_dtype(Dtype::Bfloat16)?;
            }
        }
        Ok(worst)
    }

    /// Latency of the pieces of one decode step: each is run as a chain of 48 launches where
    /// every launch waits for the previous one (as in a real step), minus the chain's own cost.
    pub fn microbench(&self) -> Result<()> {
        use std::time::Instant;
        let c = &self.cfg;
        let k = &self.kernels;
        let l = &self.layers[7];
        // INKLING_ROWS: tokens per step (a speculative round verifies several at once).
        let t: i32 = std::env::var("INKLING_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let rand = |shape: &[i32], dtype: Dtype| -> Result<Array> {
            let a = mlx_rs::random::normal::<f32>(shape, None, None, None)?.as_dtype(dtype)?;
            a.eval()?;
            Ok(a)
        };
        // f gets zeros (bf16 and f32 scalars) that depend on the previous launch's output.
        let chain = |f: &mut dyn FnMut(&Array, &Array) -> Result<Array>| -> Result<f64> {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut prev = Array::from_f32(0.0);
                for _ in 0..48 {
                    let zf = f32a(&prev.reshape(&[-1])?.index(0))?.multiply(scalar(0.0))?;
                    let zb = zf.as_dtype(Dtype::Bfloat16)?;
                    prev = f(&zb, &zf)?;
                }
                let t = Instant::now();
                prev.eval()?;
                best = best.min(t.elapsed().as_secs_f64() / 48.0);
            }
            Ok(best * 1e6)
        };
        let base = chain(&mut |zb, _| Ok(zb.clone()))?;
        println!("(chain overhead {base:.1} us)");
        let time = |name: &str, f: &mut dyn FnMut(&Array, &Array) -> Result<Array>| -> Result<()> {
            println!("{name:<30} {:7.1} us", chain(f)? - base);
            Ok(())
        };
        let (w, h, d, kvh) = (c.hidden, c.heads, c.head_dim, c.kv_heads);
        let bf = Dtype::Bfloat16;
        let x = rand(&[1, t, w], bf)?;
        time("add (one tiny kernel)", &mut |zb, _| Ok(x.add(zb)?))?;
        time("linear wq", &mut |zb, _| linear(&x.add(zb)?, &l.attn.wq))?;
        time("linear wkv", &mut |zb, _| linear(&x.add(zb)?, &l.attn.wkv))?;
        let qp = linear(&x, &l.attn.wq)?;
        let both = linear(&x, &l.attn.wkv)?;
        let st = zeros_dtype(&[2, 1, c.taps - 1, kvh * d], bf)?;
        time("kv kernel", &mut |zb, _| Ok(k.kv(&st, (&both.add(zb)?, 0), (&both, kvh * d), &l.attn.kv_conv, None, kvh, d, c.qk_clip, c.norm_eps)?.0))?;
        time("qr kernel", &mut |zb, _| Ok(k.qr(&qp.add(zb)?, &l.attn.q_scale, &l.attn.r_scale, None, h, d, c.qk_clip, c.rel_clip)?.0))?;
        let (q, r) = k.qr(&qp, &l.attn.q_scale, &l.attn.r_scale, None, h, d, c.qk_clip, c.rel_clip)?;
        let kv = k.kv(&st, (&both, 0), (&both, kvh * d), &l.attn.kv_conv, None, kvh, d, c.qk_clip, c.norm_eps)?.0;
        for keys in [64, 1024] {
            let soff = 2 * h * d + h * c.rel_dim;
            time(&format!("mask kernel ({keys} keys)"), &mut |zb, _| k.attn_mask(&r.add(zb)?, &l.attn.pos_proj, &qp, soff, keys - t, 0, c.window, keys))?;
            let mask = k.attn_mask(&r, &l.attn.pos_proj, &qp, soff, keys - t, 0, c.window, keys)?;
            let buf = rand(&[2, 1, kvh, keys + 256, d], Dtype::Float32)?;
            time(&format!("kv slice_update ({keys} keys)"), &mut |_, zf| slice_update(&buf, &kv.add(zf)?, &[0, 0, 0, keys - t, 0], &[2, 1, kvh, keys, d]))?;
            let (kk, vv) = (buf.index((0, .., .., 0..keys, ..)), buf.index((1, .., .., 0..keys, ..)));
            time(&format!("sdpa on views ({keys} keys)"), &mut |_, zf| sdpa(&q.add(zf)?, &kk, &vv, 0.088, &mask, &self.sinks))?;
        }
        let att = rand(&[1, h, t, d], Dtype::Float32)?;
        time("attn_gate kernel", &mut |_, zf| k.attn_gate(&att.add(zf)?, &qp, h * d + h * c.rel_dim, bf))?;
        let g = rand(&[1, t, l.attn.wo.dim(1)], bf)?;
        time("linear wo", &mut |zb, _| linear(&g.add(zb)?, &l.attn.wo))?;
        let cs = zeros_dtype(&[1, c.taps - 1, w], bf)?;
        time("conv kernel (+res)", &mut |zb, _| Ok(k.conv(&cs, &x.add(zb)?, 0, bf, &l.attn_out.w, None, Some((&x, bf)))?.0))?;
        let parts: Vec<Array> = (0..4).map(|_| rand(&[1, t, w], bf)).collect::<Result<_>>()?;
        time("resmix (4 src, norm)", &mut |zb, _| {
            let first = parts[0].add(zb)?;
            k.resmix(&[&first, &parts[1], &parts[2], &parts[3]], &l.ffn_query, Some((&l.ffn_norm, c.norm_eps)))
        })?;
        time("model_rms", &mut |zb, _| model_rms(&x.add(zb)?, &l.ffn_norm, c.norm_eps))?;
        let flat = rand(&[t, w], bf)?;
        time("router astype+matmul", &mut |zb, _| linear(&f32a(&flat.add(zb)?)?, &l.ffn.router))?;
        let logits = linear(&f32a(&flat)?, &l.ffn.router)?;
        time("route kernel", &mut |_, zf| Ok(k.route(&logits.add(zf)?, &l.ffn.bias, c.top_k, c.route_scale)?.1))?;
        let (indices, weights) = k.route(&logits, &l.ffn.bias, c.top_k, c.route_scale)?;
        let x4 = flat.reshape(&[t, 1, 1, w])?;
        time("nvfp4 quant", &mut |zb, _| Ok(k.nvfp4(&x4.add(zb)?, NVFP4_DENOMINATOR)?.0))?;
        let (values, amax) = k.nvfp4(&x4, NVFP4_DENOMINATOR)?;
        time("gather_qmm gate/up", &mut |_, zf| gather_qmm(&values.add(zf)?, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &indices, 16, 4, "nvfp4", false))?;
        let paired = gather_qmm(&values, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &indices, 16, 4, "nvfp4", false)?;
        if t > 1 && t <= SHARED_EXPERT_ROWS {
            let plan = k.shared_plan(&indices)?;
            time("shared_qmm gate/up", &mut |_, zf| k.shared_qmm(&values.add(zf)?, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &indices, &plan, 16, 4))?;
            let mine = k.shared_qmm(&values, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &indices, &plan, 16, 4)?;
            let scale = paired.abs()?.max(None)?.item::<f32>();
            println!("  (largest difference from gather_qmm {:.5} of {scale:.2})", mine.subtract(&paired)?.abs()?.max(None)?.item::<f32>());
            // Every row routed like the first: all experts shared.
            let same = mlx_rs::ops::broadcast_to(&indices.index((0..1, ..)), indices.shape())?;
            let plan = k.shared_plan(&same)?;
            time("gather_qmm, same experts", &mut |_, zf| gather_qmm(&values.add(zf)?, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &same, 16, 4, "nvfp4", false))?;
            time("shared_qmm, same experts", &mut |_, zf| k.shared_qmm(&values.add(zf)?, &l.ffn.gate_up.w, &l.ffn.gate_up.scales, &same, &plan, 16, 4))?;
        }
        let amax = Some(amax);
        let up = l.ffn.gate_up.outer(&amax, false);
        time("swiglu, then nvfp4", &mut |_, zf| {
            let a = k.swiglu(&paired.add(zf)?, &indices, &up, &weights, c.gate_cap, c.up_cap)?;
            Ok(k.nvfp4(&a, NVFP4_DENOMINATOR)?.0)
        })?;
        let act = k.swiglu(&paired, &indices, &up, &weights, c.gate_cap, c.up_cap)?;
        let (dv, damax) = k.nvfp4(&act, NVFP4_DENOMINATOR)?;
        time("gather_qmm down", &mut |_, zf| gather_qmm(&dv.add(zf)?, &l.ffn.down.w, &l.ffn.down.scales, &indices, 16, 4, "nvfp4", false))?;
        let down = gather_qmm(&dv, &l.ffn.down.w, &l.ffn.down.scales, &indices, 16, 4, "nvfp4", false)?;
        let damax = Some(damax);
        time("expert_sum kernel", &mut |_, zf| k.expert_sum(&down.add(zf)?, &indices, &l.ffn.down.outer(&damax, true), None, indices.dim(0)))?;
        time("experts (whole)", &mut |zb, _| l.ffn.forward(c, k, &x.add(zb)?))?;
        for stage in 1..=6 {
            time(&format!("experts up to stage {stage}"), &mut |zb, _| {
                let e = &l.ffn;
                let flat = x.add(zb)?.reshape(&[t, w])?;
                let logits = linear(&f32a(&flat)?, &e.router)?;
                if stage == 1 {
                    return Ok(logits);
                }
                let (indices, weights) = k.route(&logits, &e.bias, c.top_k, c.route_scale)?;
                if stage == 2 {
                    return Ok(weights);
                }
                let (values, amax) = k.nvfp4(&flat.reshape(&[t, 1, 1, w])?, NVFP4_DENOMINATOR)?;
                if stage == 3 {
                    return Ok(values.add(weights.sum(None)?)?);
                }
                let paired = gather_qmm(&values, &e.gate_up.w, &e.gate_up.scales, &indices, 16, 4, "nvfp4", false)?;
                if stage == 4 {
                    return Ok(paired);
                }
                let amax = Some(amax);
                let activated = k.swiglu(&paired, &indices, &e.gate_up.outer(&amax, false), &weights, c.gate_cap, c.up_cap)?;
                if stage == 5 {
                    return Ok(activated);
                }
                let (down, _) = e.down.project(k, &activated, &indices, false, None)?;
                Ok(down)
            })?;
        }
        if let (Some(g), Some(mq)) = (&l.msg_gates, &l.msg_queries) {
            let msgs = mix_residuals(k, &[&parts[0], &parts[1]], mq, None)?;
            let mixed = rand(&[1, t, w], Dtype::Float32)?;
            time("pair kernel x3", &mut |_, zf| k.pair(&mixed.add(zf)?, &msgs, 0, 3))?;
            time("message gates x3 (whole)", &mut |_, zf| g.apply(k, &mixed.add(zf)?, &msgs, 0, 3, &l.attn_norms, c.norm_eps))?;
            time("message gate ffn (whole)", &mut |_, zf| g.apply(k, &mixed.add(zf)?, &msgs, 3, 1, &l.ffn_norm, c.norm_eps))?;
        }
        let xf = rand(&[t, w], Dtype::Float32)?;
        time("unembed kernel", &mut |_, zf| k.unembed(&xf.add(zf)?, &self.unembed, c.logit_cap))?;
        Ok(())
    }

    pub fn make_cache(&self) -> Cache {
        Cache {
            visits: self.plan.iter().map(|_| VisitCache::default()).collect(),
            offset: 0,
            rows: 0,
            pending: None,
            draft: self.draft.as_ref().map(|d| d.make_cache()),
            images: Default::default(),
        }
    }

    /// tokens: [1, T] int32. Returns f32 logits for the last position: [1, vocab].
    /// Long inputs are fed in chunks, which keeps the attention masks and intermediates small.
    pub fn forward(&self, tokens: &Array, cache: &mut Cache) -> Result<Array> {
        let hidden = self.prefill(tokens, cache)?;
        self.logits(&hidden.index((.., hidden.dim(1) - 1.., ..)))
    }

    /// Feed tokens [1, T] (after any pending generated token); returns the last chunk's output
    /// rows [1, t, W].
    pub fn prefill(&self, tokens: &Array, cache: &mut Cache) -> Result<Array> {
        let joined;
        let tokens = match cache.pending.take() {
            Some(p) => {
                joined = concatenate(&[Array::from_slice(&[p], &[1, 1]), tokens.as_dtype(Dtype::Int32)?], 1)?;
                &joined
            }
            None => tokens,
        };
        let t = tokens.dim(1);
        // Equal chunks of at most PREFILL_CHUNK tokens: a pass costs a fixed amount (it reads
        // every expert), so a short last chunk would be expensive for what it feeds.
        let chunk = chunk_size(t, std::env::var("INKLING_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(PREFILL_CHUNK));
        // With images attached, embed the whole prompt first and put them in place.
        let embedded = if cache.images.is_empty() { None } else { Some(self.embed_with_images(tokens, cache)?) };
        let mut hidden = None;
        let mut start = 0;
        while start < t {
            let end = (start + chunk).min(t);
            let out = match &embedded {
                Some(e) => self.step_embedded(e.index((.., start..end, ..)), cache)?,
                None => self.step(&tokens.index((.., start..end)), cache)?,
            };
            if let (Some(d), Some(dc)) = (&self.draft, &mut cache.draft) {
                d.flush_if_full(dc)?;
            }
            if t > chunk {
                // Evaluate each chunk's state before building the next.
                let mut live = cache.state();
                live.push(&out);
                mlx_rs::transforms::eval(live)?;
            }
            hidden = Some(out);
            start = end;
        }
        hidden.context("no tokens to feed")
    }

    /// Logits [T, vocab] (f32, softcapped) for output rows [1, T, W].
    pub fn logits(&self, hidden: &Array) -> Result<Array> {
        let (t, w) = (hidden.dim(1), hidden.dim(2));
        let rows = f32a(&hidden.reshape(&[t, w])?)?;
        let k = &self.kernels;
        if t <= UNEMBED_ROWS {
            return k.unembed(&rows, &self.unembed, self.cfg.logit_cap);
        }
        let parts: Vec<Array> = (0..t)
            .step_by(UNEMBED_ROWS as usize)
            .map(|i| k.unembed(&rows.index((i..(i + UNEMBED_ROWS).min(t), ..)), &self.unembed, self.cfg.logit_cap))
            .collect::<Result<_>>()?;
        Ok(concatenate(&parts, 0)?)
    }

    /// The embedding of tokens [.., T]: [.., T, W] bf16.
    pub(crate) fn embed(&self, tokens: &Array) -> Result<Array> {
        model_rms(&self.embed.take_axis(tokens, 0)?, &self.embed_norm, self.cfg.norm_eps)
    }

    /// The image encoder, loaded on first use.
    pub fn vision(&self) -> Result<&vision::Vision> {
        if self.vision.get().is_none() {
            let _ = self.vision.set(vision::Vision::load(&self.manifest)?);
        }
        Ok(self.vision.get().expect("just set"))
    }

    /// Load the image encoder and run it once on a blank image, so that the first real image
    /// does not pay for reading the weights and compiling the kernels.
    pub fn warm_vision(&self) -> Result<()> {
        self.vision()?.warm()
    }

    /// Encode an image file and queue it for the next forward, whose prompt must hold a run
    /// of that many image-placeholder tokens. Returns the number of tokens.
    pub fn attach_image(&self, cache: &mut Cache, path: &std::path::Path) -> Result<usize> {
        let t0 = std::time::Instant::now();
        let loaded = self.vision.get().is_some();
        let vision = self.vision()?;
        let t1 = std::time::Instant::now();
        let pixels = vision.read(path)?;
        let t2 = std::time::Instant::now();
        cache.images.push_back(vision.encode(&pixels)?);
        if std::env::var_os("INKLING_STATS").is_some() {
            let load = if loaded { String::new() } else { format!("encoder loaded in {:.3}s, ", (t1 - t0).as_secs_f64()) };
            eprintln!("[image {}x{} tokens: {load}read {:.3}s, encode {:.3}s]", pixels.cols, pixels.rows, (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64());
        }
        Ok(pixels.tokens())
    }

    /// Embeddings of a prompt [1, T] with each run of image tokens replaced by the next
    /// queued image.
    fn embed_with_images(&self, tokens: &Array, cache: &mut Cache) -> Result<Array> {
        // Whatever happens, no image stays queued for a later prompt.
        let mut images = std::mem::take(&mut cache.images);
        let image_token = self.image_token.context("this model's tokenizer has no image token")?;
        let ids = tokens.as_dtype(Dtype::Int32)?;
        ids.eval()?;
        let ids = ids.as_slice::<i32>().to_vec();
        let mut embedded = self.embed(tokens)?;
        let (t, w) = (embedded.dim(1), embedded.dim(2));
        let mut i = 0;
        while i < ids.len() {
            if ids[i] != image_token {
                i += 1;
                continue;
            }
            let run = ids[i..].iter().take_while(|&&id| id == image_token).count();
            let image = images.pop_front().context("the prompt has more image placeholders than images attached")?;
            if image.dim(0) as usize != run {
                bail!("an image of {} tokens met a placeholder of {run}", image.dim(0));
            }
            let image = image.reshape(&[1, run as i32, w])?.as_dtype(embedded.dtype())?;
            embedded = slice_update(&embedded, &image, &[0, i as i32, 0], &[1, (i + run) as i32, w])?;
            i += run;
        }
        if !images.is_empty() {
            bail!("an attached image has no placeholder in the prompt ({t} tokens)");
        }
        Ok(embedded)
    }

    /// One forward over tokens [1, T]: returns the normalized output rows [1, T, W] (bf16). The
    /// cache can take back trailing rows afterwards (`Cache::rollback`). With a draft network,
    /// its input features for these rows are queued in the draft cache.
    pub fn step(&self, tokens: &Array, cache: &mut Cache) -> Result<Array> {
        self.step_embedded(self.embed(tokens)?, cache)
    }

    /// `step` from input embeddings [1, T, W].
    fn step_embedded(&self, embedded: Array, cache: &mut Cache) -> Result<Array> {
        let c = &self.cfg;
        let k = &self.kernels;
        let offset = cache.offset;
        let t = embedded.dim(1);
        let taps: &[usize] = self.draft.as_ref().map_or(&[], |d| d.taps());
        let mut features: Vec<Array> = Vec::with_capacity(taps.len());
        mark("start", &[&embedded])?;
        let mut state = Residual { blocks: Vec::new(), partial: embedded };
        let mut prelude: Option<Residual> = None;
        let mut previous: Vec<Array> = Vec::new();
        for (i, &(layer_index, pass)) in self.plan.iter().enumerate() {
            if c.passes > 1 && layer_index == c.loop_start {
                if pass == 0 {
                    prelude = Some(state.clone());
                } else {
                    let start = prelude.as_ref().unwrap().blocks.len() + 1;
                    previous = state.blocks[start..].to_vec();
                    previous.push(state.partial.clone());
                    state = prelude.clone().unwrap();
                }
            }
            if taps.get(features.len()) == Some(&i) {
                features.push(f32a(&state.partial)?);
            }
            let layer = &self.layers[layer_index];
            let messages = if pass > 0 {
                let sources: Vec<&Array> = previous.iter().collect();
                Some(mix_residuals(k, &sources, layer.msg_queries.as_ref().context("loop layer without message queries")?, None)?)
            } else {
                None
            };
            let boundary = layer_index % c.block_size == 0;
            state = layer.forward(c, k, &self.sinks, state, messages.as_ref(), &mut cache.visits[i], boundary, offset)?;
        }
        cache.offset += t;
        cache.rows = t;
        let hidden = state.mix(k, &self.out_query, Some((&self.out_norm, c.norm_eps)))?;
        if let Some(dc) = &mut cache.draft {
            // The last tap is the output itself.
            features.push(f32a(&hidden)?);
            dc.queue(concatenate(&features, -1)?);
        }
        mark("output", &[&hidden])?;
        mark("report", &[])?;
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_is_fed_in_equal_chunks() {
        assert_eq!(chunk_size(1, 1024), 1);
        assert_eq!(chunk_size(553, 1024), 553);
        assert_eq!(chunk_size(1024, 1024), 1024);
        // Not 1,024 and then a costly pass over 76 tokens.
        assert_eq!(chunk_size(1100, 1024), 550);
        assert_eq!(chunk_size(2049, 1024), 683);
        for t in 1..5000 {
            let chunk = chunk_size(t, 1024);
            assert!(chunk <= 1024 && (t + chunk - 1) / chunk == (t + 1023) / 1024, "{t} tokens in chunks of {chunk}");
        }
    }

    #[test]
    fn the_expert_sum_reads_routes_wherever_the_sort_put_them() {
        let k = Kernels::new();
        let (n, routes, w) = (3, 4, 6);
        let y: Vec<f32> = (0..n * routes * w).map(|i| ((i * 31 % 97) as f32 - 48.0) / 8.0).collect();
        let indices = Array::from_slice(&(0..(n * routes) as u32).map(|i| i * 7 % 5).collect::<Vec<_>>(), &[n, routes]);
        let plain = Outer { nvfp4: None, per_route: true };
        let in_order = k.expert_sum(&Array::from_slice(&y, &[n, routes, w]), &indices, &plain, None, n).unwrap();
        // The same rows, shuffled: row `at[r]` holds route r.
        let at: Vec<u32> = (0..(n * routes) as u32).map(|r| (r * 5 + 3) % (n * routes) as u32).collect();
        let mut shuffled = vec![0f32; y.len()];
        for (r, &row) in at.iter().enumerate() {
            shuffled[row as usize * w as usize..(row as usize + 1) * w as usize].copy_from_slice(&y[r * w as usize..(r + 1) * w as usize]);
        }
        let moved = k.expert_sum(&Array::from_slice(&shuffled, &[n * routes, w]), &indices, &plain, Some(&Array::from_slice(&at, &[n * routes])), n).unwrap();
        let read = |a: &Array| {
            let a = f32a(a).unwrap();
            a.eval().unwrap();
            a.as_slice::<f32>().to_vec()
        };
        assert_eq!(read(&in_order), read(&moved));
        // And it is the sum over each token's routes.
        let first: f32 = (0..routes as usize).map(|r| y[r * w as usize]).sum();
        assert!((read(&in_order)[0] - first).abs() < 0.05, "{} vs {first}", read(&in_order)[0]);
    }
}
