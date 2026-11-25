//! Low-level pieces Inkling 2 needs that mlx-rs does not wrap: quantized expert products in the
//! NVFP4 and MXFP8 modes, the activation quantizers that feed them, and small fused Metal
//! kernels that replace chains of tiny graph ops (a decode step visits 48 layers, so launch
//! count, not arithmetic, sets the speed). The math follows ollama/mlxrunner/model/inkling2 and
//! ollama/mlx/nvfp4_activation.go.

use std::ffi::CString;

use anyhow::{Result, bail};
use mlx_rs::{Array, Dtype, Stream};
use mlx_sys as sys;

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("mlx-c call failed: {what}");
    }
    Ok(())
}

/// `gather_qmm(x, w, scales, rhs_indices, transpose=true)` in a microscaled `mode`
/// ("nvfp4": group 16, 4 bits; "mxfp8": group 32, 8 bits). mlx-rs only passes the affine mode.
/// `sorted`: the indices are in ascending order, so MLX can batch each expert's rows.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(x: &Array, w: &Array, scales: &Array, rhs: &Array, group: i32, bits: i32, mode: &str, sorted: bool) -> Result<Array> {
    let mode = CString::new(mode).unwrap();
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = sys::mlx_array_new();
        let none = sys::mlx_array_new();
        let rc = sys::mlx_gather_qmm(
            &mut res,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            none,
            none,
            rhs.as_ptr(),
            true,
            sys::mlx_optional_int { value: group, has_value: true },
            sys::mlx_optional_int { value: bits, has_value: true },
            mode.as_ptr(),
            sorted,
            stream.as_ptr(),
        );
        sys::mlx_array_free(none);
        if rc != 0 {
            sys::mlx_array_free(res);
            bail!("gather_qmm ({}) failed", mode.to_string_lossy());
        }
        Ok(Array::from_ptr(res))
    }
}

/// Fused attention with an additive mask and per-head sinks (mlx-rs has no `force_fused` and
/// passes sinks fine, but this keeps the call in one place).
pub fn sdpa(q: &Array, k: &Array, v: &Array, scale: f32, mask: &Array, sinks: &Array) -> Result<Array> {
    let mode = CString::new("array").unwrap();
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = sys::mlx_array_new();
        let rc = sys::mlx_fast_scaled_dot_product_attention(
            &mut res,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            scale,
            mode.as_ptr(),
            mask.as_ptr(),
            sinks.as_ptr(),
            false,
            stream.as_ptr(),
        );
        if rc != 0 {
            sys::mlx_array_free(res);
            bail!("scaled_dot_product_attention failed");
        }
        Ok(Array::from_ptr(res))
    }
}

/// `src[start..stop] = update` along every axis (in place when `src` is not shared).
pub fn slice_update(src: &Array, update: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides = vec![1i32; start.len()];
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = sys::mlx_array_new();
        let rc = sys::mlx_slice_update(
            &mut res,
            src.as_ptr(),
            update.as_ptr(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ptr(),
        );
        if rc != 0 {
            sys::mlx_array_free(res);
            bail!("slice_update failed");
        }
        Ok(Array::from_ptr(res))
    }
}

struct Kernel(sys::mlx_fast_metal_kernel);

// SAFETY: the kernel object is immutable after creation; MLX takes its own reference per launch.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

fn vec_string(names: &[&str]) -> sys::mlx_vector_string {
    unsafe {
        let v = sys::mlx_vector_string_new();
        for n in names {
            let c = CString::new(*n).unwrap();
            sys::mlx_vector_string_append_value(v, c.as_ptr());
        }
        v
    }
}

struct Launch<'a> {
    inputs: &'a [&'a Array],
    ints: &'a [(&'a str, i32)],
    dtypes: &'a [(&'a str, Dtype)],
    grid: (i32, i32, i32),
    group: (i32, i32, i32),
    outputs: &'a [(&'a [i32], Dtype)],
}

impl Kernel {
    fn new(name: &str, inputs: &[&str], outputs: &[&str], header: &str, source: &str) -> Self {
        let name = CString::new(name).unwrap();
        let source = CString::new(source).unwrap();
        let header = CString::new(header).unwrap();
        unsafe {
            let ins = vec_string(inputs);
            let outs = vec_string(outputs);
            let k = sys::mlx_fast_metal_kernel_new(name.as_ptr(), ins, outs, source.as_ptr(), header.as_ptr(), true, false);
            sys::mlx_vector_string_free(ins);
            sys::mlx_vector_string_free(outs);
            Self(k)
        }
    }

    fn run(&self, l: Launch) -> Result<Vec<Array>> {
        let stream = Stream::thread_local_or_default();
        unsafe {
            let cfg = sys::mlx_fast_metal_kernel_config_new();
            let result = (|| -> Result<Vec<Array>> {
                for (n, v) in l.ints {
                    let c = CString::new(*n).unwrap();
                    check(sys::mlx_fast_metal_kernel_config_add_template_arg_int(cfg, c.as_ptr(), *v), "template int")?;
                }
                for (n, d) in l.dtypes {
                    let c = CString::new(*n).unwrap();
                    check(sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, c.as_ptr(), *d as _), "template dtype")?;
                }
                check(sys::mlx_fast_metal_kernel_config_set_grid(cfg, l.grid.0, l.grid.1, l.grid.2), "grid")?;
                check(sys::mlx_fast_metal_kernel_config_set_thread_group(cfg, l.group.0, l.group.1, l.group.2), "threadgroup")?;
                for (shape, dtype) in l.outputs {
                    check(sys::mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.as_ptr(), shape.len(), *dtype as _), "output")?;
                }
                let ins = sys::mlx_vector_array_new();
                for a in l.inputs {
                    sys::mlx_vector_array_append_value(ins, a.as_ptr());
                }
                let mut outs = sys::mlx_vector_array_new();
                let rc = sys::mlx_fast_metal_kernel_apply(&mut outs, self.0, ins, cfg, stream.as_ptr());
                sys::mlx_vector_array_free(ins);
                check(rc, "metal kernel apply")?;
                let mut result = Vec::with_capacity(l.outputs.len());
                for i in 0..l.outputs.len() {
                    let mut y = sys::mlx_array_new();
                    sys::mlx_vector_array_get(&mut y, outs, i);
                    result.push(Array::from_ptr(y));
                }
                sys::mlx_vector_array_free(outs);
                Ok(result)
            })();
            sys::mlx_fast_metal_kernel_config_free(cfg);
            result
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) };
    }
}

/// Threadgroup size for the per-token reduction kernels over a row of `w` values: as wide as
/// the row divides evenly, so each thread does little serial work.
fn wide(w: i32) -> i32 {
    [1024, 768, 512].into_iter().find(|g| w % g == 0).unwrap_or(256)
}

/// For a pass over a few rows (a speculative round) the conv kernels return the state after
/// every row, so that rows can be taken back for free.
pub fn prefix_states(rows: i32) -> bool {
    (2..=8).contains(&rows)
}

/// Threadgroup size for a 1-D launch of `n` threads.
fn group(n: i32) -> i32 {
    n.clamp(1, 256)
}

const ROUNDING: &str = r#"
inline float round_e4m3(float x) {
    uint exponent = (as_type<uint>(x) >> 23) & 255u;
    float spacing = as_type<float>((max(exponent, 121u) - 3u) << 23);
    return rint(x / spacing) * spacing;
}

inline float round_e2m1(float x) {
    float m = abs(x);
    float q = m > 0.25f ? 0.5f : 0.0f;
    if (m >= 0.75f) q = 1.0f;
    if (m > 1.25f) q = 1.5f;
    if (m >= 1.75f) q = 2.0f;
    if (m > 2.5f) q = 3.0f;
    if (m >= 3.5f) q = 4.0f;
    if (m > 5.0f) q = 6.0f;
    return x < 0.0f ? -q : q;
}
"#;

// One threadgroup per row, one thread per 16-value block: the row maximum is reduced in the
// threadgroup, then each block picks the 4- or 6-range scale with the smaller L1 error.
const NVFP4_SOURCE: &str = r#"
#pragma clang fp contract(off) reassociate(off)
uint row = threadgroup_position_in_grid.y;
uint worker = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
const uint BPR = W / 16;
bool active = worker < BPR;
uint start = row * W + worker * 16;
float data[16];
float block_max = 0.0f;
for (uint i = 0; i < 16; ++i) {
    data[i] = active ? float(x[start + i]) : 0.0f;
    block_max = max(block_max, abs(data[i]));
}
threadgroup float partial[32];
float m = simd_max(block_max);
if (lane == 0) partial[simd] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_max = simd_max(lane < (G + 31) / 32 ? partial[lane] : 0.0f);
if (worker == 0) amax[row] = row_max;
if (!active) return;
float max_product = float(MaxProduct);
float normalizer = min(max_product / row_max, FLT_MAX);
float decoder = 1.0f / normalizer;
float outer = row_max * (1.0f / max_product);
float scale_a = round_e4m3(min(block_max * (normalizer * (1.0f / 6.0f)), 448.0f));
float scale_b = round_e4m3(min(block_max * (normalizer * 0.25f), 448.0f));
float inv_a = min(1.0f / (scale_a * decoder), FLT_MAX);
float inv_b = min(1.0f / (scale_b * decoder), FLT_MAX);
float candidate_a[16], candidate_b[16];
float error_a = 0.0f, error_b = 0.0f;
for (uint i = 0; i < 16; ++i) {
    candidate_a[i] = round_e2m1(data[i] * inv_a) * scale_a;
    candidate_b[i] = round_e2m1(data[i] * inv_b) * scale_b;
    error_a += abs(data[i] - candidate_a[i] * outer);
    error_b += abs(data[i] - candidate_b[i] * outer);
}
for (uint i = 0; i < 16; ++i) inner[start + i] = error_b < error_a ? candidate_b[i] : candidate_a[i];
"#;

// One thread per 32-value block: power-of-two scale, E4M3 values (the MX recipe).
const MXFP8_SOURCE: &str = r#"
#pragma clang fp contract(off) reassociate(off)
uint b = thread_position_in_grid.x;
if (b >= threads_per_grid.x) return;
float data[32];
float block_max = 0.0f;
for (uint i = 0; i < 32; ++i) {
    data[i] = float(x[b * 32 + i]);
    block_max = max(block_max, abs(data[i]));
}
float raw = block_max * (1.0f / 448.0f);
uint bits = as_type<uint>(raw);
uint exponent = bits >> 23;
exponent += (bits & 0x7fffffu) > 0u ? 1u : 0u;
exponent = min(exponent, 254u);
float minimum = as_type<float>(0x00400000u);
if (!(raw > minimum)) exponent = 0u;
float scale = max(as_type<float>(exponent << 23), minimum);
float inverse = exponent == 0u ? as_type<float>(0x7f000000u) : 1.0f / scale;
for (uint i = 0; i < 32; ++i) {
    float s = clamp(data[i] * inverse, -448.0f, 448.0f);
    out[b * 32 + i] = round_e4m3(s) * scale;
}
"#;

// Causal depthwise convolution over time with a scaled residual, from the carried state and
// this forward's inputs. x is read at columns XOFF.. of `xs` [B, T, *]; per-token dynamic taps
// (HEADS > 0) at columns DOFF.. of `ds`. Fixed taps are stored oldest first, dynamic ones by
// lag. Also writes the next state (the last TAPS-1 inputs). With RES, the output is the f32 sum
// of `res` and the (rounded) convolution, rounded to OT.
const CONV_SOURCE: &str = r#"
uint c = thread_position_in_grid.x;
uint r = thread_position_in_grid.y;
uint b = thread_position_in_grid.z;
const int C = w_shape[1];
const int T = xs_shape[1];
const int XS = xs_shape[2];
const int DS = ds_shape[2];
const int tail = TAPS - 1;
if (c >= uint(C)) return;
#define WIN(j) ((j) < tail ? float(state[(b * tail + (j)) * C + c]) : float(xs[(b * T + ((j) - tail)) * XS + XOFF + c]))
if (int(r) < T) {
    float sum = 0.0f;
    if (HEADS == 0) {
        for (int s = 0; s < TAPS; ++s) {
            float p = WIN(int(r) + s) * float(w[s * C + c]);
            sum = s == 0 ? p : sum + p;
        }
    } else {
        int h = c / (C / HEADS);
        for (int lag = 0; lag < TAPS; ++lag) {
            float coef = float(w[lag * C + c]) + float(ds[(b * T + r) * DS + DOFF + h * TAPS + lag]);
            float p = WIN(int(r) + tail - lag) * coef;
            sum = lag == 0 ? p : sum + p;
        }
    }
    sum = sum + float(xs[(b * T + r) * XS + XOFF + c]);
    const int terms = TAPS + 1 + (HEADS > 0 ? TAPS : 0);
    float y = sum * (1.0f / metal::precise::sqrt(float(terms)));
    XT rounded = static_cast<XT>(y);
    if (RES) {
        out[(b * T + r) * C + c] = static_cast<OT>(float(res[(b * T + r) * C + c]) + float(rounded));
    } else {
        out[(b * T + r) * C + c] = static_cast<OT>(rounded);
    }
}
if (PREFIX) {
    // The state after each row, [B, T, tail, C].
    if (int(r) < T) {
        for (int q = 0; q < tail; ++q) next_state[((b * T + r) * tail + q) * C + c] = static_cast<XT>(WIN(int(r) + 1 + q));
    }
} else if (int(r) < tail) {
    next_state[(b * tail + r) * C + c] = static_cast<XT>(WIN(T + int(r)));
}
"#;

// Additive attention mask [B, H, T, S]: the relative-position bias (r [B, T, H, RD] times column
// `distance` of pos [RD, R], rounded to RT; zero past the table), minus the token's sink logit
// (column SOFF + h of `qp`); -inf for future keys and keys outside the window. meta: query
// position, first key position, window, keys.
const MASK_SOURCE: &str = r#"
uint s = thread_position_in_grid.x;
uint th = thread_position_in_grid.y;
uint b = thread_position_in_grid.z;
const int T = r_shape[1];
const int H = r_shape[2];
const int RD = r_shape[3];
const int R = pos_shape[1];
const int QS = qp_shape[2];
const int S = meta[3];
if (int(s) >= S) return;
int t = th / H;
int h = th % H;
int d = meta[0] + t - (meta[1] + int(s));
bool visible = d >= 0 && (meta[2] == 0 || d < meta[2]);
float value = -INFINITY;
if (visible) {
    float bias = 0.0f;
    if (d < R) {
        const device auto* rr = r + ((b * T + t) * H + h) * RD;
        float acc = 0.0f;
        for (int j = 0; j < RD; ++j) acc += float(rr[j]) * float(pos[j * R + d]);
        bias = float(static_cast<RT>(acc));
    }
    value = bias - float(qp[(b * T + t) * QS + SOFF + h]);
}
out[((b * H + h) * T + t) * S + s] = value;
"#;

// Attention output gate: att [B, H, T, D] (f32) times sigmoid of the gate columns GOFF.. of
// `qp`, in f32, rounded to OT as [B, T, H*D].
const ATTN_GATE_SOURCE: &str = r#"
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
const int H = att_shape[1];
const int T = att_shape[2];
const int D = att_shape[3];
const int QS = qp_shape[2];
int j = i % (H * D);
int bt = i / (H * D);
int h = j / D;
int d = j % D;
int b = bt / T;
int t = bt % T;
float g = float(qp[bt * QS + GOFF + j]);
float sg = 1.0f / (1.0f + metal::precise::exp(-g));
out[i] = static_cast<OT>(att[((b * H + h) * T + t) * D + d] * sg);
"#;

// Expert routing, one thread per token: top-K of sigmoid(logit) + bias, weights
// softmax(log sigmoid(chosen logits)) * scale.
const ROUTE_SOURCE: &str = r#"
uint n = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
const int E = logits_shape[1];
const device float* l = logits + n * E;
const int per = (E + 31) / 32;
float sel[8];
for (int j = 0; j < per; ++j) {
    int e = lane + 32 * j;
    sel[j] = e < E ? 1.0f / (1.0f + metal::precise::exp(-l[e])) + bias[e] : -INFINITY;
}
float mine = -INFINITY;
for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = 0; j < per; ++j) {
        if (sel[j] > best) { best = sel[j]; bi = lane + 32 * j; }
    }
    float m = simd_max(best);
    int win = simd_min(best == m ? bi : 0x7fffffff);
    if (int(lane) == win % 32) sel[win / 32] = -INFINITY;
    if (lane == 0) indices[n * K + k] = uint(win);
    float c = l[win];
    // log sigmoid(c) = -softplus(-c)
    if (int(lane) == k) mine = -(max(-c, 0.0f) + metal::precise::log(1.0f + metal::precise::exp(-abs(c))));
}
float m = simd_max(mine);
float e = int(lane) < K ? metal::precise::exp(mine - m) : 0.0f;
float total = simd_sum(e);
if (int(lane) < K) weights[n * K + lane] = e / total * scale[0];
"#;

// The outer scale of route `route` (token n): activation amax times the expert's weight amax
// over the bank denominator; 1 for MXFP8 banks.
const OUTER_HEADER: &str = r#"
#define OUTER(route, n) (NV ? amax[PER_ROUTE ? (route) : (n)] * maxima[indices[(route)]] / denom[0] : 1.0f)
"#;

// Routed SwiGLU: the gate/up product (interleaved columns, times the route's outer scale and
// rounded to bf16) through softcapped gate * sigmoid(gate) * softcapped up, times the route
// weight; rounded to bf16.
const SWIGLU_SOURCE: &str = r#"
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
const int I = paired_shape[paired_ndim - 1] / 2;
const int K = indices_shape[1];
uint route = i / I;
uint j = i % I;
float o = OUTER(route, route / K);
float g = float(static_cast<bfloat16_t>(paired[route * 2 * I + 2 * j] * o));
float u = float(static_cast<bfloat16_t>(paired[route * 2 * I + 2 * j + 1] * o));
float gc = caps[0] > 0.0f ? metal::precise::tanh(g / caps[0]) * caps[0] : g;
float uc = caps[1] > 0.0f ? metal::precise::tanh(u / caps[1]) * caps[1] : u;
float sg = 1.0f / (1.0f + metal::precise::exp(-g));
out[i] = static_cast<bfloat16_t>(gc * sg * uc * weights[route]);
"#;

// Sum each token's routed down products (times their outer scales, rounded to bf16) in f32.
const EXPERT_SUM_SOURCE: &str = r#"
uint i = thread_position_in_grid.x;
if (i >= threads_per_grid.x) return;
const int K = indices_shape[1];
const int W = y_shape[y_ndim - 1];
uint n = i / W;
uint j = i % W;
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    uint route = n * K + k;
    sum += float(static_cast<bfloat16_t>(y[route * W + j] * OUTER(route, n)));
}
out[i] = static_cast<bfloat16_t>(sum);
"#;

// Vocabulary projection for one token: bf16 rows, f32 sums (one simdgroup per row), softcap.
const UNEMBED_SOURCE: &str = r#"
uint row = thread_position_in_grid.x / 32;
uint lane = thread_index_in_simdgroup;
const int V = w_shape[0];
const int W = w_shape[1];
if (int(row) >= V) return;
const device auto* wr = w + size_t(row) * W;
float sums[ROWS];
for (int r = 0; r < ROWS; ++r) sums[r] = 0.0f;
for (int i = lane * 4; i < W; i += 128) {
    float w0 = float(wr[i]), w1 = float(wr[i + 1]), w2 = float(wr[i + 2]), w3 = float(wr[i + 3]);
    for (int r = 0; r < ROWS; ++r) {
        const device float* xr = x + r * W + i;
        sums[r] += w0 * xr[0] + w1 * xr[1] + w2 * xr[2] + w3 * xr[3];
    }
}
for (int r = 0; r < ROWS; ++r) {
    float sum = simd_sum(sums[r]);
    if (lane == 0) out[size_t(r) * V + row] = cap[0] > 0.0f ? metal::precise::tanh(sum / cap[0]) * cap[0] : sum;
}
"#;

// Which route slots of a few tokens share an expert, [S, N + 1] int32 per slot: how many slots
// use its expert (0 unless this is the first slot that does), then those slots.
const SHARED_PLAN_SOURCE: &str = r#"
const int s = thread_position_in_grid.x;
const int S = N * K;
if (s >= S) return;
const uint e = uint(idx[s]);
bool first = true;
for (int j = 0; j < s; ++j) {
    if (uint(idx[j]) == e) first = false;
}
int nu = 0;
for (int j = s; j < S; ++j) {
    if (first && uint(idx[j]) == e && nu < N) {
        plan[s * (N + 1) + 1 + nu] = j;
        nu++;
    }
}
plan[s * (N + 1)] = nu;
for (int u = nu; u < N; ++u) plan[s * (N + 1) + 1 + u] = 0;
"#;

// Expert products for a few tokens whose routes share experts (see SHARED_PLAN_SOURCE). One
// simdgroup per route slot and output row. Only the first slot of each expert works: it decodes
// the expert's weights once and multiplies them with the input of every slot that uses it.
// x: [N * XPER, IN] f32 (XPER = 1: one input per token; K: one per route). w: [E, OUT, IN *
// BITS / 32] u32, packed as MLX packs "nvfp4" (4-bit E2M1 values, an E4M3 scale per 16) and
// "mxfp8" (E4M3 values, a power-of-two scale per 32). y: [N * K, OUT] f32.
const SHARED_QMM_HEADER: &str = r#"
inline float4 fp4x4(uint v) {
    uint4 n = (uint4(v) >> uint4(0, 4, 8, 12));
    float4 m = float4(as_type<half4>(ushort4((n & 7u) << 9u))) * 16384.0f;
    return select(m, -m, (n & 8u) != 0u);
}
inline float4 e4m3x4(uint v) {
    uint4 b = (uint4(v) >> uint4(0, 8, 16, 24));
    float4 m = float4(as_type<half4>(ushort4((b & 127u) << 7u))) * 256.0f;
    return select(m, -m, (b & 128u) != 0u);
}
inline float e4m3x1(uint b) {
    float m = float(as_type<half>(ushort((b & 127) << 7))) * 256.0f;
    return (b & 128) ? -m : m;
}
inline float e8m0x1(uint b) {
    return as_type<float>(b == 0 ? uint(0x400000) : (b << 23));
}
// RPS output rows of one expert against the inputs of the NU slots that use it, 16 values at a
// time: the weights are decoded once into registers and reused for every slot. NU and RPS are
// compile-time counts so the sums stay in registers too.
template <int NU, int RPS, int BITS, int GROUP>
inline void shared_rows(
    const device float* x, const device uint* wrow, const device uint8_t* srow, const device int* users,
    device float* y, int IN, int K, int XPER, int OUT, int o, uint lane) {
    const int C = IN / 16;
    const int WPR = IN * BITS / 32;
    const int SPR = IN / GROUP;
    float acc[RPS * NU];
    int xo[NU];
    for (int i = 0; i < RPS * NU; ++i) acc[i] = 0.0f;
    for (int u = 0; u < NU; ++u) xo[u] = (XPER == 1 ? users[u] / K : users[u]) * IN;
    for (int c = lane; c < C; c += 32) {
        float4 wv[RPS * 4];
        for (int r = 0; r < RPS; ++r) {
            const float sc = BITS == 4 ? e4m3x1(uint(srow[r * SPR + c * 16 / GROUP])) : e8m0x1(uint(srow[r * SPR + c * 16 / GROUP]));
            const device uint* wp = wrow + r * WPR + c * (16 * BITS / 32);
            if (BITS == 4) {
                const uint p0 = wp[0], p1 = wp[1];
                wv[r * 4] = fp4x4(p0) * sc;
                wv[r * 4 + 1] = fp4x4(p0 >> 16) * sc;
                wv[r * 4 + 2] = fp4x4(p1) * sc;
                wv[r * 4 + 3] = fp4x4(p1 >> 16) * sc;
            } else {
                for (int i = 0; i < 4; ++i) wv[r * 4 + i] = e4m3x4(wp[i]) * sc;
            }
        }
        for (int u = 0; u < NU; ++u) {
            const device packed_float4* xv = (const device packed_float4*)(x + xo[u] + c * 16);
            const float4 x0 = float4(xv[0]), x1 = float4(xv[1]), x2 = float4(xv[2]), x3 = float4(xv[3]);
            for (int r = 0; r < RPS; ++r) {
                acc[r * NU + u] += dot(wv[r * 4], x0) + dot(wv[r * 4 + 1], x1) + dot(wv[r * 4 + 2], x2) + dot(wv[r * 4 + 3], x3);
            }
        }
    }
    for (int r = 0; r < RPS; ++r) {
        for (int u = 0; u < NU; ++u) {
            float total = simd_sum(acc[r * NU + u]);
            if (lane == 0) y[size_t(users[u]) * OUT + o + r] = total;
        }
    }
}
"#;

const SHARED_QMM_SOURCE: &str = r#"
uint lane = thread_index_in_simdgroup;
const int o = (thread_position_in_grid.x / 32) * RPS;
const int s = thread_position_in_grid.y;
if (o >= OUT) return;
const device int* mine = plan + s * (N + 1);
const int nu = mine[0];
if (nu == 0) return;
const size_t row = size_t(uint(idx[s])) * OUT + o;
const device uint* wrow = (const device uint*)w + row * size_t(IN * BITS / 32);
const device uint8_t* srow = (const device uint8_t*)scales + row * size_t(IN / GROUP);
const device float* xf = (const device float*)x;
#define SHARED(c) case c: shared_rows<c, RPS, BITS, GROUP>(xf, wrow, srow, mine + 1, y, IN, K, XPER, OUT, o, lane); break;
switch (nu) {
    SHARED(1) SHARED(2) SHARED(3) SHARED(4) SHARED(5) SHARED(6) SHARED(7) SHARED(8)
}
"#;

// Attention over residual blocks for one token per threadgroup: sources s0.. ([N, W] each) are
// RMS-normalized (keys = values), scored against Q queries [Q, W] (pre-scaled), softmaxed over
// the sources and mixed. NORM: also RMS-normalize each mixture with weight nw [Q or 1, W] and
// epsilon p[1], rounded to bf16. p[0] is the residual epsilon.
const RESMIX_SOURCE: &str = r#"
uint n = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
const uint G = threads_per_threadgroup.x;
const int W = query_shape[query_ndim - 1];
#define SRC(s, w) ((s) == 0 ? float(s0[size_t(n) * W + (w)]) : (s) == 1 ? float(s1[size_t(n) * W + (w)]) : (s) == 2 ? float(s2[size_t(n) * W + (w)]) : (s) == 3 ? float(s3[size_t(n) * W + (w)]) : (s) == 4 ? float(s4[size_t(n) * W + (w)]) : float(s5[size_t(n) * W + (w)]))
threadgroup float red[32][32];
threadgroup float shared[32];

// 1. sum of squares per source
float acc[S];
for (int s = 0; s < S; ++s) acc[s] = 0.0f;
for (int w = tid; w < W; w += G) {
    for (int s = 0; s < S; ++s) {
        float v = SRC(s, w);
        acc[s] += v * v;
    }
}
for (int s = 0; s < S; ++s) {
    float t = simd_sum(acc[s]);
    if (lane == 0) red[simd][s] = t;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < S) {
    float t = 0.0f;
    for (uint g = 0; g < (G + 31) / 32; ++g) t += red[g][tid];
    shared[tid] = metal::precise::rsqrt(t / float(W) + p[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv[S];
for (int s = 0; s < S; ++s) inv[s] = shared[s];
threadgroup_barrier(mem_flags::mem_threadgroup);

// 2. scores per (query, source)
float prob[Q * S];
if (S == 1) {
    for (int q = 0; q < Q; ++q) prob[q] = 1.0f;
} else {
    float sc[Q * S];
    for (int i = 0; i < Q * S; ++i) sc[i] = 0.0f;
    for (int w = tid; w < W; w += G) {
        for (int s = 0; s < S; ++s) {
            float k = SRC(s, w) * inv[s];
            for (int q = 0; q < Q; ++q) sc[q * S + s] += k * query[q * W + w];
        }
    }
    for (int i = 0; i < Q * S; ++i) {
        float t = simd_sum(sc[i]);
        if (lane == 0) red[simd][i] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Q * S) {
        float t = 0.0f;
        for (uint g = 0; g < (G + 31) / 32; ++g) t += red[g][tid];
        shared[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < Q; ++q) {
        float m = -INFINITY;
        for (int s = 0; s < S; ++s) m = max(m, shared[q * S + s]);
        float total = 0.0f;
        for (int s = 0; s < S; ++s) total += metal::precise::exp(shared[q * S + s] - m);
        for (int s = 0; s < S; ++s) prob[q * S + s] = metal::precise::exp(shared[q * S + s] - m) / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 3. mix (and optionally normalize)
float vals[Q * 16];
float ss[Q];
for (int q = 0; q < Q; ++q) ss[q] = 0.0f;
int count = 0;
for (int w = tid; w < W; w += G) {
    for (int q = 0; q < Q; ++q) {
        float v = 0.0f;
        for (int s = 0; s < S; ++s) v += prob[q * S + s] * (SRC(s, w) * inv[s]);
        if (NORM) {
            ss[q] += v * v;
            vals[count * Q + q] = v;
        } else {
            out[(size_t(n) * Q + q) * W + w] = static_cast<OT>(v);
        }
    }
    count++;
}
if (NORM) {
    for (int q = 0; q < Q; ++q) {
        float t = simd_sum(ss[q]);
        if (lane == 0) red[simd][q] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Q) {
        float t = 0.0f;
        for (uint g = 0; g < (G + 31) / 32; ++g) t += red[g][tid];
        shared[tid] = metal::precise::rsqrt(t / float(W) + p[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    count = 0;
    for (int w = tid; w < W; w += G) {
        for (int q = 0; q < Q; ++q) {
            out[(size_t(n) * Q + q) * W + w] = static_cast<OT>(vals[count * Q + q] * shared[q] * nw[(NWQ ? q : 0) * W + w]);
        }
        count++;
    }
}
"#;

// Key and value preparation in one launch, one simdgroup per KV head: the short convolution of
// each (columns of ksrc / vsrc, with optional per-token dynamic taps), rounded to XT; then keys
// are L2-clipped to p[0] and values RMS-normalized (epsilon p[1]). Output kv [2, B, KV, T, D]
// f32 (XT-rounded) and the next conv states [2, B, TAPS-1, C].
const KV_SOURCE: &str = r#"
uint lane = thread_index_in_simdgroup;
uint head = thread_position_in_grid.x / 32;
uint r = thread_position_in_grid.y;
uint wb = thread_position_in_grid.z;
const int B = ksrc_shape[0];
const int T = ksrc_shape[1];
const int C = KV * D;
const int tail = TAPS - 1;
if (int(head) >= KV) return;
const int which = wb / B;
const int b = wb % B;
const int SS = which == 0 ? ksrc_shape[2] : vsrc_shape[2];
const int OFF = which == 0 ? KOFF : VOFF;
const int DOFF = which == 0 ? KDOFF : VDOFF;
#define SRC(idx) (which == 0 ? float(ksrc[idx]) : float(vsrc[idx]))
#define WIN(j, c) ((j) < tail ? float(state[((which * B + b) * tail + (j)) * C + (c)]) : SRC((b * T + ((j) - tail)) * SS + OFF + (c)))
const int per = (D + 31) / 32;
if (int(r) < T) {
    float vals[8];
    float partial = 0.0f;
    for (int i = 0; i < per; ++i) {
        int dd = lane + 32 * i;
        vals[i] = 0.0f;
        if (dd >= D) continue;
        int c = head * D + dd;
        float sum = 0.0f;
        if (DYN == 0) {
            for (int s = 0; s < TAPS; ++s) {
                float pr = WIN(int(r) + s, c) * w[(which * TAPS + s) * C + c];
                sum = s == 0 ? pr : sum + pr;
            }
        } else {
            for (int lag = 0; lag < TAPS; ++lag) {
                float coef = w[(which * TAPS + lag) * C + c] + SRC((b * T + r) * SS + DOFF + head * TAPS + lag);
                float pr = WIN(int(r) + tail - lag, c) * coef;
                sum = lag == 0 ? pr : sum + pr;
            }
        }
        sum = sum + SRC((b * T + r) * SS + OFF + c);
        const int terms = TAPS + 1 + (DYN ? TAPS : 0);
        float y = float(static_cast<XT>(sum * (1.0f / metal::precise::sqrt(float(terms)))));
        vals[i] = y;
        partial += y * y;
    }
    float total = simd_sum(partial);
    float factor = which == 0 ? min(p[0] / (metal::precise::sqrt(total) + 1e-6f), 1.0f)
                              : metal::precise::rsqrt(total / float(D) + p[1]);
    for (int i = 0; i < per; ++i) {
        int dd = lane + 32 * i;
        if (dd >= D) continue;
        kv[(((which * B + b) * KV + head) * T + r) * D + dd] = float(static_cast<XT>(vals[i] * factor));
    }
}
if (PREFIX) {
    // The state after each row, [2, B, T, tail, C].
    if (int(r) < T) {
        for (int i = 0; i < per; ++i) {
            int dd = lane + 32 * i;
            if (dd >= D) continue;
            int c = head * D + dd;
            for (int q = 0; q < tail; ++q) next_state[(((which * B + b) * T + r) * tail + q) * C + c] = static_cast<XT>(WIN(int(r) + 1 + q, c));
        }
    }
} else if (int(r) < tail) {
    for (int i = 0; i < per; ++i) {
        int dd = lane + 32 * i;
        if (dd >= D) continue;
        int c = head * D + dd;
        next_state[((which * B + b) * tail + r) * C + c] = static_cast<XT>(WIN(T + int(r), c));
    }
}
"#;

// Query and relative-position rows of `qp` in one launch (one simdgroup per head row): each
// times its per-head weight, L2-clipped (p[0] for queries, p[1] for relative rows), times the
// optional per-token temperature, rounded to RT. q: [B, H, T, D] f32; r: [B, T, H, RD] RT.
const QR_SOURCE: &str = r#"
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint t = thread_position_in_grid.y;
uint b = thread_position_in_grid.z;
const int T = qp_shape[1];
const int QS = qp_shape[2];
if (int(row) >= 2 * H) return;
const bool rel = int(row) >= H;
const int h = rel ? row - H : row;
const int dim = rel ? RD : D;
const int off = rel ? H * D + h * RD : h * D;
const device auto* x = qp + (b * T + t) * QS + off;
float partial = 0.0f;
for (int i = lane; i < dim; i += 32) {
    float v = float(x[i]) * (rel ? float(rw[h * RD + i]) : float(qw[h * D + i]));
    partial += v * v;
}
float total = simd_sum(partial);
float factor = min(p[rel ? 1 : 0] / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
float temp = HAS_TAU ? tau[t] : 1.0f;
for (int i = lane; i < dim; i += 32) {
    float v = float(x[i]) * (rel ? float(rw[h * RD + i]) : float(qw[h * D + i])) * factor;
    if (HAS_TAU) v = v * temp;
    if (rel) {
        r[((b * T + t) * H + h) * RD + i] = static_cast<RT>(v);
    } else {
        q[((b * H + h) * T + t) * D + i] = float(static_cast<RT>(v));
    }
}
"#;

// RMS-normalized (input | message) pairs for the loop controllers, one threadgroup per
// (gate, token): out [NG, N, 2W] bf16. Inputs are rows of `mixed`
// ([N, QI, W]; one shared row if QI == 1) and rows FIRST.. of `messages` [N, 4, W].
const PAIR_SOURCE: &str = r#"
uint gn = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
const uint G = threads_per_threadgroup.x;
const int W = messages_shape[messages_ndim - 1];
const int N = threads_per_grid.x / G / NG;
const int g = gn % NG;
const int n = gn / NG;
const device auto* in = mixed + (size_t(n) * QI + (QI == 1 ? 0 : g)) * W;
const device auto* msg = messages + (size_t(n) * 4 + FIRST + g) * W;
threadgroup float red[32][2];
threadgroup float shared[2];
float a = 0.0f, m = 0.0f;
for (int w = tid; w < W; w += G) {
    float x = float(in[w]);
    float y = float(msg[w]);
    a += x * x;
    m += y * y;
}
a = simd_sum(a);
m = simd_sum(m);
if (lane == 0) { red[simd][0] = a; red[simd][1] = m; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 2) {
    float t = 0.0f;
    for (uint s = 0; s < (G + 31) / 32; ++s) t += red[s][tid];
    shared[tid] = metal::precise::rsqrt(t / float(W) + p[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float ia = shared[0], im = shared[1];
for (int w = tid; w < W; w += G) {
    size_t at = (size_t(g) * N + n) * 2 * W + w;
    out[at] = static_cast<bfloat16_t>(float(in[w]) * ia);
    out[at + W] = static_cast<bfloat16_t>(float(msg[w]) * im);
}
"#;

// The loop-controller gate and the following input norm, one threadgroup per (gate, token):
// v = input * (1 - tanh(forget)) + message * tanh(gain), then RMS norm (epsilon p[0]) times
// nw [NG or 1, W], rounded to bf16. e0..e2 ([N, 2W] per gate) hold gain then forget.
const GATE_NORM_SOURCE: &str = r#"
uint gn = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
const uint G = threads_per_threadgroup.x;
const int W = messages_shape[messages_ndim - 1];
const int N = threads_per_grid.x / G / NG;
const int g = gn % NG;
const int n = gn / NG;
const device auto* in = mixed + (size_t(n) * QI + (QI == 1 ? 0 : g)) * W;
const device auto* msg = messages + (size_t(n) * 4 + FIRST + g) * W;
const device float* ex = (g == 0 ? e0 : g == 1 ? e1 : e2) + size_t(n) * 2 * W;
threadgroup float red[32];
threadgroup float shared[1];
float vals[16];
float ss = 0.0f;
int count = 0;
for (int w = tid; w < W; w += G) {
    float gain = metal::precise::tanh(ex[w]);
    float keep = 1.0f - metal::precise::tanh(ex[W + w]);
    float v = float(in[w]) * keep + float(msg[w]) * gain;
    vals[count++] = v;
    ss += v * v;
}
ss = simd_sum(ss);
if (lane == 0) red[simd] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float t = 0.0f;
    for (uint s = 0; s < (G + 31) / 32; ++s) t += red[s];
    shared[0] = metal::precise::rsqrt(t / float(W) + p[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = shared[0];
count = 0;
for (int w = tid; w < W; w += G) {
    out[(size_t(g) * N + n) * W + w] = static_cast<bfloat16_t>(vals[count++] * inv * nw[(NWQ ? g : 0) * W + w]);
}
"#;

pub struct Kernels {
    nvfp4: Kernel,
    mxfp8: Kernel,
    conv: Kernel,
    mask: Kernel,
    attn_gate: Kernel,
    route: Kernel,
    swiglu: Kernel,
    expert_sum: Kernel,
    unembed: Kernel,
    shared_plan: Kernel,
    shared_qmm: Kernel,
    resmix: Kernel,
    kv: Kernel,
    qr: Kernel,
    pair: Kernel,
    gate_norm: Kernel,
}

/// The activation scales of a bank's products, for the fused expert kernels.
pub struct Outer<'a> {
    /// NVFP4: activation amax (per token, or per route) and the bank's maxima and denominator.
    pub nvfp4: Option<(&'a Array, &'a Array, &'a Array)>,
    pub per_route: bool,
}

impl Kernels {
    pub fn new() -> Self {
        Self {
            nvfp4: Kernel::new("inkling_nvfp4", &["x"], &["inner", "amax"], ROUNDING, NVFP4_SOURCE),
            mxfp8: Kernel::new("inkling_mxfp8", &["x"], &["out"], ROUNDING, MXFP8_SOURCE),
            conv: Kernel::new("inkling_conv", &["state", "xs", "w", "ds", "res"], &["out", "next_state"], "", CONV_SOURCE),
            mask: Kernel::new("inkling_attn_mask", &["r", "pos", "qp", "meta"], &["out"], "", MASK_SOURCE),
            attn_gate: Kernel::new("inkling_attn_gate", &["att", "qp"], &["out"], "", ATTN_GATE_SOURCE),
            route: Kernel::new("inkling_route", &["logits", "bias", "scale"], &["indices", "weights"], "", ROUTE_SOURCE),
            swiglu: Kernel::new(
                "inkling_swiglu",
                &["paired", "indices", "amax", "maxima", "denom", "weights", "caps"],
                &["out"],
                OUTER_HEADER,
                SWIGLU_SOURCE,
            ),
            expert_sum: Kernel::new("inkling_expert_sum", &["y", "indices", "amax", "maxima", "denom"], &["out"], OUTER_HEADER, EXPERT_SUM_SOURCE),
            unembed: Kernel::new("inkling_unembed", &["x", "w", "cap"], &["out"], "", UNEMBED_SOURCE),
            shared_plan: Kernel::new("inkling_shared_plan", &["idx"], &["plan"], "", SHARED_PLAN_SOURCE),
            shared_qmm: Kernel::new("inkling_shared_qmm", &["x", "w", "scales", "idx", "plan"], &["y"], SHARED_QMM_HEADER, SHARED_QMM_SOURCE),
            resmix: Kernel::new("inkling_resmix", &["s0", "s1", "s2", "s3", "s4", "s5", "query", "nw", "p"], &["out"], "", RESMIX_SOURCE),
            kv: Kernel::new("inkling_kv", &["state", "ksrc", "vsrc", "w", "p"], &["kv", "next_state"], "", KV_SOURCE),
            qr: Kernel::new("inkling_qr", &["qp", "qw", "rw", "tau", "p"], &["q", "r"], "", QR_SOURCE),
            pair: Kernel::new("inkling_pair", &["mixed", "messages", "p"], &["out"], "", PAIR_SOURCE),
            gate_norm: Kernel::new("inkling_gate_norm", &["mixed", "messages", "e0", "e1", "e2", "nw", "p"], &["out"], "", GATE_NORM_SOURCE),
        }
    }

    /// NVFP4 "4 over 6" activation quantization with a per-row outer scale of amax / `max_product`.
    /// Returns the block-scaled values (f32, without the outer scale) and the row maxima [..., 1].
    pub fn nvfp4(&self, x: &Array, max_product: i32) -> Result<(Array, Array)> {
        let width = x.dim(-1);
        let rows = x.size() as i32 / width;
        let g = (width / 16 + 31) / 32 * 32;
        let mut amax_shape = x.shape().to_vec();
        *amax_shape.last_mut().unwrap() = 1;
        let mut out = self.nvfp4.run(Launch {
            inputs: &[x],
            ints: &[("W", width), ("G", g), ("MaxProduct", max_product)],
            dtypes: &[],
            grid: (g, rows, 1),
            group: (g, 1, 1),
            outputs: &[(x.shape(), Dtype::Float32), (&amax_shape, Dtype::Float32)],
        })?;
        let amax = out.pop().unwrap();
        Ok((out.pop().unwrap(), amax))
    }

    /// MXFP8 activation quantization, returned dequantized (f32).
    pub fn mxfp8(&self, x: &Array) -> Result<Array> {
        let blocks = x.size() as i32 / 32;
        Ok(self.mxfp8.run(Launch {
            inputs: &[x],
            ints: &[],
            dtypes: &[],
            grid: (blocks, 1, 1),
            group: (group(blocks), 1, 1),
            outputs: &[(x.shape(), Dtype::Float32)],
        })?.pop().unwrap())
    }

    /// Short convolution of x (columns `x_off..` of `xs` [B, T, *]) after `state` [B, taps-1, C].
    /// `w`: [taps, C] f32. `dynamic`: per-token taps at columns `off..` of a [B, T, *] tensor,
    /// `heads` heads (then `w` is indexed by lag). `residual`: add the result to it in f32 and
    /// round to that dtype. Returns (y [B, T, C], next state); for a few rows (`prefix_states`) the
    /// second is the state after each row, [B, T, taps-1, C].
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        state: &Array,
        xs: &Array,
        x_off: i32,
        dtype: Dtype,
        w: &Array,
        dynamic: Option<(&Array, i32, i32)>,
        residual: Option<(&Array, Dtype)>,
    ) -> Result<(Array, Array)> {
        let (b, t, c) = (xs.dim(0), xs.dim(1), w.dim(1));
        let taps = w.dim(0);
        let (ds, d_off, heads) = dynamic.unwrap_or((xs, 0, 0));
        let (res, out_dtype) = residual.unwrap_or((xs, dtype));
        let prefix = prefix_states(t);
        let states: &[i32] = if prefix { &[b, t, taps - 1, c] } else { &[b, taps - 1, c] };
        let mut out = self.conv.run(Launch {
            inputs: &[state, xs, w, ds, res],
            ints: &[("TAPS", taps), ("HEADS", heads), ("RES", residual.is_some() as i32), ("XOFF", x_off), ("DOFF", d_off), ("PREFIX", prefix as i32)],
            dtypes: &[("XT", dtype), ("OT", out_dtype)],
            grid: (c, t.max(taps - 1), b),
            group: (group(c), 1, 1),
            outputs: &[(&[b, t, c], out_dtype), (states, dtype)],
        })?;
        let next = out.pop().unwrap();
        Ok((out.pop().unwrap(), next))
    }

    /// Additive mask [B, H, T, S] from the clipped relative rows r [B, T, H, RD], the position
    /// projection pos [RD, R] and the sink columns `sink_off..` of `qp`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_mask(&self, r: &Array, pos: &Array, qp: &Array, sink_off: i32, q_pos: i32, k_pos: i32, window: i32, keys: i32) -> Result<Array> {
        let (b, t, h) = (r.dim(0), r.dim(1), r.dim(2));
        let meta = Array::from_slice(&[q_pos, k_pos, window, keys], &[4]);
        Ok(self.mask.run(Launch {
            inputs: &[r, pos, qp, &meta],
            ints: &[("SOFF", sink_off)],
            dtypes: &[("RT", r.dtype())],
            grid: (keys, t * h, b),
            group: (group(keys), 1, 1),
            outputs: &[(&[b, h, t, keys], Dtype::Float32)],
        })?.pop().unwrap())
    }

    /// Keys and values from their projections: conv (taps `w` [2, taps, C] f32, state
    /// [2, B, taps-1, C]), then key clip / value RMS norm. `k`/`v`: (tensor [B, T, *], column
    /// offset); `dynamic`: column offsets of the per-token taps in each. Returns
    /// (kv [2, B, KV, T, D] f32, next state); for a few rows (`prefix_states`) the second is the
    /// state after each row, [2, B, T, taps-1, C].
    #[allow(clippy::too_many_arguments)]
    pub fn kv(
        &self,
        state: &Array,
        k: (&Array, i32),
        v: (&Array, i32),
        w: &Array,
        dynamic: Option<(i32, i32)>,
        heads: i32,
        dim: i32,
        clip: f32,
        eps: f32,
    ) -> Result<(Array, Array)> {
        let (b, t) = (k.0.dim(0), k.0.dim(1));
        let taps = w.dim(1);
        let dtype = k.0.dtype();
        let p = Array::from_slice(&[clip, eps], &[2]);
        let (kd, vd) = dynamic.unwrap_or((0, 0));
        let prefix = prefix_states(t);
        let states: &[i32] = if prefix { &[2, b, t, taps - 1, heads * dim] } else { &[2, b, taps - 1, heads * dim] };
        let mut out = self.kv.run(Launch {
            inputs: &[state, k.0, v.0, w, &p],
            ints: &[
                ("TAPS", taps),
                ("KV", heads),
                ("D", dim),
                ("DYN", dynamic.is_some() as i32),
                ("KOFF", k.1),
                ("VOFF", v.1),
                ("KDOFF", kd),
                ("VDOFF", vd),
                ("PREFIX", prefix as i32),
            ],
            dtypes: &[("XT", dtype)],
            grid: (32 * heads, t.max(taps - 1), 2 * b),
            group: (group(32 * heads), 1, 1),
            outputs: &[(&[2, b, heads, t, dim], Dtype::Float32), (states, dtype)],
        })?;
        let next = out.pop().unwrap();
        Ok((out.pop().unwrap(), next))
    }

    /// Clipped queries [B, H, T, D] (f32) and relative rows [B, T, H, RD] (qp's dtype) from the
    /// query projection.
    #[allow(clippy::too_many_arguments)]
    pub fn qr(&self, qp: &Array, qw: &Array, rw: &Array, tau: Option<&Array>, heads: i32, dim: i32, q_clip: f32, r_clip: f32) -> Result<(Array, Array)> {
        let (b, t) = (qp.dim(0), qp.dim(1));
        let rd = rw.dim(1);
        let p = Array::from_slice(&[q_clip, r_clip], &[2]);
        let mut out = self.qr.run(Launch {
            inputs: &[qp, qw, rw, tau.unwrap_or(&p), &p],
            ints: &[("H", heads), ("D", dim), ("RD", rd), ("HAS_TAU", tau.is_some() as i32)],
            dtypes: &[("RT", qp.dtype())],
            grid: (64 * heads, t, b),
            group: (group(64 * heads), 1, 1),
            outputs: &[(&[b, heads, t, dim], Dtype::Float32), (&[b, t, heads, rd], qp.dtype())],
        })?;
        let r = out.pop().unwrap();
        Ok((out.pop().unwrap(), r))
    }

    /// Loop-controller inputs: [gates, N, 2W] bf16 of RMS-normalized (input | message) pairs.
    /// `mixed`: [B, T, W] (shared) or [B, T, 3, W]; `messages`: [B, T, 4, W], rows `first..`.
    pub fn pair(&self, mixed: &Array, messages: &Array, first: i32, gates: i32) -> Result<Array> {
        let w = messages.dim(-1);
        let n = messages.size() as i32 / (4 * w);
        let qi = mixed.size() as i32 / (n * w);
        let p = Array::from_slice(&[RESIDUAL_EPS], &[1]);
        Ok(self.pair.run(Launch {
            inputs: &[mixed, messages, &p],
            ints: &[("NG", gates), ("FIRST", first), ("QI", qi)],
            dtypes: &[],
            grid: (gates * n * wide(w), 1, 1),
            group: (wide(w), 1, 1),
            outputs: &[(&[gates, n, 2 * w], Dtype::Bfloat16)],
        })?.pop().unwrap())
    }

    /// Gate the inputs with their messages and RMS-normalize: [gates, N, W] bf16.
    /// `expansions`: [N, 2W] f32 per gate (at most 3); `nw`: norm weights [gates or 1, W].
    #[allow(clippy::too_many_arguments)]
    pub fn gate_norm(&self, mixed: &Array, messages: &Array, expansions: &[&Array], nw: &Array, eps: f32, first: i32, gates: i32) -> Result<Array> {
        let w = messages.dim(-1);
        if w > 16 * 256 || gates > 3 || expansions.len() != gates as usize {
            bail!("gate kernel supports widths up to 4096 and at most 3 gates");
        }
        let e = |i: usize| expansions[i.min(expansions.len() - 1)];
        let n = messages.size() as i32 / (4 * w);
        let qi = mixed.size() as i32 / (n * w);
        let nwq = (nw.ndim() == 2 && nw.dim(0) > 1) as i32;
        let p = Array::from_slice(&[eps], &[1]);
        Ok(self.gate_norm.run(Launch {
            inputs: &[mixed, messages, e(0), e(1), e(2), nw, &p],
            ints: &[("NG", gates), ("FIRST", first), ("QI", qi), ("NWQ", nwq)],
            dtypes: &[],
            grid: (gates * n * wide(w), 1, 1),
            group: (wide(w), 1, 1),
            outputs: &[(&[gates, n, w], Dtype::Bfloat16)],
        })?.pop().unwrap())
    }

    /// att [B, H, T, D] f32 times sigmoid(qp[..., gate_off..]) -> [B, T, H*D] in `out`.
    pub fn attn_gate(&self, att: &Array, qp: &Array, gate_off: i32, out: Dtype) -> Result<Array> {
        let (b, h, t, d) = (att.dim(0), att.dim(1), att.dim(2), att.dim(3));
        let n = b * h * t * d;
        Ok(self.attn_gate.run(Launch {
            inputs: &[att, qp],
            ints: &[("GOFF", gate_off)],
            dtypes: &[("OT", out)],
            grid: (n, 1, 1),
            group: (group(n), 1, 1),
            outputs: &[(&[b, t, h * d], out)],
        })?.pop().unwrap())
    }

    /// Top-`k` routing from f32 logits [N, E]: (indices [N, k] u32, weights [N, k] f32).
    pub fn route(&self, logits: &Array, bias: &Array, k: i32, scale: f32) -> Result<(Array, Array)> {
        let n = logits.dim(0);
        if logits.dim(1) > 256 || k > 32 {
            bail!("routing kernel supports at most 256 experts and 32 routes");
        }
        let scale = Array::from_slice(&[scale], &[1]);
        let mut out = self.route.run(Launch {
            inputs: &[logits, bias, &scale],
            ints: &[("K", k)],
            dtypes: &[],
            grid: (n * 32, 1, 1),
            group: (32, 1, 1),
            outputs: &[(&[n, k], Dtype::Uint32), (&[n, k], Dtype::Float32)],
        })?;
        let weights = out.pop().unwrap();
        Ok((out.pop().unwrap(), weights))
    }

    fn outer_inputs<'a>(o: &'a Outer<'a>, fallback: &'a Array) -> ([&'a Array; 3], i32) {
        match o.nvfp4 {
            Some((amax, maxima, denom)) => ([amax, maxima, denom], 1),
            None => ([fallback, fallback, fallback], 0),
        }
    }

    /// paired [N, K, 1, 2I] f32 gate/up products -> [N, K, 1, I] bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(&self, paired: &Array, indices: &Array, outer: &Outer, weights: &Array, gate_cap: f32, up_cap: f32) -> Result<Array> {
        let mut shape = paired.shape().to_vec();
        *shape.last_mut().unwrap() /= 2;
        let n: i32 = shape.iter().product();
        let caps = Array::from_slice(&[gate_cap, up_cap], &[2]);
        let ([amax, maxima, denom], nv) = Self::outer_inputs(outer, &caps);
        Ok(self.swiglu.run(Launch {
            inputs: &[paired, indices, amax, maxima, denom, weights, &caps],
            ints: &[("NV", nv), ("PER_ROUTE", outer.per_route as i32)],
            dtypes: &[],
            grid: (n, 1, 1),
            group: (group(n), 1, 1),
            outputs: &[(&shape, Dtype::Bfloat16)],
        })?.pop().unwrap())
    }

    /// y [N, K, ..., W] f32 down products -> [N, W] bf16.
    pub fn expert_sum(&self, y: &Array, indices: &Array, outer: &Outer) -> Result<Array> {
        let (n, w) = (y.dim(0), y.dim(-1));
        let ([amax, maxima, denom], nv) = Self::outer_inputs(outer, y);
        Ok(self.expert_sum.run(Launch {
            inputs: &[y, indices, amax, maxima, denom],
            ints: &[("NV", nv), ("PER_ROUTE", outer.per_route as i32)],
            dtypes: &[],
            grid: (n * w, 1, 1),
            group: (group(n * w), 1, 1),
            outputs: &[(&[n, w], Dtype::Bfloat16)],
        })?.pop().unwrap())
    }

    /// Which routes of a few tokens share an expert (indices [N, K]), for `shared_qmm`.
    pub fn shared_plan(&self, indices: &Array) -> Result<Array> {
        let (n, routes) = (indices.dim(0), indices.dim(1));
        Ok(self.shared_plan.run(Launch {
            inputs: &[indices],
            ints: &[("N", n), ("K", routes)],
            dtypes: &[],
            grid: (n * routes, 1, 1),
            group: (group(n * routes), 1, 1),
            outputs: &[(&[n * routes, n + 1], Dtype::Int32)],
        })?.pop().unwrap())
    }

    /// Expert products for a few tokens, decoding each distinct expert once. x: [N, 1 or K, 1, in]
    /// quantized activations (f32); indices [N, K] with their `shared_plan`. Returns
    /// [N, K, 1, out] f32, as `gather_qmm` does.
    #[allow(clippy::too_many_arguments)]
    pub fn shared_qmm(&self, x: &Array, w: &Array, scales: &Array, indices: &Array, plan: &Array, group: i32, bits: i32) -> Result<Array> {
        let (n, per, input) = (x.dim(0), x.dim(1), x.dim(-1));
        let (routes, out) = (indices.dim(-1), w.dim(1));
        // Output rows per simdgroup: more than two and the decoded weights no longer fit in
        // registers (eight is several times slower).
        let rps = 2;
        if w.dim(2) * 32 != input * bits || scales.dim(2) * group != input || input % 32 != 0 || out % rps != 0 || x.dtype() != Dtype::Float32 {
            bail!("shared_qmm: unexpected packed weights or input");
        }
        let y = self.shared_qmm.run(Launch {
            inputs: &[x, w, scales, indices, plan],
            ints: &[("N", n), ("K", routes), ("IN", input), ("OUT", out), ("GROUP", group), ("BITS", bits), ("XPER", per), ("RPS", rps)],
            dtypes: &[],
            grid: (out / rps * 32, n * routes, 1),
            group: (64, 1, 1),
            outputs: &[(&[n * routes, out], Dtype::Float32)],
        })?.pop().unwrap();
        Ok(y.reshape(&[n, routes, 1, out])?)
    }

    /// A few rows (at most 8) times a matrix: x [R, W] f32 against rows w [V, W] (bf16 or f32),
    /// f32 sums, softcap if `cap` > 0. The matrix is read once for all rows. Returns [R, V] f32.
    /// This is the vocabulary head's product, and the router's in a pass over a few tokens.
    pub fn unembed(&self, x: &Array, w: &Array, cap: f32) -> Result<Array> {
        let (r, v) = (x.dim(0), w.dim(0));
        let cap = Array::from_slice(&[cap], &[1]);
        Ok(self.unembed.run(Launch {
            inputs: &[x, w, &cap],
            ints: &[("ROWS", r)],
            dtypes: &[],
            grid: (v * 32, 1, 1),
            group: (256, 1, 1),
            outputs: &[(&[r, v], Dtype::Float32)],
        })?.pop().unwrap())
    }

    /// Residual-block attention: sources ([B, T, W] each, at most 6), queries [Q, W]
    /// (pre-scaled) -> [N, Q, W] f32, or with `norm` = (weight [Q or 1, W], eps) RMS-normalized
    /// and rounded to bf16.
    pub fn resmix(&self, sources: &[&Array], query: &Array, norm: Option<(&Array, f32)>) -> Result<Array> {
        let s = sources.len() as i32;
        let w = query.dim(-1);
        let n = sources[0].size() as i32 / w;
        let q = if query.ndim() == 1 { 1 } else { query.dim(0) };
        if s > 6 || q * s > 32 || w > 16 * 256 {
            bail!("residual mix kernel supports at most 6 sources, 32 query-source pairs and width 4096");
        }
        let (nw, eps) = norm.unwrap_or((query, 0.0));
        let nwq = (nw.ndim() == 2 && nw.dim(0) > 1) as i32;
        let p = Array::from_slice(&[RESIDUAL_EPS, eps], &[2]);
        let out_dtype = if norm.is_some() { Dtype::Bfloat16 } else { Dtype::Float32 };
        let src = |i: usize| sources[i.min(sources.len() - 1)];
        Ok(self.resmix.run(Launch {
            inputs: &[src(0), src(1), src(2), src(3), src(4), src(5), query, nw, &p],
            ints: &[("S", s), ("Q", q), ("NORM", norm.is_some() as i32), ("NWQ", nwq)],
            dtypes: &[("OT", out_dtype)],
            grid: (n * wide(w), 1, 1),
            group: (wide(w), 1, 1),
            outputs: &[(&[n, q, w], out_dtype)],
        })?.pop().unwrap())
    }
}

const RESIDUAL_EPS: f32 = 1e-6;

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}
