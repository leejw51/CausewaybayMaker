//! Whisper speech-to-text on MLX, ported from mlx_whisper/whisper.py (Apple, MIT).
//!
//! Loads the mlx-community `whisper-*` weights (safetensors or npz), encodes one
//! 30 s log-mel window and greedily decodes text without timestamps.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::nn::gelu;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::{concatenate_axis, conv1d};
use mlx_rs::{Array, Dtype};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::npy::load_npz;

#[derive(Debug, Clone)]
pub struct Dims {
    pub n_mels: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_vocab: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
}

impl Dims {
    fn from_config(path: &Path) -> Result<Self> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?).context("config.json")?;
        let g = |k: &str| v[k].as_i64().map(|x| x as i32).with_context(|| format!("config.json: missing {k}"));
        Ok(Self {
            n_mels: g("n_mels")?,
            n_audio_ctx: g("n_audio_ctx")?,
            n_audio_state: g("n_audio_state")?,
            n_audio_head: g("n_audio_head")?,
            n_audio_layer: g("n_audio_layer")?,
            n_vocab: g("n_vocab")?,
            n_text_ctx: g("n_text_ctx")?,
            n_text_state: g("n_text_state")?,
            n_text_head: g("n_text_head")?,
            n_text_layer: g("n_text_layer")?,
        })
    }
}

struct Weights(HashMap<String, Array>);

impl Weights {
    fn take(&mut self, key: &str) -> Result<Array> {
        let w = self.0.remove(key).with_context(|| format!("missing weight {key}"))?;
        Ok(w.as_dtype(Dtype::Float16)?)
    }
    fn take_opt(&mut self, key: &str) -> Result<Option<Array>> {
        self.0.remove(key).map(|w| Ok(w.as_dtype(Dtype::Float16)?)).transpose()
    }
}

struct Linear {
    w: Array,
    b: Option<Array>,
}

impl Linear {
    fn load(ws: &mut Weights, p: &str) -> Result<Self> {
        Ok(Self { w: ws.take(&format!("{p}.weight"))?, b: ws.take_opt(&format!("{p}.bias"))? })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = x.matmul(self.w.t())?;
        Ok(match &self.b {
            Some(b) => y.add(b)?,
            None => y,
        })
    }
}

struct LayerNorm {
    w: Array,
    b: Array,
}

impl LayerNorm {
    fn load(ws: &mut Weights, p: &str) -> Result<Self> {
        Ok(Self { w: ws.take(&format!("{p}.weight"))?, b: ws.take(&format!("{p}.bias"))? })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(x, &self.w, &self.b, 1e-5)?)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: i32,
}

impl Attention {
    fn load(ws: &mut Weights, p: &str, n_head: i32) -> Result<Self> {
        Ok(Self {
            query: Linear::load(ws, &format!("{p}.query"))?,
            key: Linear::load(ws, &format!("{p}.key"))?,
            value: Linear::load(ws, &format!("{p}.value"))?,
            out: Linear::load(ws, &format!("{p}.out"))?,
            n_head,
        })
    }

    fn kv(&self, x: &Array) -> Result<(Array, Array)> {
        Ok((self.key.forward(x)?, self.value.forward(x)?))
    }

    /// Attention of `x` over keys/values `(k, v)`; `causal` masks future positions.
    fn attend(&self, x: &Array, k: &Array, v: &Array, causal: bool) -> Result<Array> {
        let q = self.query.forward(x)?;
        let (b, l, d) = (q.dim(0), q.dim(1), q.dim(2));
        let heads = |a: &Array| -> Result<Array> {
            Ok(a.reshape(&[b, a.dim(1), self.n_head, -1])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let scale = 1.0 / ((d / self.n_head) as f32).sqrt();
        let mask = causal.then_some(ScaledDotProductAttentionMask::Causal);
        let o = fast::scaled_dot_product_attention(heads(&q)?, heads(k)?, heads(v)?, scale, mask, None)?;
        self.out.forward(&o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, d])?)
    }
}

#[derive(Default)]
pub struct LayerCache {
    /// Self-attention keys/values of all tokens decoded so far.
    kv: Option<(Array, Array)>,
    /// Cross-attention keys/values of the encoder output (computed once).
    cross_kv: Option<(Array, Array)>,
}

struct Block {
    attn: Attention,
    attn_ln: LayerNorm,
    cross_attn: Option<(Attention, LayerNorm)>,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

impl Block {
    fn load(ws: &mut Weights, p: &str, n_head: i32, cross: bool) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(ws, &format!("{p}.attn"), n_head)?,
            attn_ln: LayerNorm::load(ws, &format!("{p}.attn_ln"))?,
            cross_attn: cross
                .then(|| -> Result<_> {
                    Ok((
                        Attention::load(ws, &format!("{p}.cross_attn"), n_head)?,
                        LayerNorm::load(ws, &format!("{p}.cross_attn_ln"))?,
                    ))
                })
                .transpose()?,
            mlp1: Linear::load(ws, &format!("{p}.mlp1"))?,
            mlp2: Linear::load(ws, &format!("{p}.mlp2"))?,
            mlp_ln: LayerNorm::load(ws, &format!("{p}.mlp_ln"))?,
        })
    }

    /// Encoder block: full self-attention, no cache.
    fn encode(&self, x: &Array) -> Result<Array> {
        let h = self.attn_ln.forward(x)?;
        let (k, v) = self.attn.kv(&h)?;
        let x = x.add(self.attn.attend(&h, &k, &v, false)?)?;
        self.mlp(&x)
    }

    /// Decoder block: cached causal self-attention, then cross-attention over `xa`.
    fn decode(&self, x: &Array, xa: &Array, cache: &mut LayerCache) -> Result<Array> {
        let h = self.attn_ln.forward(x)?;
        let (k, v) = self.attn.kv(&h)?;
        let (k, v) = match cache.kv.take() {
            Some((pk, pv)) => (concatenate_axis(&[&pk, &k], 1)?, concatenate_axis(&[&pv, &v], 1)?),
            None => (k, v),
        };
        let mut x = x.add(self.attn.attend(&h, &k, &v, x.dim(1) > 1)?)?;
        cache.kv = Some((k, v));

        if let Some((cross, ln)) = &self.cross_attn {
            let (ck, cv) = match cache.cross_kv.take() {
                Some(kv) => kv,
                None => cross.kv(xa)?,
            };
            x = x.add(cross.attend(&ln.forward(&x)?, &ck, &cv, false)?)?;
            cache.cross_kv = Some((ck, cv));
        }
        self.mlp(&x)
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        let h = gelu(self.mlp1.forward(&self.mlp_ln.forward(x)?)?)?;
        Ok(x.add(self.mlp2.forward(&h)?)?)
    }
}

/// Fixed sinusoidal positions for the encoder, (length, channels).
fn sinusoids(length: i32, channels: i32) -> Array {
    let half = (channels / 2) as usize;
    let inc = (10000f64).ln() / (half as f64 - 1.0);
    let mut out = vec![0f32; length as usize * channels as usize];
    for t in 0..length as usize {
        for i in 0..half {
            let v = t as f64 * (-inc * i as f64).exp();
            out[t * channels as usize + i] = v.sin() as f32;
            out[t * channels as usize + half + i] = v.cos() as f32;
        }
    }
    Array::from_slice(&out, &[length, channels])
}

struct Encoder {
    conv1: Linear,
    conv2: Linear,
    pos: Array,
    blocks: Vec<Block>,
    ln_post: LayerNorm,
}

impl Encoder {
    fn load(ws: &mut Weights, d: &Dims) -> Result<Self> {
        Ok(Self {
            conv1: Linear::load(ws, "encoder.conv1")?,
            conv2: Linear::load(ws, "encoder.conv2")?,
            pos: sinusoids(d.n_audio_ctx, d.n_audio_state).as_dtype(Dtype::Float16)?,
            blocks: (0..d.n_audio_layer)
                .map(|i| Block::load(ws, &format!("encoder.blocks.{i}"), d.n_audio_head, false))
                .collect::<Result<_>>()?,
            ln_post: LayerNorm::load(ws, "encoder.ln_post")?,
        })
    }

    /// `mel`: (1, 3000, n_mels) -> (1, 1500, n_audio_state)
    fn forward(&self, mel: &Array) -> Result<Array> {
        let conv = |x: &Array, l: &Linear, stride: i32| -> Result<Array> {
            let y = conv1d(x, &l.w, stride, 1, None, None)?;
            Ok(gelu(y.add(l.b.as_ref().unwrap())?)?)
        };
        let x = conv(mel, &self.conv1, 1)?;
        let x = conv(&x, &self.conv2, 2)?;
        if x.dim(1) != self.pos.dim(0) {
            bail!("unexpected audio length {} (want {})", x.dim(1), self.pos.dim(0));
        }
        let mut x = x.add(&self.pos)?;
        for b in &self.blocks {
            x = b.encode(&x)?;
        }
        self.ln_post.forward(&x)
    }
}

struct Decoder {
    token_embedding: Array,
    positional_embedding: Array,
    blocks: Vec<Block>,
    ln: LayerNorm,
}

impl Decoder {
    fn load(ws: &mut Weights, d: &Dims) -> Result<Self> {
        Ok(Self {
            token_embedding: ws.take("decoder.token_embedding.weight")?,
            positional_embedding: ws.take("decoder.positional_embedding")?,
            blocks: (0..d.n_text_layer)
                .map(|i| Block::load(ws, &format!("decoder.blocks.{i}"), d.n_text_head, true))
                .collect::<Result<_>>()?,
            ln: LayerNorm::load(ws, "decoder.ln")?,
        })
    }

    /// Logits (n_vocab,) for the last of `tokens`, appending them to `cache`.
    fn forward(&self, tokens: &[i32], xa: &Array, cache: &mut [LayerCache]) -> Result<Array> {
        let offset = cache[0].kv.as_ref().map_or(0, |(k, _)| k.dim(1));
        let n = tokens.len() as i32;
        let ids = Array::from_slice(tokens, &[1, n]);
        let mut x = self
            .token_embedding
            .take_axis(&ids, 0)?
            .add(self.positional_embedding.index((offset..offset + n, ..)))?;
        for (b, c) in self.blocks.iter().zip(cache.iter_mut()) {
            x = b.decode(&x, xa, c)?;
        }
        let x = self.ln.forward(&x.index((.., n - 1..n, ..)))?;
        Ok(x.matmul(self.token_embedding.t())?.flatten(None, None)?)
    }
}

pub struct Whisper {
    pub dims: Dims,
    encoder: Encoder,
    decoder: Decoder,
}

impl Whisper {
    pub fn load(dir: &Path) -> Result<Self> {
        let dims = Dims::from_config(&dir.join("config.json"))?;
        let st = dir.join("weights.safetensors");
        let raw = if st.exists() {
            Array::load_safetensors(&st).context("load weights.safetensors")?
        } else {
            load_npz(&dir.join("weights.npz")).context("load weights.npz")?
        };
        let mut ws = Weights(raw);
        let model = Self { encoder: Encoder::load(&mut ws, &dims)?, decoder: Decoder::load(&mut ws, &dims)?, dims };
        Ok(model)
    }

    pub fn make_cache(&self) -> Vec<LayerCache> {
        (0..self.decoder.blocks.len()).map(|_| LayerCache::default()).collect()
    }

    /// Encode a (3000, n_mels) log-mel spectrogram.
    pub fn encode(&self, mel: &Array) -> Result<Array> {
        let xa = self.encoder.forward(&mel.as_dtype(Dtype::Float16)?.expand_dims(0)?)?;
        xa.eval()?;
        Ok(xa)
    }

    /// Greedy decode: returns the generated text token ids (no specials).
    /// Greedy decode; `on_token` returns false to stop early (e.g. the user is quitting).
    pub fn decode(&self, xa: &Array, prompt: &[i32], rules: &Rules, mut on_token: impl FnMut(i32) -> bool) -> Result<Vec<i32>> {
        let mut cache = self.make_cache();
        let max_new = (self.dims.n_text_ctx as usize).saturating_sub(prompt.len() + 1);
        let mut out = Vec::new();
        let mut logits = self.decoder.forward(prompt, xa, &mut cache)?;
        while out.len() < max_new {
            let masked = logits.as_dtype(Dtype::Float32)?.add(if out.is_empty() { &rules.first } else { &rules.always })?;
            let tok = argmax_axis(&masked, -1, None)?.item::<u32>() as i32;
            if tok == rules.eot {
                break;
            }
            out.push(tok);
            if !on_token(tok) {
                break;
            }
            logits = self.decoder.forward(&[tok], xa, &mut cache)?;
        }
        Ok(out)
    }
}

/// Special tokens and logit masks for plain (no-timestamp) transcription.
pub struct Rules {
    pub eot: i32,
    pub prompt: Vec<i32>,
    /// -inf at tokens never allowed (control tokens, non-speech symbols).
    always: Array,
    /// `always` plus blank/eot, applied to the first generated token only.
    first: Array,
}

impl Rules {
    pub fn new(tok: &Tokenizer, n_vocab: i32, language: &str) -> Result<Self> {
        let id = |s: &str| tok.token_to_id(s).map(|i| i as i32).with_context(|| format!("tokenizer lacks {s}"));
        let eot = id("<|endoftext|>")?;
        let sot = id("<|startoftranscript|>")?;
        let lang = id(&format!("<|{language}|>")).with_context(|| format!("unknown language {language:?}"))?;
        let prompt = vec![sot, lang, id("<|transcribe|>")?, id("<|notimestamps|>")?];

        let mut suppress: Vec<i32> = vec![sot, id("<|transcribe|>")?, id("<|translate|>")?];
        for s in ["<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|nocaptions|>"] {
            suppress.extend(tok.token_to_id(s).map(|i| i as i32));
        }
        // Timestamp tokens: everything after <|notimestamps|>.
        suppress.extend(id("<|notimestamps|>")? + 1..n_vocab);
        suppress.extend(non_speech_tokens(tok)?);

        let mask = |ids: &[i32]| {
            let mut m = vec![0f32; n_vocab as usize];
            for &i in ids {
                m[i as usize] = f32::NEG_INFINITY;
            }
            Array::from_slice(&m, &[n_vocab])
        };
        let always = mask(&suppress);
        let mut first_ids = suppress.clone();
        first_ids.push(eot);
        first_ids.extend(encode(tok, " ")?);
        Ok(Self { eot, prompt, always, first: mask(&first_ids) })
    }
}

fn encode(tok: &Tokenizer, s: &str) -> Result<Vec<i32>> {
    Ok(tok.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().iter().map(|&i| i as i32).collect())
}

/// Tokens for speaker tags / sound annotations, as in mlx_whisper/tokenizer.py.
fn non_speech_tokens(tok: &Tokenizer) -> Result<Vec<i32>> {
    let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』".chars().map(String::from).collect();
    symbols.extend("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split(' ').map(String::from));
    let misc = "♩♪♫♬♭♮♯";
    let mut out = vec![encode(tok, " -")?[0], encode(tok, " '")?[0]];
    for s in symbols.iter().map(String::as_str).chain(misc.chars().map(|c| c.to_string()).collect::<Vec<_>>().iter().map(String::as_str)) {
        for t in [encode(tok, s)?, encode(tok, &format!(" {s}"))?] {
            if t.len() == 1 || misc.contains(s) {
                out.push(t[0]);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}
