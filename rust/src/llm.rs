//! Text generation with the Qwen3.5 model: prompt format, sampling, pipelined decoding and a
//! multi-turn chat that keeps the KV cache between turns.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::ops::{partition_axis, r#where, split_sections};
use mlx_rs::{Array, random, transforms};
use tokenizers::Tokenizer;

use crate::hf;
use crate::model::{LayerCache, Model};

/// One user turn in Qwen's chat format, with thinking disabled unless asked for.
pub fn build_prompt(user: &str, think: bool) -> String {
    let mut s = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    if !think {
        s.push_str("\n</think>\n\n");
    }
    s
}

/// A system turn, which must come first in the conversation.
pub fn system_prompt(system: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n")
}

pub struct Sampler {
    pub temp: f32,
    pub top_k: i32,
}

impl Sampler {
    pub fn sample(&self, logits: &Array) -> Result<Array> {
        if self.temp <= 0.0 {
            return Ok(argmax_axis(logits, -1, None)?);
        }
        let mut logits = logits.multiply(Array::from_f32(1.0 / self.temp))?;
        let vocab = logits.dim(-1);
        if self.top_k > 0 && self.top_k < vocab {
            // Keep only logits >= the k-th largest value.
            let part = partition_axis(&logits, vocab - self.top_k, -1)?;
            let kth = split_sections(&part, &[vocab - self.top_k, vocab - self.top_k + 1], -1)?.swap_remove(1);
            let ninf = Array::from_f32(f32::NEG_INFINITY).as_dtype(logits.dtype())?;
            logits = r#where(&logits.lt(&kth)?, &ninf, &logits)?;
        }
        Ok(random::categorical(&logits, -1, None, None)?)
    }
}

pub struct Stats {
    pub prompt_tokens: usize,
    pub prompt_tps: f64,
    pub gen_tokens: usize,
    pub gen_tps: f64,
    /// The pipelined decode already fed <|im_end|> into the cache.
    pub eos_in_cache: bool,
}

impl Stats {
    pub fn print(&self) {
        let peak = mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9;
        eprintln!(
            "\n[prompt {} tok @ {:.1} tok/s | gen {} tok @ {:.1} tok/s | peak mem {:.2} GB]",
            self.prompt_tokens, self.prompt_tps, self.gen_tokens, self.gen_tps, peak
        );
    }
}

pub struct Engine {
    pub model: Model,
    pub tok: Tokenizer,
    pub eos: Vec<u32>,
    pub sampler: Sampler,
    /// "mlx-community/Qwen3.5-9B-4bit · rev ... · ~/.cache/... · 6.0 GB"
    pub info: String,
    /// "mlx-community/Qwen3.5-9B-4bit · rev ..."
    pub info_short: String,
}

impl Engine {
    /// Load a model (repo id or local directory) and its tokenizer.
    pub fn load(spec: &str, sampler: Sampler, use_kernel: bool) -> Result<Self> {
        let dir = hf::resolve(spec, "config.json", &[])?;
        let t0 = Instant::now();
        let model = Model::load(&dir, use_kernel)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let eos = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
        eprintln!("Loaded {} in {:.1}s", spec, t0.elapsed().as_secs_f64());
        let info = hf::describe(spec, &dir);
        let info_short = hf::describe_short(spec, &dir);
        Ok(Self { model, tok, eos, sampler, info, info_short })
    }

    /// Generate from `prompt`, handing decoded text to `on_text` as it arrives (partial UTF-8
    /// sequences are held back until complete); returns the whole text and the statistics.
    /// `on_text` returns false to stop generating (the reply so far is returned).
    pub fn generate_text(
        &self,
        prompt: &str,
        cache: &mut [LayerCache],
        max_tokens: usize,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, Stats)> {
        let ids = self.encode(prompt)?;
        let mut out: Vec<u32> = Vec::new();
        let mut printed = 0;
        let stats = self.generate(&ids, cache, max_tokens, |t| {
            out.push(t);
            if let Ok(text) = self.tok.decode(&out, false) {
                if !text.ends_with('\u{FFFD}') && text.len() > printed {
                    let go_on = on_text(&text[printed..]);
                    printed = text.len();
                    return go_on;
                }
            }
            true
        })?;
        let text = self.tok.decode(&out, false).map_err(anyhow::Error::msg)?;
        Ok((text, stats))
    }

    /// Prefill `prompt_ids`, then decode until EOS or `max_tokens`, handing each token to
    /// `on_token`, which returns false to stop early (the cache then holds an unfinished turn).
    /// Decoding is pipelined: the next step's graph is queued (async_eval) before the
    /// current token is read back, so the GPU never idles waiting on the CPU.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        cache: &mut [LayerCache],
        max_tokens: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Stats> {
        let ids: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let input = Array::from_slice(&ids, &[1, ids.len() as i32]);

        let t0 = Instant::now();
        let logits = self.model.forward(&input, cache)?;
        let mut y = self.sampler.sample(&logits)?;
        transforms::async_eval([&y])?;

        let mut n = 0;
        let mut t_gen = Instant::now();
        let mut prompt_secs = 0.0;
        let mut eos_in_cache = false;
        while n < max_tokens {
            let next = if n + 1 < max_tokens {
                let logits = self.model.forward(&y.reshape(&[1, 1])?, cache)?;
                let nx = self.sampler.sample(&logits)?;
                transforms::async_eval([&nx])?;
                Some(nx)
            } else {
                None
            };
            let token = y.item::<u32>();
            if n == 0 {
                prompt_secs = t0.elapsed().as_secs_f64();
                t_gen = Instant::now();
            }
            n += 1;
            if self.eos.contains(&token) {
                eos_in_cache = next.is_some();
                break;
            }
            if !on_token(token) {
                break;
            }
            match next {
                Some(nx) => y = nx,
                None => break,
            }
        }
        let gen_secs = t_gen.elapsed().as_secs_f64();
        Ok(Stats {
            prompt_tokens: prompt_ids.len(),
            prompt_tps: prompt_ids.len() as f64 / prompt_secs.max(1e-9),
            gen_tokens: n,
            gen_tps: (n.saturating_sub(1)) as f64 / gen_secs.max(1e-9),
            eos_in_cache,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tok.encode(text, false).map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    /// Generate and stream decoded text to stdout.
    pub fn run(&self, prompt: &str, cache: &mut [LayerCache], max_tokens: usize, print_ids: bool) -> Result<Stats> {
        let ids = self.encode(prompt)?;
        let mut out: Vec<u32> = Vec::new();
        let mut printed = 0;
        let stdout = std::io::stdout();
        let stats = self.generate(&ids, cache, max_tokens, |t| {
            out.push(t);
            if let Ok(text) = self.tok.decode(&out, false) {
                // Hold back partial UTF-8 sequences until they complete.
                if !text.ends_with('\u{FFFD}') && text.len() > printed {
                    let mut lock = stdout.lock();
                    let _ = write!(lock, "{}", &text[printed..]);
                    let _ = lock.flush();
                    printed = text.len();
                }
            }
            true
        })?;
        println!();
        if print_ids {
            println!("ids: {out:?}");
        }
        Ok(stats)
    }
}


/// A conversation. One KV cache for all turns, so each turn only processes its new tokens.
pub struct Chat {
    pub engine: Engine,
    cache: Vec<LayerCache>,
    system: Option<String>,
    think: bool,
    pub max_tokens: usize,
    /// How to close the previous assistant turn before the next user message.
    prefix: &'static str,
    started: bool,
    tokens: usize,
}

impl Chat {
    pub fn new(engine: Engine, system: Option<String>, think: bool, max_tokens: usize) -> Self {
        let cache = engine.model.make_cache();
        Self { engine, cache, system, think, max_tokens, prefix: "", started: false, tokens: 0 }
    }

    /// Forget the conversation (the system prompt is sent again with the next turn).
    pub fn reset(&mut self) {
        self.cache = self.engine.model.make_cache();
        self.prefix = "";
        self.started = false;
        self.tokens = 0;
    }

    /// Tokens held in the cache so far (prompt and generated).
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Send one user message; the reply streams to `on_text` and is returned whole.
    /// `on_text` returns false to cut the reply short.
    pub fn reply(&mut self, user: &str, on_text: impl FnMut(&str) -> bool) -> Result<(String, Stats)> {
        let mut prompt = String::new();
        match (&self.system, self.started) {
            (Some(system), false) => prompt.push_str(&system_prompt(system)),
            (_, true) => prompt.push_str(self.prefix),
            _ => {}
        }
        prompt.push_str(&build_prompt(user, self.think));
        let (text, stats) = self.engine.generate_text(&prompt, &mut self.cache, self.max_tokens, on_text)?;
        self.started = true;
        // Close the assistant turn before the next user message.
        self.prefix = if stats.eos_in_cache { "\n" } else { "<|im_end|>\n" };
        self.tokens += stats.prompt_tokens + stats.gen_tokens;
        Ok((text, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_format_matches_the_chat_template() {
        assert_eq!(build_prompt("hi", false), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(build_prompt("hi", true), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert_eq!(system_prompt("Be brief."), "<|im_start|>system\nBe brief.<|im_end|>\n");
    }
}
