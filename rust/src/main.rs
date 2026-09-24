//! Run Qwen3.5-9B (MLX 4-bit) on Apple Silicon from Rust via mlx-rs.

mod kernel;
mod model;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::ops::{partition_axis, r#where, split_sections};
use mlx_rs::{Array, random, transforms};
use tokenizers::Tokenizer;

use model::{LayerCache, Model};

#[derive(Parser)]
#[command(about = "Qwen3.5 on Apple Silicon with MLX (Rust)")]
struct Args {
    /// Hugging Face repo id or local model directory
    #[arg(long, default_value = "mlx-community/Qwen3.5-9B-4bit")]
    model: String,
    #[arg(long, default_value = "Explain quantum entanglement in 3 sentences.")]
    prompt: String,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    /// 0 = greedy
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    /// Sample only from the k most likely tokens (0 = off)
    #[arg(long, default_value_t = 20)]
    top_k: i32,
    /// Enable Qwen thinking mode (slower)
    #[arg(long)]
    think: bool,
    /// Interactive multi-turn chat
    #[arg(long)]
    chat: bool,
    /// Measure throughput
    #[arg(long)]
    bench: bool,
    /// Use the ops-based recurrence instead of the Metal kernel (for verification)
    #[arg(long)]
    no_kernel: bool,
    /// Print generated token ids (for comparing against the Python run)
    #[arg(long)]
    print_ids: bool,
}

fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return p.into();
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface"));
    home.join("hub")
}

fn cached_snapshot(repo: &str) -> Option<PathBuf> {
    let base = hf_cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    let rev = std::fs::read_to_string(base.join("refs/main")).ok()?;
    let dir = base.join("snapshots").join(rev.trim());
    dir.join("config.json").exists().then_some(dir)
}

/// Local dir, else the Hugging Face cache, else download with the `hf` CLI.
fn resolve_model(spec: &str) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.join("config.json").exists() {
        return Ok(p.to_path_buf());
    }
    if let Some(dir) = cached_snapshot(spec) {
        return Ok(dir);
    }
    eprintln!("Downloading {spec} from Hugging Face...");
    let status = Command::new("hf")
        .args(["download", spec])
        .status()
        .context("`hf` CLI not found; run `make download` or `pip install huggingface_hub[cli]`")?;
    if !status.success() {
        bail!("hf download {spec} failed");
    }
    cached_snapshot(spec).context("model not found in cache after download")
}

fn build_prompt(user: &str, think: bool) -> String {
    let mut s = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    if !think {
        s.push_str("\n</think>\n\n");
    }
    s
}

struct Sampler {
    temp: f32,
    top_k: i32,
}

impl Sampler {
    fn sample(&self, logits: &Array) -> Result<Array> {
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

struct Stats {
    prompt_tokens: usize,
    prompt_tps: f64,
    gen_tokens: usize,
    gen_tps: f64,
    /// The pipelined decode already fed <|im_end|> into the cache.
    eos_in_cache: bool,
}

impl Stats {
    fn print(&self) {
        let peak = mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9;
        eprintln!(
            "\n[prompt {} tok @ {:.1} tok/s | gen {} tok @ {:.1} tok/s | peak mem {:.2} GB]",
            self.prompt_tokens, self.prompt_tps, self.gen_tokens, self.gen_tps, peak
        );
    }
}

struct Engine {
    model: Model,
    tok: Tokenizer,
    eos: Vec<u32>,
    sampler: Sampler,
}

impl Engine {
    /// Prefill `prompt_ids`, then decode until EOS or `max_tokens`, streaming text via `on_text`.
    /// Decoding is pipelined: the next step's graph is queued (async_eval) before the
    /// current token is read back, so the GPU never idles waiting on the CPU.
    fn generate(
        &self,
        prompt_ids: &[u32],
        cache: &mut [LayerCache],
        max_tokens: usize,
        mut on_token: impl FnMut(u32),
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
            on_token(token);
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

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tok.encode(text, false).map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    /// Generate and stream decoded text to stdout.
    fn run(&self, prompt: &str, cache: &mut [LayerCache], max_tokens: usize, print_ids: bool) -> Result<Stats> {
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
        })?;
        println!();
        if print_ids {
            println!("ids: {out:?}");
        }
        Ok(stats)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dir = resolve_model(&args.model)?;

    let t0 = Instant::now();
    let model = Model::load(&dir, !args.no_kernel)?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let eos = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect();
    eprintln!("Loaded {} in {:.1}s", args.model, t0.elapsed().as_secs_f64());

    let engine = Engine {
        model,
        tok,
        eos,
        sampler: Sampler { temp: args.temp, top_k: args.top_k },
    };

    if args.bench {
        let prompt = build_prompt("Write a long story about a robot.", false);
        let ids = engine.encode(&prompt)?;
        engine.generate(&ids, &mut engine.model.make_cache(), 16, |_| {})?; // warm up Metal kernels
        // Ignore EOS so every run produces the same number of tokens.
        let bench = Engine { eos: vec![], ..engine };
        bench.generate(&ids, &mut bench.model.make_cache(), 512, |_| {})?.print();
        return Ok(());
    }

    if args.chat {
        // One cache for the whole conversation: each turn only processes its new tokens.
        let mut cache = engine.model.make_cache();
        eprintln!("Chat ready. Ctrl-D or /exit to quit.");
        let stdin = std::io::stdin();
        let mut prefix = "";
        loop {
            print!("\n>>> ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let user = line.trim();
            if user == "/exit" || user == "/quit" {
                break;
            }
            if user.is_empty() {
                continue;
            }
            let prompt = format!("{prefix}{}", build_prompt(user, args.think));
            let stats = engine.run(&prompt, &mut cache, args.max_tokens, false)?;
            stats.print();
            // Close the assistant turn before the next user message.
            prefix = if stats.eos_in_cache { "\n" } else { "<|im_end|>\n" };
        }
        return Ok(());
    }

    let prompt = build_prompt(&args.prompt, args.think);
    let mut cache = engine.model.make_cache();
    engine.run(&prompt, &mut cache, args.max_tokens, args.print_ids)?.print();
    Ok(())
}
