//! Run Qwen3.5-9B (MLX 4-bit) on Apple Silicon from Rust via mlx-rs.

use std::io::{BufRead, Write};

use anyhow::Result;
use clap::Parser;

use qwen35::llm::{Chat, Engine, Sampler, build_prompt};

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

fn main() -> Result<()> {
    let args = Args::parse();
    let engine = Engine::load(&args.model, Sampler { temp: args.temp, top_k: args.top_k }, !args.no_kernel)?;

    if args.bench {
        let prompt = build_prompt("Write a long story about a robot.", false);
        let ids = engine.encode(&prompt)?;
        engine.generate(&ids, &mut engine.model.make_cache(), 16, |_| true)?; // warm up Metal kernels
        // Ignore EOS so every run produces the same number of tokens.
        let bench = Engine { eos: vec![], ..engine };
        bench.generate(&ids, &mut bench.model.make_cache(), 512, |_| true)?.print();
        return Ok(());
    }

    if args.chat {
        let mut chat = Chat::new(engine, None, args.think, args.max_tokens);
        eprintln!("Chat ready. Ctrl-D or /exit to quit.");
        let stdin = std::io::stdin();
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
            let stdout = std::io::stdout();
            let (_, stats) = chat.reply(user, |t| {
                let mut lock = stdout.lock();
                let _ = write!(lock, "{t}");
                let _ = lock.flush();
                true
            })?;
            println!();
            stats.print();
        }
        return Ok(());
    }

    let prompt = build_prompt(&args.prompt, args.think);
    let mut cache = engine.model.make_cache();
    engine.run(&prompt, &mut cache, args.max_tokens, args.print_ids)?.print();
    Ok(())
}
