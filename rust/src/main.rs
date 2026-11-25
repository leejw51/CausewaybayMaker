//! Run Qwen3.5 (MLX 4-bit) or Inkling 2 (from Ollama's store) on Apple Silicon from Rust via mlx-rs.

use std::io::{BufRead, Write};

use anyhow::Result;
use clap::Parser;

use qwen35::llm::{Chat, Engine, Sampler};

#[derive(Parser)]
#[command(about = "Qwen3.5 on Apple Silicon with MLX (Rust)")]
struct Args {
    /// Hugging Face repo id, local model directory, or an Ollama model (inkling-2-mini:26b-a5b-nvfp4)
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
    /// Enable thinking (Qwen: thinking mode; Inkling: effort 0.9 instead of 0)
    #[arg(long)]
    think: bool,
    /// Interactive multi-turn chat
    #[arg(long)]
    chat: bool,
    /// System prompt for --chat
    #[arg(long)]
    system: Option<String>,
    /// Measure throughput
    #[arg(long)]
    bench: bool,
    /// Use the ops-based recurrence instead of the Metal kernel (for verification)
    #[arg(long)]
    no_kernel: bool,
    /// Print generated token ids (for comparing against the Python run)
    #[arg(long)]
    print_ids: bool,
    /// Inkling: an image file to show with the prompt (repeatable). In --chat it goes with the
    /// first message; `/image <file>` on a line of its own attaches one to the next message.
    #[arg(long)]
    image: Vec<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut engine = Engine::load(&args.model, Sampler { temp: args.temp, top_k: args.top_k }, !args.no_kernel)?;
    engine.show_thinking = true;
    let format = engine.format;

    if args.bench {
        let prompt = format.start(None, false) + &format.user("Write a long story about a robot.", false);
        let ids = engine.encode(&prompt)?;
        engine.generate(&ids, &mut engine.make_cache(), 16, |_| true)?; // warm up Metal kernels
        // Ignore EOS so every run produces the same number of tokens.
        let bench = Engine { eos: vec![], ..engine };
        bench.generate(&ids, &mut bench.make_cache(), 512, |_| true)?.print();
        return Ok(());
    }

    if args.chat {
        let mut chat = Chat::new(engine, args.system.clone(), args.think, args.max_tokens);
        eprintln!("Chat ready. Ctrl-D or /exit to quit.");
        let stdin = std::io::stdin();
        for image in &args.image {
            chat.attach_image(image)?;
        }
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
            if let Some(path) = user.strip_prefix("/image").filter(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace)) {
                // The image is read and encoded now, while the question is being typed.
                match image_path(path).and_then(|p| chat.attach_image(&p)) {
                    Ok(()) => eprintln!("({} attached to your next message)", if chat.images_attached() == 1 { "image".to_string() } else { format!("{} images", chat.images_attached()) }),
                    Err(e) => eprintln!("(not attached: {e:#})"),
                }
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

    let mut cache = engine.make_cache();
    if !args.image.is_empty() {
        engine.warm_images()?;
    }
    let mut prompt = format.start(None, args.think);
    for image in &args.image {
        prompt.push_str(&engine.attach_image(&mut cache, image)?);
    }
    prompt.push_str(&format.user(&args.prompt, args.think));
    engine.run(&prompt, &mut cache, args.max_tokens, args.print_ids)?.print();
    Ok(())
}

/// The file a `/image` line names: quotes around it and a leading `~/` are understood, as is a
/// path dragged into the terminal (spaces escaped with backslashes).
fn image_path(typed: &str) -> Result<std::path::PathBuf> {
    let typed = typed.trim();
    let typed = typed.strip_prefix('"').and_then(|t| t.strip_suffix('"')).or_else(|| typed.strip_prefix('\'').and_then(|t| t.strip_suffix('\''))).unwrap_or(typed);
    if typed.is_empty() {
        anyhow::bail!("give a file: /image photo.jpg");
    }
    let typed = if std::path::Path::new(typed).exists() { typed.to_string() } else { typed.replace("\\ ", " ") };
    Ok(match (typed.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => std::path::Path::new(&home).join(rest),
        _ => typed.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::image_path;
    use std::path::PathBuf;

    #[test]
    fn an_image_line_names_a_file() {
        assert_eq!(image_path(" photo.jpg ").unwrap(), PathBuf::from("photo.jpg"));
        assert_eq!(image_path("\"my photo.jpg\"").unwrap(), PathBuf::from("my photo.jpg"));
        assert_eq!(image_path("'my photo.jpg'").unwrap(), PathBuf::from("my photo.jpg"));
        // A path dragged into the terminal.
        assert_eq!(image_path("/no/such/my\\ photo.jpg").unwrap(), PathBuf::from("/no/such/my photo.jpg"));
        assert!(image_path("").is_err());
        assert!(image_path("  ").is_err());
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(image_path("~/Pictures/a.png").unwrap(), PathBuf::from(home).join("Pictures/a.png"));
        }
    }
}
