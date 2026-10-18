//! Speech to text: locate the Whisper weights and tokenizer, load them on the GPU, transcribe.

use std::io::Write;
use std::time::Instant;

use anyhow::{Result, bail};
use tokenizers::Tokenizer;

use super::listen::quit_requested;
use super::mel;
use crate::hf::{self, resolve};
use super::whisper::{Rules, Whisper};

/// The mlx-community repos ship no tokenizer; take the matching one from openai/.
fn tokenizer_repo(n_vocab: i32) -> Result<&'static str> {
    Ok(match n_vocab {
        51866 => "openai/whisper-large-v3-turbo",
        51865 => "openai/whisper-base",
        51864 => "openai/whisper-base.en",
        n => bail!("no known tokenizer for n_vocab {n}"),
    })
}

// ---- transcription ----

pub struct Transcriber {
    pub model: Whisper,
    tok: Tokenizer,
    rules: Rules,
    /// "mlx-community/whisper-large-v3-turbo · rev ... · ~/.cache/... · 1.6 GB"
    pub info: String,
    /// "mlx-community/whisper-large-v3-turbo · rev ..."
    pub info_short: String,
}

// SAFETY: the Whisper weights are MLX arrays used from one thread at a time; the transcriber is
// created on the main thread and then owned by the transcription thread.
unsafe impl Send for Transcriber {}

impl Transcriber {
    /// Load a Whisper model (repo id or local dir) for `language`, warming up the GPU kernels.
    pub fn load(model: &str, language: &str) -> Result<Self> {
        let dir = resolve(model, "config.json", &[])?;
        let t0 = Instant::now();
        let whisper = Whisper::load(&dir)?;
        let tok_dir = resolve(tokenizer_repo(whisper.dims.n_vocab)?, "tokenizer.json", &["tokenizer.json"])?;
        let tok = Tokenizer::from_file(tok_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let rules = Rules::new(&tok, whisper.dims.n_vocab, language)?;
        // Warm up the Metal kernels so the first real transcription is not slow.
        whisper.encode(&mel::silence(whisper.dims.n_mels)?)?;
        eprintln!("Loaded {} in {:.1}s", model, t0.elapsed().as_secs_f64());
        let info = hf::describe(model, &dir);
        let info_short = hf::describe_short(model, &dir);
        Ok(Self { model: whisper, tok, rules, info, info_short })
    }

    /// Transcribe 16 kHz mono audio, streaming words to stdout after `USER: `; returns the text.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let t0 = Instant::now();
        let mel = mel::log_mel_spectrogram(audio, self.model.dims.n_mels)?;
        let xa = self.model.encode(&mel)?;
        let t_enc = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = 0;
        let stdout = std::io::stdout();
        print!("USER:");
        self.model.decode(&xa, &self.rules.prompt, &self.rules, |t| {
            ids.push(t as u32);
            if let Ok(text) = self.tok.decode(&ids, true) {
                if !text.ends_with('\u{FFFD}') && text.len() > printed {
                    let mut lock = stdout.lock();
                    let _ = write!(lock, "{}", &text[printed..]);
                    let _ = lock.flush();
                    printed = text.len();
                }
            }
            quit_requested().is_none()
        })?;
        println!();
        let n = ids.len();
        eprintln!(
            "[audio {:.1} s | encode {:.2} s | decode {} tok @ {:.0} tok/s]",
            audio.len() as f64 / mel::SAMPLE_RATE as f64,
            t_enc,
            n,
            n as f64 / t1.elapsed().as_secs_f64().max(1e-9)
        );
        Ok(self.tok.decode(&ids, true).map_err(anyhow::Error::msg)?.trim().to_string())
    }
}
