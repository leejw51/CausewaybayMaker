//! Audio agent example: listen on the microphone continuously, transcribe with Whisper on MLX.
//!
//!   make audio                 # keep listening; each utterance is transcribed when you pause
//!   make audio WAV=clip.wav    # transcribe a WAV file instead of the mic
//!
//! Whisper runs on the GPU through mlx-rs (src/audio/whisper.rs); mic capture and the energy-based
//! voice activity detector that cuts the stream into utterances live in src/audio/listen.rs.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use qwen35::audio::{listen, mel, npy, stt};

#[derive(Parser)]
#[command(about = "Microphone -> text with Whisper on Apple Silicon (MLX, Rust)")]
struct Args {
    /// Hugging Face repo id or local directory of an mlx-community Whisper model
    #[arg(long, default_value = "mlx-community/whisper-large-v3-turbo")]
    model: String,
    /// Transcribe this WAV file (any rate/channels) instead of the microphone
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Spoken language code, e.g. en, ko, ja, zh
    #[arg(long, default_value = "en")]
    language: String,
    #[command(flatten)]
    listen: listen::ListenArgs,
    /// List input devices and exit
    #[arg(long)]
    list_devices: bool,
    /// Compare the Rust log-mel against a (3000, n_mels) .npy from mlx_whisper and exit
    #[arg(long, hide = true)]
    mel_ref: Option<PathBuf>,
    /// With --wav: feed the file through the live mic path (streaming resampler + VAD) in small chunks
    #[arg(long, hide = true)]
    simulate: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    listen::quit_cleanly_on_signals();

    if let Some(path) = &args.mel_ref {
        let wav = args.wav.as_ref().context("--mel-ref needs --wav")?;
        let reference = npy::parse_npy(&std::fs::read(path)?)?;
        let n_mels = reference.dim(1);
        let ours = mel::log_mel_spectrogram(&listen::read_wav(wav)?, n_mels)?;
        let diff = ours.subtract(&reference)?.abs()?.max(None)?.item::<f32>();
        println!("mel shape {:?} vs {:?}, max abs diff {diff:.3e}", ours.shape(), reference.shape());
        return Ok(());
    }

    if args.list_devices {
        return listen::list_devices();
    }

    let stt = stt::Transcriber::load(&args.model, &args.language)?;
    if let Some(key) = listen::quit_requested() {
        listen::announce_quit(key);
        return Ok(());
    }

    if let Some(wav) = &args.wav {
        if args.simulate {
            return listen::simulate(wav, &args.listen, |u| stt.transcribe(u).map(drop));
        }
        stt.transcribe(&listen::read_wav(wav)?)?;
        return Ok(());
    }

    let result = listen::listen(&args.listen, false, |u| stt.transcribe(u).map(drop));
    if listen::quit_requested().is_some() {
        drop(stt);
        listen::exit_now(if result.is_ok() { 0 } else { 1 });
    }
    result
}
