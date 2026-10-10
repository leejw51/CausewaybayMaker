//! Audio agent example: listen on the microphone continuously, transcribe with Whisper on MLX.
//!
//!   make audio                 # keep listening; each utterance is transcribed when you pause
//!   make audio WAV=clip.wav    # transcribe a WAV file instead of the mic
//!
//! Whisper runs on the GPU through mlx-rs (see whisper.rs); mic capture uses cpal.
//! Utterances are cut by a simple energy-based voice activity detector (VAD).

mod mel;
mod npy;
mod whisper;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokenizers::Tokenizer;

use whisper::{Rules, Whisper};

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
    /// Input device to use (substring of its name, e.g. "AirPods"); default: system input
    #[arg(long)]
    device: Option<String>,
    /// List input devices and exit
    #[arg(long)]
    list_devices: bool,
    /// Speech level: RMS above this counts as voice (raise if noise triggers transcripts)
    #[arg(long, default_value_t = 0.01)]
    vad_threshold: f32,
    /// A pause this long (ms) ends an utterance
    #[arg(long, default_value_t = 700)]
    silence_ms: u32,
    /// Ignore utterances with less voice than this (ms)
    #[arg(long, default_value_t = 300)]
    min_speech_ms: u32,
    /// Transcribe one utterance and exit instead of listening forever
    #[arg(long)]
    once: bool,
    /// Compare the Rust log-mel against a (3000, n_mels) .npy from mlx_whisper and exit
    #[arg(long, hide = true)]
    mel_ref: Option<PathBuf>,
    /// With --wav: feed the file through the live mic path (streaming resampler + VAD) in small chunks
    #[arg(long, hide = true)]
    simulate: bool,
}

// ---- model files ----

fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return p.into();
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface"));
    home.join("hub")
}

fn cached_snapshot(repo: &str, marker: &str) -> Option<PathBuf> {
    let base = hf_cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    let rev = std::fs::read_to_string(base.join("refs/main")).ok()?;
    let dir = base.join("snapshots").join(rev.trim());
    dir.join(marker).exists().then_some(dir)
}

/// Local dir, else the Hugging Face cache, else download `files` of `repo` with the `hf` CLI.
fn resolve(repo: &str, marker: &str, files: &[&str]) -> Result<PathBuf> {
    let p = Path::new(repo);
    if p.join(marker).exists() {
        return Ok(p.to_path_buf());
    }
    if let Some(dir) = cached_snapshot(repo, marker) {
        return Ok(dir);
    }
    eprintln!("Downloading {repo} from Hugging Face...");
    let status = Command::new("hf")
        .args(["download", repo])
        .args(files)
        .status()
        .context("`hf` CLI not found; run `make setup` or `pip install huggingface_hub[cli]`")?;
    if !status.success() {
        bail!("hf download {repo} failed");
    }
    cached_snapshot(repo, marker).context("model not found in cache after download")
}

/// The mlx-community repos ship no tokenizer; take the matching one from openai/.
fn tokenizer_repo(n_vocab: i32) -> Result<&'static str> {
    Ok(match n_vocab {
        51866 => "openai/whisper-large-v3-turbo",
        51865 => "openai/whisper-base",
        51864 => "openai/whisper-base.en",
        n => bail!("no known tokenizer for n_vocab {n}"),
    })
}

// ---- audio input ----

/// Interleaved samples as recorded, with (channels, sample rate).
fn read_wav_raw(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok((samples, spec.channels as usize, spec.sample_rate as usize))
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let (samples, channels, rate) = read_wav_raw(path)?;
    Ok(to_16k_mono(&samples, channels, rate))
}

/// Replay a WAV as if it came from the microphone: 10 ms device chunks through the
/// streaming resampler and the VAD, transcribing each utterance (for testing without a mic).
fn simulate_mic(agent: &Agent, args: &Args, wav: &Path) -> Result<()> {
    let (samples, channels, rate) = read_wav_raw(wav)?;
    let mut rs = Resampler::new(channels, rate);
    let mut vad = Vad::new(args);
    let mut pending = Vec::new();
    let chunk = rate / 100 * channels;
    // Trailing silence so the last utterance is closed like a real pause would.
    let tail = vec![0f32; rate * channels];
    for c in samples.chunks(chunk).chain(tail.chunks(chunk)) {
        pending.extend(rs.push(c));
        let n = pending.len() / vad.frame * vad.frame;
        for frame in pending[..n].chunks_exact(vad.frame) {
            if let Some(u) = vad.push(frame) {
                eprintln!("[voice {:.1} s, peak level {:.3}]", u.len() as f64 / mel::SAMPLE_RATE as f64, vad.peak);
                agent.transcribe(&u)?;
            }
        }
        pending.drain(..n);
    }
    Ok(())
}

/// Streaming converter from the device format to 16 kHz mono; keeps state across chunks.
struct Resampler {
    channels: usize,
    rate: usize,
    pending: Vec<f32>,
    /// Fractional read position into `pending` (non-integer ratios only).
    pos: f64,
}

impl Resampler {
    fn new(channels: usize, rate: usize) -> Self {
        Self { channels: channels.max(1), rate, pending: Vec::new(), pos: 0.0 }
    }

    fn push(&mut self, interleaved: &[f32]) -> Vec<f32> {
        self.pending.extend(interleaved.chunks(self.channels).map(|c| c.iter().sum::<f32>() / c.len() as f32));
        let target = mel::SAMPLE_RATE;
        if self.rate == target {
            return std::mem::take(&mut self.pending);
        }
        let mut out = Vec::new();
        if self.rate % target == 0 {
            // Integer ratio (48 kHz is the usual mic rate): box-average every `k` samples.
            let k = self.rate / target;
            let n = self.pending.len() / k * k;
            out.extend(self.pending[..n].chunks_exact(k).map(|c| c.iter().sum::<f32>() / k as f32));
            self.pending.drain(..n);
        } else {
            let step = self.rate as f64 / target as f64;
            while (self.pos as usize) + 1 < self.pending.len() {
                let j = self.pos as usize;
                let frac = (self.pos - j as f64) as f32;
                out.push(self.pending[j] + (self.pending[j + 1] - self.pending[j]) * frac);
                self.pos += step;
            }
            let consumed = self.pos as usize;
            self.pending.drain(..consumed);
            self.pos -= consumed as f64;
        }
        out
    }
}

/// Convert a whole buffer at once (WAV input).
fn to_16k_mono(samples: &[f32], channels: usize, rate: usize) -> Vec<f32> {
    Resampler::new(channels, rate).push(samples)
}

fn input_devices() -> Result<Vec<(String, cpal::Device)>> {
    Ok(cpal::default_host()
        .input_devices()?
        .map(|d| (d.description().map(|x| x.to_string()).unwrap_or_else(|_| "?".into()), d))
        .collect())
}

struct Mic {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
}

impl Mic {
    fn open(name: Option<&str>) -> Result<Self> {
        let hint = "no usable microphone: connect one, allow microphone access for your terminal \
                    (System Settings > Privacy & Security > Microphone), or pass --wav FILE";
        let device = match name {
            Some(n) => {
                let n = n.to_lowercase();
                input_devices()?
                    .into_iter()
                    .find(|(d, _)| d.to_lowercase().contains(&n))
                    .map(|(_, d)| d)
                    .with_context(|| format!("no input device matching {n:?} (see --list-devices)"))?
            }
            None => cpal::default_host().default_input_device().context(hint)?,
        };
        let config = device.default_input_config().context(hint)?;
        eprintln!(
            "Microphone: {} ({} ch, {} Hz, {:?})",
            device.description().map(|d| d.to_string()).unwrap_or_else(|_| "?".into()),
            config.channels(),
            config.sample_rate(),
            config.sample_format()
        );
        Ok(Self { device, config })
    }

    /// Open the input stream; it runs until dropped, appending 16 kHz mono samples to `sink`.
    fn start(&self, sink: Arc<Mutex<Vec<f32>>>) -> Result<cpal::Stream> {
        let err = |e| eprintln!("audio stream error: {e}");
        let cfg = self.config.config();
        let mut rs = Resampler::new(self.config.channels() as usize, self.config.sample_rate() as usize);
        macro_rules! stream {
            ($t:ty, $conv:expr) => {{
                let mut scratch: Vec<f32> = Vec::new();
                self.device.build_input_stream::<$t, _, _>(
                    cfg,
                    move |data: &[$t], _| {
                        scratch.clear();
                        scratch.extend(data.iter().map($conv));
                        let mono = rs.push(&scratch);
                        sink.lock().unwrap().extend(mono);
                    },
                    err,
                    None,
                )?
            }};
        }
        let stream = match self.config.sample_format() {
            SampleFormat::F32 => stream!(f32, |&s| s),
            SampleFormat::I16 => stream!(i16, |&s| s as f32 / 32768.0),
            SampleFormat::I32 => stream!(i32, |&s| s as f32 / 2147483648.0),
            SampleFormat::U16 => stream!(u16, |&s| (s as f32 - 32768.0) / 32768.0),
            f => bail!("unsupported microphone sample format {f:?}"),
        };
        stream.play()?;
        Ok(stream)
    }
}

/// Energy-based voice activity detector that cuts the stream into utterances.
struct Vad {
    threshold: f32,
    silence: usize,
    min_speech: usize,
    /// 20 ms analysis frames.
    frame: usize,
    /// Audio kept from just before speech starts, so the first syllable is not clipped.
    preroll: Vec<f32>,
    utterance: Vec<f32>,
    in_speech: bool,
    speech_samples: usize,
    trailing_silence: usize,
    peak: f32,
}

impl Vad {
    fn new(args: &Args) -> Self {
        let per_ms = mel::SAMPLE_RATE / 1000;
        Self {
            threshold: args.vad_threshold,
            silence: args.silence_ms as usize * per_ms,
            min_speech: args.min_speech_ms as usize * per_ms,
            frame: 20 * per_ms,
            preroll: Vec::new(),
            utterance: Vec::new(),
            in_speech: false,
            speech_samples: 0,
            trailing_silence: 0,
            peak: 0.0,
        }
    }

    /// Feed one 20 ms frame; returns a finished utterance when a pause (or the 30 s cap) ends it.
    fn push(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        let voiced = rms > self.threshold;
        if !self.in_speech {
            self.preroll.extend_from_slice(frame);
            let keep = 300 * mel::SAMPLE_RATE / 1000;
            if self.preroll.len() > keep {
                self.preroll.drain(..self.preroll.len() - keep);
            }
            if !voiced {
                return None;
            }
            self.in_speech = true;
            self.utterance = std::mem::take(&mut self.preroll);
            self.speech_samples = 0;
            self.trailing_silence = 0;
            self.peak = 0.0;
        }
        self.utterance.extend_from_slice(frame);
        self.peak = self.peak.max(rms);
        if voiced {
            self.speech_samples += frame.len();
            self.trailing_silence = 0;
        } else {
            self.trailing_silence += frame.len();
        }
        let done = self.trailing_silence >= self.silence || self.utterance.len() >= mel::N_SAMPLES;
        if !done {
            return None;
        }
        self.in_speech = false;
        let audio = std::mem::take(&mut self.utterance);
        (self.speech_samples >= self.min_speech).then_some(audio)
    }
}

// ---- main ----

struct Agent {
    model: Whisper,
    tok: Tokenizer,
    rules: Rules,
}

impl Agent {
    /// Transcribe 16 kHz mono audio, streaming words to stdout; returns the text.
    fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let t0 = Instant::now();
        let mel = mel::log_mel_spectrogram(audio, self.model.dims.n_mels)?;
        let xa = self.model.encode(&mel)?;
        let t_enc = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = 0;
        let stdout = std::io::stdout();
        print!(">>> ");
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

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(path) = &args.mel_ref {
        let wav = args.wav.as_ref().context("--mel-ref needs --wav")?;
        let reference = npy::parse_npy(&std::fs::read(path)?)?;
        let n_mels = reference.dim(1);
        let ours = mel::log_mel_spectrogram(&read_wav(wav)?, n_mels)?;
        let diff = ours.subtract(&reference)?.abs()?.max(None)?.item::<f32>();
        println!("mel shape {:?} vs {:?}, max abs diff {diff:.3e}", ours.shape(), reference.shape());
        return Ok(());
    }

    if args.list_devices {
        for (name, _) in input_devices()? {
            println!("{name}");
        }
        return Ok(());
    }

    let dir = resolve(&args.model, "config.json", &[])?;
    let t0 = Instant::now();
    let model = Whisper::load(&dir)?;
    let tok_dir = resolve(tokenizer_repo(model.dims.n_vocab)?, "tokenizer.json", &["tokenizer.json"])?;
    let tok = Tokenizer::from_file(tok_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let rules = Rules::new(&tok, model.dims.n_vocab, &args.language)?;
    let agent = Agent { model, tok, rules };
    // Warm up the Metal kernels so the first real transcription is not slow.
    agent.model.encode(&mel::silence(agent.model.dims.n_mels)?)?;
    eprintln!("Loaded {} in {:.1}s", args.model, t0.elapsed().as_secs_f64());

    if let Some(wav) = &args.wav {
        if args.simulate {
            return simulate_mic(&agent, &args, wav);
        }
        agent.transcribe(&read_wav(wav)?)?;
        return Ok(());
    }

    let mic = Mic::open(args.device.as_deref())?;
    let sink: Arc<Mutex<Vec<f32>>> = Arc::default();
    let _stream = mic.start(sink.clone())?;
    let mut vad = Vad::new(&args);
    let mut pending: Vec<f32> = Vec::new();
    eprintln!(
        "Listening (pause {} ms ends an utterance, level > {:.3} counts as speech). Ctrl-C to quit.",
        args.silence_ms, args.vad_threshold
    );
    loop {
        std::thread::sleep(Duration::from_millis(50));
        pending.append(&mut sink.lock().unwrap());
        let n = pending.len() / vad.frame * vad.frame;
        let mut utterances = Vec::new();
        for frame in pending[..n].chunks_exact(vad.frame) {
            if let Some(u) = vad.push(frame) {
                utterances.push(u);
            }
        }
        pending.drain(..n);
        for u in utterances {
            eprintln!("[voice {:.1} s, peak level {:.3}]", u.len() as f64 / mel::SAMPLE_RATE as f64, vad.peak);
            agent.transcribe(&u)?;
            if args.once {
                return Ok(());
            }
        }
    }
}
