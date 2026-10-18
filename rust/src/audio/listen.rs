//! Shared by the audio examples: microphone capture (cpal), voice activity detection that cuts
//! the stream into utterances, WAV input, and clean shutdown on Ctrl-C / Ctrl-Z.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::mel;

/// Command-line options shared by the listening examples.
#[derive(clap::Args)]
pub struct ListenArgs {
    /// Input device to use (substring of its name, e.g. "AirPods"); default: system input
    #[arg(long)]
    pub device: Option<String>,
    /// Speech level: RMS above this counts as voice (raise if noise triggers transcripts)
    #[arg(long, default_value_t = 0.01)]
    pub vad_threshold: f32,
    /// A pause this long (ms) ends an utterance
    #[arg(long, default_value_t = 700)]
    pub silence_ms: u32,
    /// Ignore utterances with less voice than this (ms)
    #[arg(long, default_value_t = 300)]
    pub min_speech_ms: u32,
    /// Handle one utterance and exit instead of listening forever
    #[arg(long)]
    pub once: bool,
}

// ---- WAV input ----

/// Interleaved samples as recorded, with (channels, sample rate).
pub fn read_wav_raw(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
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

/// A WAV file as 16 kHz mono.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let (samples, channels, rate) = read_wav_raw(path)?;
    Ok(Resampler::new(channels, rate).push(&samples))
}

/// Replay a WAV as if it came from the microphone: 10 ms device chunks through the
/// streaming resampler and the VAD, calling `handle` for each utterance (testing without a mic).
pub fn simulate(wav: &Path, args: &ListenArgs, mut handle: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
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
                handle(&u)?;
            }
        }
        pending.drain(..n);
    }
    Ok(())
}

// ---- resampling ----

/// Streaming converter from the device format to 16 kHz mono; keeps state across chunks.
pub struct Resampler {
    channels: usize,
    rate: usize,
    pending: Vec<f32>,
    /// Fractional read position into `pending` (non-integer ratios only).
    pos: f64,
}

impl Resampler {
    pub fn new(channels: usize, rate: usize) -> Self {
        Self { channels: channels.max(1), rate, pending: Vec::new(), pos: 0.0 }
    }

    pub fn push(&mut self, interleaved: &[f32]) -> Vec<f32> {
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

// ---- microphone ----

pub fn input_devices() -> Result<Vec<(String, cpal::Device)>> {
    Ok(cpal::default_host()
        .input_devices()?
        .map(|d| (d.description().map(|x| x.to_string()).unwrap_or_else(|_| "?".into()), d))
        .collect())
}

/// Print the input devices, or a hint when there are none.
pub fn list_devices() -> Result<()> {
    let devices = input_devices()?;
    if devices.is_empty() {
        println!("(no input devices: connect a microphone such as AirPods or a USB mic)");
    }
    for (name, _) in devices {
        println!("{name}");
    }
    Ok(())
}

pub struct Mic {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    /// Set by the stream's error callback when the device goes away (AirPods in their case,
    /// iPhone walked off); the listening loop then waits for a microphone to come back.
    disconnected: Arc<AtomicBool>,
}

impl Mic {
    pub fn open(name: Option<&str>) -> Result<Self> {
        let devices = input_devices()?;
        // What to do when there is no usable microphone: this depends on whether any input
        // device exists at all (a Mac Studio/mini has no built-in mic; AirPods disconnect when idle).
        let hint = || {
            if devices.is_empty() {
                "no input device on this Mac: connect a microphone (AirPods, USB mic, iPhone) \
                 and run again, or pass --wav FILE"
                    .to_string()
            } else {
                format!(
                    "no usable microphone (if the input shows silence, allow microphone access for your \
                     terminal in System Settings > Privacy & Security > Microphone). Inputs: {}; pick one \
                     with --device NAME",
                    devices.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>().join(", ")
                )
            }
        };
        let device = match name {
            Some(n) => {
                let n = n.to_lowercase();
                devices
                    .iter()
                    .find(|(d, _)| d.to_lowercase().contains(&n))
                    .map(|(_, d)| d.clone())
                    .with_context(|| format!("no input device matching {n:?}. {}", hint()))?
            }
            None => cpal::default_host().default_input_device().with_context(hint)?,
        };
        let config = device.default_input_config().with_context(hint)?;
        eprintln!(
            "Microphone: {} ({} ch, {} Hz, {:?})",
            device.description().map(|d| d.to_string()).unwrap_or_else(|_| "?".into()),
            config.channels(),
            config.sample_rate(),
            config.sample_format()
        );
        Ok(Self { device, config, disconnected: Arc::default() })
    }

    pub fn disconnected(&self) -> bool {
        self.disconnected.load(Ordering::Relaxed)
    }

    /// Open the input stream; it runs until dropped, appending 16 kHz mono samples to `sink`.
    pub fn start(&self, sink: Arc<Mutex<Vec<f32>>>) -> Result<cpal::Stream> {
        let gone = self.disconnected.clone();
        let mut last_warning: Option<std::time::Instant> = None;
        let err = move |e: cpal::Error| {
            let text = e.to_string();
            // Buffer over/underruns happen when the machine is busy (a picture being generated);
            // they lose a few milliseconds of audio and are not a disconnect. Mention them rarely.
            if text.contains("underrun") || text.contains("overrun") {
                if last_warning.is_none_or(|t| t.elapsed() > Duration::from_secs(30)) {
                    eprintln!("[mic: busy machine, a little audio was dropped]");
                    last_warning = Some(std::time::Instant::now());
                }
                return;
            }
            eprintln!("audio stream error: {text}");
            gone.store(true, Ordering::Relaxed);
        };
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

// ---- voice activity detection ----

/// Energy-based voice activity detector that cuts the stream into utterances.
pub struct Vad {
    threshold: f32,
    silence: usize,
    min_speech: usize,
    /// 20 ms analysis frames.
    pub frame: usize,
    /// Audio kept from just before speech starts, so the first syllable is not clipped.
    preroll: Vec<f32>,
    utterance: Vec<f32>,
    in_speech: bool,
    speech_samples: usize,
    trailing_silence: usize,
    /// Loudest frame of the last utterance.
    pub peak: f32,
    /// RMS of the last few frames, for the level meter.
    recent: std::collections::VecDeque<f32>,
}

impl Vad {
    pub fn new(args: &ListenArgs) -> Self {
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
            recent: std::collections::VecDeque::with_capacity(24),
        }
    }

    /// A little bar graph of the last 20 frames (400 ms) of input level.
    pub fn meter(&self) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        self.recent
            .iter()
            .map(|&rms| {
                let level = (rms / (self.threshold * 12.0)).clamp(0.0, 0.999);
                BARS[(level * BARS.len() as f32) as usize]
            })
            .collect()
    }

    /// Is someone talking right now (an utterance is being collected)?
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Length of the utterance being collected, in seconds.
    pub fn current_secs(&self) -> f32 {
        self.utterance.len() as f32 / mel::SAMPLE_RATE as f32
    }

    /// Forget any partial utterance (after a gap in the audio, or our own speech).
    pub fn reset(&mut self) {
        self.preroll.clear();
        self.utterance.clear();
        self.in_speech = false;
    }

    /// Feed one 20 ms frame; returns a finished utterance when a pause (or the 30 s cap) ends it.
    pub fn push(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        if self.recent.len() >= 20 {
            self.recent.pop_front();
        }
        self.recent.push_back(rms);
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

// ---- Ctrl-C / Ctrl-Z ----

/// The signal that asked us to quit (SIGINT for Ctrl-C, SIGTSTP for Ctrl-Z); 0 until then.
static QUIT: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_quit_signal(sig: libc::c_int) {
    if QUIT.swap(sig, Ordering::SeqCst) != 0 {
        // Pressed again while we were still busy (e.g. mid-transcription): quit right now.
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(128 + sig) };
    }
}

/// Ctrl-C and Ctrl-Z both shut the example down cleanly: the microphone stream is closed, the
/// model is freed and the process exits normally. Ctrl-Z would otherwise suspend the process
/// while it keeps the microphone open (and a stopped job is easy to forget).
pub fn quit_cleanly_on_signals() {
    // SAFETY: the handler only touches an atomic and _exit, both async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, on_quit_signal as libc::sighandler_t);
        libc::signal(libc::SIGTSTP, on_quit_signal as libc::sighandler_t);
    }
    // Watchdog: whatever the program is stuck in when the user quits (a long GPU call, a blocked
    // read), the process ends a few seconds after the keypress.
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(Duration::from_millis(100));
            if QUIT.load(Ordering::SeqCst) != 0 {
                std::thread::sleep(Duration::from_secs(5));
                eprintln!("(shutdown is taking too long; exiting now)");
                exit_now(0);
            }
        }
    });
}

/// Which key asked us to quit, if any.
pub fn quit_requested() -> Option<&'static str> {
    match QUIT.load(Ordering::SeqCst) {
        0 => None,
        libc::SIGTSTP => Some("Ctrl-Z"),
        libc::SIGTERM => Some("End of input"),
        _ => Some("Ctrl-C"),
    }
}

/// Ask to quit from code (e.g. typed input ended), as if a signal had arrived.
pub fn request_quit() {
    let _ = QUIT.compare_exchange(0, libc::SIGTERM, Ordering::SeqCst, Ordering::SeqCst);
}

/// Say goodbye; after Ctrl-Z also wake `make`/`sh` in our process group, which were stopped
/// by the same keypress, so they see our exit and finish instead of lingering suspended.
/// (The watchdog armed in `quit_cleanly_on_signals` ends the process if shutdown stalls.)
pub fn announce_quit(key: &str) {
    eprintln!("\n{key}: shutting down.");
    if key == "Ctrl-Z" {
        // SAFETY: plain libc call; SIGCONT is harmless to processes that are not stopped.
        unsafe { libc::kill(0, libc::SIGCONT) };
    }
}

/// End the process right now, skipping global destructors (MLX/Metal and CoreAudio ones can
/// hang at exit). Call after children have been stopped and output flushed.
pub fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: _exit only ends the process.
    unsafe { libc::_exit(code) }
}

// ---- the listening loop ----

/// Listen on the microphone until Ctrl-C / Ctrl-Z, calling `handle` with each utterance
/// (16 kHz mono). With `discard_while_busy`, audio captured while `handle` runs is dropped,
/// so an assistant does not transcribe its own spoken reply.
pub fn listen(args: &ListenArgs, discard_while_busy: bool, mut handle: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
    let device = args.device.as_deref();
    // No microphone yet (AirPods in their case, iPhone not nearby): say so once and wait for one.
    let mut mic = match Mic::open(device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e:#}");
            eprintln!("Waiting for a microphone... (Ctrl-C to quit)");
            loop {
                std::thread::sleep(Duration::from_secs(1));
                if let Some(key) = quit_requested() {
                    announce_quit(key);
                    return Ok(());
                }
                if let Ok(m) = Mic::open(device) {
                    break m;
                }
            }
        }
    };
    let sink: Arc<Mutex<Vec<f32>>> = Arc::default();
    let mut stream = Some(mic.start(sink.clone())?);
    let mut vad = Vad::new(args);
    let mut pending: Vec<f32> = Vec::new();
    let mut showing = false;
    eprintln!(
        "Listening (pause {} ms ends an utterance, level > {:.3} counts as speech). Ctrl-C or Ctrl-Z to quit (twice to force).",
        args.silence_ms, args.vad_threshold
    );
    loop {
        if let Some(key) = quit_requested() {
            drop(stream); // release the microphone before saying goodbye
            announce_quit(key);
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));

        if mic.disconnected() {
            stream = None;
            pending.clear();
            vad.reset();
            eprintln!("Microphone disconnected; waiting for one to come back...");
            loop {
                std::thread::sleep(Duration::from_secs(1));
                if let Some(key) = quit_requested() {
                    announce_quit(key);
                    return Ok(());
                }
                let reopened = Mic::open(device).and_then(|m| {
                    let s = m.start(sink.clone())?;
                    Ok((m, s))
                });
                if let Ok((m, s)) = reopened {
                    mic = m;
                    stream = Some(s);
                    sink.lock().unwrap().clear();
                    eprintln!("Listening again.");
                    break;
                }
            }
        }

        pending.append(&mut sink.lock().unwrap());
        let n = pending.len() / vad.frame * vad.frame;
        let mut utterances = Vec::new();
        for frame in pending[..n].chunks_exact(vad.frame) {
            if let Some(u) = vad.push(frame) {
                utterances.push(u);
            }
        }
        pending.drain(..n);
        // Show that speech is being heard while it is still going on: a live level meter.
        if vad.in_speech() {
            eprint!("\r\x1b[K🎤 {} {:.1} s", vad.meter(), vad.current_secs());
            showing = true;
        } else if showing {
            eprint!("\r\x1b[K");
            showing = false;
        }
        for u in utterances {
            if showing {
                eprint!("\r\x1b[K");
                showing = false;
            }
            eprintln!("[voice {:.1} s, peak level {:.3}]", u.len() as f64 / mel::SAMPLE_RATE as f64, vad.peak);
            handle(&u)?;
            if args.once {
                return Ok(());
            }
            if discard_while_busy {
                sink.lock().unwrap().clear();
                pending.clear();
                vad.reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(threshold: f32) -> ListenArgs {
        ListenArgs { device: None, vad_threshold: threshold, silence_ms: 300, min_speech_ms: 100, once: false }
    }

    /// `secs` of a 440 Hz tone at `amp`, at `rate` Hz, `channels` interleaved.
    fn tone(secs: f32, amp: f32, rate: usize, channels: usize) -> Vec<f32> {
        let n = (secs * rate as f32) as usize;
        (0..n)
            .flat_map(|i| {
                let v = amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin();
                std::iter::repeat_n(v, channels)
            })
            .collect()
    }

    #[test]
    fn resampler_integer_ratio_48k_stereo() {
        let mut rs = Resampler::new(2, 48000);
        let out = rs.push(&tone(1.0, 0.5, 48000, 2));
        assert_eq!(out.len(), mel::SAMPLE_RATE); // 48k stereo -> 16k mono, exactly one second
        assert!(out.iter().all(|v| v.abs() <= 0.5));
    }

    #[test]
    fn resampler_fractional_ratio_44k_keeps_state_across_chunks() {
        let mut rs = Resampler::new(1, 44100);
        let input = tone(1.0, 0.5, 44100, 1);
        let mut out = Vec::new();
        for chunk in input.chunks(441) {
            out.extend(rs.push(chunk));
        }
        let expected = mel::SAMPLE_RATE as i64;
        assert!((out.len() as i64 - expected).abs() <= 2, "got {} samples", out.len());
    }

    #[test]
    fn vad_cuts_speech_at_a_pause_and_drops_short_blips() {
        let mut vad = Vad::new(&args(0.01));
        let frame = vad.frame;
        let mut audio = vec![0.0f32; mel::SAMPLE_RATE / 2]; // 0.5 s silence
        audio.extend(tone(1.0, 0.3, mel::SAMPLE_RATE, 1)); // 1 s speech
        audio.extend(vec![0.0f32; mel::SAMPLE_RATE]); // 1 s silence closes it
        audio.extend(tone(0.04, 0.3, mel::SAMPLE_RATE, 1)); // 40 ms blip: below min_speech
        audio.extend(vec![0.0f32; mel::SAMPLE_RATE]);
        let utterances: Vec<Vec<f32>> = audio.chunks_exact(frame).filter_map(|f| vad.push(f)).collect();
        assert_eq!(utterances.len(), 1, "one utterance expected");
        let secs = utterances[0].len() as f32 / mel::SAMPLE_RATE as f32;
        // 300 ms preroll + 1 s speech + 300 ms trailing silence, give or take a frame.
        assert!((1.5..1.7).contains(&secs), "utterance is {secs} s");
        assert!(vad.peak > 0.1);
    }

    #[test]
    fn vad_reports_speech_in_progress() {
        let mut vad = Vad::new(&args(0.01));
        let frame = vad.frame;
        assert!(!vad.in_speech());
        for f in tone(0.5, 0.3, mel::SAMPLE_RATE, 1).chunks_exact(frame) {
            assert!(vad.push(f).is_none());
        }
        assert!(vad.in_speech());
        // 300 ms of preroll silence is not kept when speech starts at the very beginning.
        assert!((0.45..0.85).contains(&vad.current_secs()), "{} s", vad.current_secs());
        vad.reset();
        assert!(!vad.in_speech());
        assert_eq!(vad.current_secs(), 0.0);
    }

    #[test]
    fn vad_meter_reflects_level() {
        let mut vad = Vad::new(&args(0.01));
        let frame = vad.frame;
        for f in vec![0.0f32; frame * 5].chunks_exact(frame) {
            vad.push(f);
        }
        assert_eq!(vad.meter(), "▁▁▁▁▁");
        for f in tone(0.1, 0.5, mel::SAMPLE_RATE, 1).chunks_exact(frame) {
            vad.push(f);
        }
        assert!(vad.meter().ends_with('█'), "{}", vad.meter());
        assert!(vad.meter().chars().count() <= 20);
    }

    #[test]
    fn vad_caps_an_utterance_at_thirty_seconds() {
        let mut vad = Vad::new(&args(0.01));
        let frame = vad.frame;
        let audio = tone(31.0, 0.3, mel::SAMPLE_RATE, 1);
        let utterances: Vec<Vec<f32>> = audio.chunks_exact(frame).filter_map(|f| vad.push(f)).collect();
        assert!(!utterances.is_empty());
        assert!(utterances[0].len() <= mel::N_SAMPLES + frame);
    }

    #[test]
    fn wav_round_trip_to_16k_mono() {
        let path = std::env::temp_dir().join(format!("listen-test-{}.wav", std::process::id()));
        let spec = hound::WavSpec { channels: 2, sample_rate: 48000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for v in tone(0.5, 0.5, 48000, 2) {
            w.write_sample((v * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
        let audio = read_wav(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(audio.len(), mel::SAMPLE_RATE / 2);
        let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((0.4..=0.5).contains(&peak), "peak {peak}");
    }

    #[test]
    fn no_quit_requested_initially() {
        assert!(quit_requested().is_none());
    }
}
