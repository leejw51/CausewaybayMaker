//! Jarvis: a voice assistant that lives on this Mac.
//!
//!   make jarvis                # listen; print what you said; answer out loud
//!   make jarvis WAV=clip.wav   # replay a recording through the whole pipeline (no mic needed)
//!
//! Ears:   Whisper on MLX (the qwen35 library, shared with `make audio`).
//! Brain:  Apple's on-device foundation model (Apple Intelligence), one `apple-ai` helper process
//!         that keeps the conversation and also does the routing below. `--brain mlx` runs
//!         Qwen3.5 on MLX in this process instead.
//! Eyes:   "what is on my screen?" takes a screenshot and asks Qwen3-VL on MLX (python/see.py);
//!         Apple's Vision framework reads the text instead when the vision model is unavailable.
//! Voice:  Kokoro on MLX (python/speak.py); macOS `say` for languages Kokoro lacks (Korean).
//!         Each sentence is spoken as soon as it is written, while the rest is still being generated.
//! Duplex: the main thread only listens and transcribes; a worker thread thinks and speaks. So
//!         Jarvis hears you while it talks or thinks: talk over it and it stops at once and
//!         answers the new question; what the mic picks up of its own voice is recognised and dropped.
//! Router: Apple's on-device model decides, with a guided yes/no, whether a question needs live
//!         information. Only then, or when you start with "AI, ...", does the cloud answer.
//! Cloud:  Grok (xAI, XAI_API_KEY) with web search. Never used otherwise.
//! Pictures: "imagine a red apple" makes one (Apple's Image Playground, or FLUX.1-schnell on MLX
//!         via python/imagine.py) and shows it in the terminal (Kitty graphics protocol);
//!         "describe ~/pic.jpg" or a URL reads a picture with Apple Vision and shows it too.
//! Typing: besides the microphone you can type requests into the terminal (paths and URLs are
//!         easier typed than spoken).

mod apple;
mod brain;
mod cloud;
mod eyes;
mod helper;
mod pictures;
mod show;
mod voice;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use clap::Parser;
use qwen35::audio::{listen, stt};

use apple::AppleChat;
use brain::{Brain, PERSONA};
use cloud::Grok;
use eyes::Eyes;
use helper::Helper;
use show::Protocol;
use voice::{SpokenLog, Voice, recently_spoken};

#[derive(Parser)]
#[command(about = "Jarvis: voice assistant on Apple Silicon (Whisper, Qwen3.5, Qwen3-VL and Kokoro on MLX)")]
struct Args {
    /// Whisper model (mlx-community/whisper-*)
    #[arg(long, default_value = "mlx-community/whisper-large-v3-turbo")]
    whisper: String,
    /// Spoken language code, e.g. en, ko, ja, zh
    #[arg(long, default_value = "en")]
    language: String,
    /// Which model thinks: "apple" (Apple's on-device model) or "mlx" (Qwen3.5 in this process)
    #[arg(long, default_value = "apple")]
    brain: String,
    /// Language model for --brain mlx (an mlx-community Qwen3.5 repo)
    #[arg(long, default_value = "mlx-community/Qwen3.5-9B-4bit")]
    llm: String,
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,
    /// Sampling temperature for the brain (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    #[arg(long, default_value_t = 20)]
    top_k: i32,
    /// Vision-language model for the eyes (mlx-vlm)
    #[arg(long, default_value = "mlx-community/Qwen3-VL-4B-Instruct-4bit")]
    vlm: String,
    /// Do not start the vision model (screen questions fall back to Apple Vision OCR)
    #[arg(long)]
    no_eyes: bool,
    /// Text-to-speech model for the voice (mlx-audio, Kokoro)
    #[arg(long, default_value = "mlx-community/Kokoro-82M-bf16")]
    tts: String,
    /// Do not speak replies
    #[arg(long)]
    no_speak: bool,
    /// Voice name: a Kokoro voice (af_heart, am_adam, bf_emma, ...) or a `say` voice
    #[arg(long)]
    voice: Option<String>,
    /// Playback volume for Kokoro speech, 0 to 1
    #[arg(long, default_value_t = 1.0)]
    volume: f32,
    /// Ignore the microphone while Jarvis is thinking or speaking (no interrupting, no echo
    /// filtering); use this if the speakers leak into the mic badly
    #[arg(long)]
    half_duplex: bool,
    /// Image model for "imagine ..." when Image Playground is unavailable (mflux, FLUX.1-schnell)
    #[arg(long, default_value = "mflux-community/flux-1-schnell-mflux-q4")]
    imagine_model: String,
    /// Do not start the FLUX image model (Image Playground only)
    #[arg(long)]
    no_imagine: bool,
    /// How to show pictures in the terminal: auto, kitty, iterm or none
    #[arg(long, default_value = "auto")]
    image_protocol: String,
    /// Size of pictures drawn in the terminal, in pixels (square box, aspect ratio kept)
    #[arg(long, default_value_t = 256)]
    image_size: u32,
    /// Also read typed requests from standard input (on by default when stdin is a terminal)
    #[arg(long)]
    stdin: bool,
    /// Replay this WAV through the mic path (VAD, transcription, answer) instead of listening
    #[arg(long)]
    wav: Option<PathBuf>,
    #[command(flatten)]
    listen: listen::ListenArgs,
    /// List input devices and exit
    #[arg(long)]
    list_devices: bool,
    /// Grok model for cloud answers (or set XAI_MODEL)
    #[arg(long)]
    grok_model: Option<String>,
    /// Never use the cloud, even for "AI ..." requests
    #[arg(long)]
    local_only: bool,
    /// The apple-ai helper (built by `make jarvis`); default: next to the cargo target dir
    #[arg(long)]
    apple_ai: Option<PathBuf>,
    /// Python with mlx-vlm and mlx-audio installed (the qwen35 conda env)
    #[arg(long)]
    python: Option<PathBuf>,
    /// Directory holding see.py and speak.py; default: jarvis/python next to the binary's repo
    #[arg(long)]
    scripts: Option<PathBuf>,
    /// Full model details at startup (path on disk, size) and the helpers' stderr
    #[arg(long)]
    verbose: bool,
}

// ---- where things are ----

/// rust/target/release/jarvis -> rust/target
fn target_dir() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|p| p.ancestors().nth(2).map(Path::to_path_buf))
}

fn default_apple_ai() -> PathBuf {
    target_dir().map(|t| t.join("apple-ai")).unwrap_or_else(|| "rust/target/apple-ai".into())
}

fn default_scripts() -> PathBuf {
    target_dir()
        .and_then(|t| t.parent().and_then(Path::parent).map(|repo| repo.join("jarvis/python")))
        .unwrap_or_else(|| "jarvis/python".into())
}

fn default_python() -> PathBuf {
    let conda = PathBuf::from("/opt/anaconda3/envs/qwen35/bin/python");
    if conda.exists() { conda } else { "python3".into() }
}

// ---- what the two threads share ----

const IDLE: u8 = 0;
const THINKING: u8 = 1;
const SPEAKING: u8 = 2;

struct Shared {
    /// What the worker is doing right now.
    phase: AtomicU8,
    /// Number of the latest request; a request whose number is older has been interrupted.
    latest: AtomicU64,
    /// Highest request whose reply has been fully generated (it may still be being spoken).
    generated: AtomicU64,
    /// Highest request that is completely finished (spoken, dropped or interrupted).
    finished: AtomicU64,
    /// Utterances cut from the microphone, and how many the transcription thread has handled.
    heard: AtomicU64,
    transcribed: AtomicU64,
    /// The worker is in the middle of printing a `<<<` line.
    line_open: AtomicBool,
    /// What Jarvis has said recently, so the listener can recognise its own voice.
    spoken: SpokenLog,
}

impl Shared {
    fn phase(&self) -> u8 {
        self.phase.load(Ordering::SeqCst)
    }

    /// Every utterance heard so far has been transcribed.
    fn caught_up(&self) -> bool {
        self.transcribed.load(Ordering::SeqCst) >= self.heard.load(Ordering::SeqCst)
    }
}

// ---- the listening side ----

/// Main thread: only captures audio and cuts it into utterances; each utterance goes to the
/// transcription thread at once, so the microphone, the level meter and interruptions stay
/// responsive however busy the GPU is.
struct Ears {
    audio_tx: Sender<Vec<f32>>,
    shared: Arc<Shared>,
    /// Wait for the reply to be fully spoken before listening again (no interruptions).
    half_duplex: bool,
    /// Replaying a file: wait until the reply is written before feeding the next utterance,
    /// so the replay behaves like a conversation at human pace.
    replay: bool,
}

impl Ears {
    fn handle(&mut self, audio: &[f32]) -> Result<()> {
        self.shared.heard.fetch_add(1, Ordering::SeqCst);
        if self.audio_tx.send(audio.to_vec()).is_err() {
            bail!("the transcription thread has stopped");
        }
        if self.half_duplex {
            self.wait_until(|s| s.caught_up() && s.finished.load(Ordering::SeqCst) >= s.latest.load(Ordering::SeqCst));
        } else if self.replay {
            self.wait_until(|s| s.caught_up() && s.generated.load(Ordering::SeqCst) >= s.latest.load(Ordering::SeqCst));
        }
        Ok(())
    }

    fn wait_until(&self, done: impl Fn(&Shared) -> bool) {
        while !done(&self.shared) && listen::quit_requested().is_none() {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// Transcription thread: Whisper turns each utterance into text, drops Jarvis's own voice coming
/// back through the microphone, and hands real requests to the worker.
/// MLX binds a stream's state to the thread that made it, so a model must be loaded on the
/// thread that will run it. The transcription thread therefore loads Whisper itself and reports
/// the model's facts (or the error) back through `loaded`.
fn transcribe_loop(
    whisper: String,
    language: String,
    verbose: bool,
    loaded: Sender<Result<String>>,
    rx: Receiver<Vec<f32>>,
    shared: Arc<Shared>,
    tx: Sender<(u64, String)>,
) {
    // If this thread dies (a panic in the model), Jarvis cannot hear any more: end the program
    // instead of letting the listener wait forever.
    struct EndOnExit;
    impl Drop for EndOnExit {
        fn drop(&mut self) {
            if listen::quit_requested().is_none() {
                eprintln!("[ears: the transcription thread stopped; shutting down]");
                listen::request_quit();
            }
        }
    }
    let _guard = EndOnExit;
    qwen35::with_mlx_stream(|| {
        let stt = match stt::Transcriber::load(&whisper, &language) {
            Ok(stt) => {
                let _ = loaded.send(Ok(if verbose { stt.info.clone() } else { stt.info_short.clone() }));
                stt
            }
            Err(e) => {
                let _ = loaded.send(Err(e));
                return;
            }
        };
        if std::env::var_os("JARVIS_TEST_PANIC").is_some_and(|v| v == "ears") {
            panic!("test: the transcription thread dies");
        }
        transcribe_all(stt, rx, shared, tx)
    });
}

fn transcribe_all(stt: stt::Transcriber, rx: Receiver<Vec<f32>>, shared: Arc<Shared>, tx: Sender<(u64, String)>) {
    for audio in rx {
        let result = (|| -> Result<()> {
            if shared.line_open.swap(false, Ordering::SeqCst) {
                println!(); // do not write the transcript into the middle of a reply line
            }
            eprintln!("[ai local: Whisper (ears) · {}]", stt.info_short);
            let text = stt.transcribe(&audio)?;
            if text.chars().filter(|c| c.is_alphanumeric()).count() < 2 {
                return Ok(());
            }
            if let Some(spoken) = recently_spoken(&shared.spoken, Duration::from_secs(3)) {
                if is_echo(&text, &spoken) {
                    eprintln!("[ignored: that was my own voice]");
                    return Ok(());
                }
            }
            submit(&shared, &tx, text)?;
            Ok(())
        })();
        shared.transcribed.fetch_add(1, Ordering::SeqCst);
        if let Err(e) = result {
            eprintln!("[ears: {e}]");
        }
        if listen::quit_requested().is_some() {
            break;
        }
    }
}

/// Hand a request to the worker, interrupting whatever it was doing. Returns the request id.
fn submit(shared: &Shared, tx: &Sender<(u64, String)>, text: String) -> Result<u64> {
    if shared.phase() != IDLE {
        eprintln!("[interrupted]");
    }
    let id = shared.latest.fetch_add(1, Ordering::SeqCst) + 1;
    if tx.send((id, text)).is_err() {
        bail!("the assistant thread has stopped");
    }
    Ok(id)
}

/// Typed requests: one per line on stdin, same path as speech. When stdin is not a terminal
/// (a test piping commands in), its end also ends Jarvis once the replies are done.
fn read_typed_input(shared: Arc<Shared>, tx: Sender<(u64, String)>, interactive: bool) {
    typed_requests(std::io::stdin().lock(), shared, tx, interactive);
}

fn typed_requests(mut input: impl std::io::BufRead, shared: Arc<Shared>, tx: Sender<(u64, String)>, interactive: bool) {
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let clean = show::strip_terminal_replies(&line);
        let text = clean.trim();
        if text.is_empty() {
            continue;
        }
        if shared.line_open.swap(false, Ordering::SeqCst) {
            println!();
        }
        println!("USER: {text}");
        let Ok(id) = submit(&shared, &tx, text.to_string()) else { break };
        if !interactive {
            // Piped requests run one after another, like a conversation, not as interruptions.
            while shared.finished.load(Ordering::SeqCst) < id && listen::quit_requested().is_none() {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
    if !interactive {
        while shared.finished.load(Ordering::SeqCst) < shared.latest.load(Ordering::SeqCst) && listen::quit_requested().is_none() {
            std::thread::sleep(Duration::from_millis(50));
        }
        listen::request_quit();
    }
}

// ---- the thinking and speaking side (worker thread) ----

/// How the worker should get its brain: Apple's model (already running) or Qwen loaded on the
/// worker thread.
enum BrainSpec {
    Apple,
    Mlx { model: String, temp: f32, top_k: i32, max_tokens: usize, instructions: String },
}

struct Worker {
    brain: Brain,
    brain_spec: BrainSpec,
    /// Reports the brain's facts (or the load error) once the worker is ready.
    loaded: Option<Sender<Result<String>>>,
    /// Qwen3.5 on MLX, loaded on first need when the Apple brain fails (the answer stays local).
    fallback: Option<Brain>,
    fallback_spec: BrainSpec,
    /// The one apple-ai process: the Apple brain, the cloud router, and Vision OCR.
    apple: Option<AppleChat>,
    eyes: Option<Eyes>,
    voice: Option<Voice>,
    cloud: Option<Grok>,
    /// FLUX.1-schnell on MLX (python/imagine.py), the picture maker when Image Playground is not available.
    flux: Option<Helper>,
    /// Image Playground worked (Some(true)), failed as unsupported (Some(false)), or untried.
    playground: Option<bool>,
    protocol: Protocol,
    image_size: u32,
    /// Full model facts in the startup table.
    verbose: bool,
    shared: Arc<Shared>,
    rx: Receiver<(u64, String)>,
}

impl Worker {
    fn run(self) {
        // The Qwen brain evaluates MLX arrays on this thread: load it here, on its own stream.
        qwen35::with_mlx_stream(|| self.serve_forever());
    }

    fn serve_forever(mut self) {
        let loaded = self.loaded.take();
        let verbose = self.verbose;
        let info = match &self.brain_spec {
            BrainSpec::Apple => Ok(self.brain_info(verbose)),
            BrainSpec::Mlx { model, temp, top_k, max_tokens, instructions } => match Brain::load_mlx(model, *temp, *top_k, *max_tokens, instructions) {
                Ok(brain) => {
                    self.brain = brain;
                    Ok(self.brain_info(verbose))
                }
                Err(e) => Err(e),
            },
        };
        let failed = info.is_err();
        if let Some(l) = loaded {
            let _ = l.send(info);
        }
        if failed {
            return;
        }
        loop {
            match self.rx.recv_timeout(Duration::from_millis(50)) {
                Ok((id, text)) => self.serve(id, &text),
                Err(RecvTimeoutError::Timeout) => {
                    if self.voice.as_mut().is_some_and(|v| v.busy()) {
                        self.shared.phase.store(SPEAKING, Ordering::SeqCst);
                    } else {
                        self.shared.phase.store(IDLE, Ordering::SeqCst);
                    }
                    if listen::quit_requested().is_some() {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        if let Some(v) = &mut self.voice {
            v.stop();
        }
        // Dropping self kills the helpers (apple-ai, see.py, speak.py).
    }

    /// The brain's facts: full (path, size) for the startup table with --verbose, else short.
    fn brain_info(&self, verbose: bool) -> String {
        match (&self.brain, verbose) {
            (Brain::Apple, true) => self.apple.as_ref().map(|a| a.info.clone()).unwrap_or_default(),
            (Brain::Apple, false) => self.apple.as_ref().map(|a| a.info_short.clone()).unwrap_or_default(),
            (Brain::Mlx(chat), true) => chat.engine.info.clone(),
            (Brain::Mlx(chat), false) => chat.engine.info_short.clone(),
        }
    }

    fn interrupted(&self, id: u64) -> bool {
        self.shared.latest.load(Ordering::SeqCst) != id || listen::quit_requested().is_some()
    }

    /// Answer one request, speaking each sentence as soon as it is complete.
    fn serve(&mut self, id: u64, text: &str) {
        self.shared.phase.store(THINKING, Ordering::SeqCst);
        if let Some(v) = &mut self.voice {
            v.stop(); // whatever was still being said belongs to the previous question
        }
        let reply = self.answer(id, text);
        self.shared.generated.store(id, Ordering::SeqCst);
        let mut stopped = false;
        if !reply.is_empty() && !self.interrupted(id) {
            self.shared.phase.store(SPEAKING, Ordering::SeqCst);
            // Keep playback moving until it is over or the user says something else.
            while self.voice.as_mut().is_some_and(|v| v.busy()) {
                if self.interrupted(id) {
                    stopped = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(30));
            }
        } else {
            stopped = true;
        }
        if stopped {
            if let Some(v) = &mut self.voice {
                v.stop();
            }
        }
        self.shared.phase.store(IDLE, Ordering::SeqCst);
        self.shared.finished.store(id, Ordering::SeqCst);
    }

    /// Print `AI: ` (or `CLOUD AI: `) + the streamed reply and speak it sentence by sentence;
    /// returns the text.
    fn stream_out(&mut self, id: u64, label: &str, produce: impl FnMut(&mut dyn FnMut(&str) -> bool) -> Result<String>) -> String {
        self.stream_out_with(id, label, false, produce)
    }

    /// `speak_after`: queue the sentences and synthesize them only once the text is complete.
    /// Used when the brain itself runs on the GPU (Qwen on MLX): Kokoro and Qwen fighting for
    /// the GPU made generation ten times slower than speaking afterwards.
    fn stream_out_with(
        &mut self,
        id: u64,
        label: &str,
        speak_after: bool,
        mut produce: impl FnMut(&mut dyn FnMut(&str) -> bool) -> Result<String>,
    ) -> String {
        let shared = self.shared.clone();
        let mut voice = self.voice.take();
        let mut buf = String::new();
        let mut started = false;
        let mut deferred: Vec<String> = Vec::new();
        let stdout = std::io::stdout();
        let mut on_text = |t: &str| {
            if shared.latest.load(Ordering::SeqCst) != id || listen::quit_requested().is_some() {
                return false;
            }
            let mut lock = stdout.lock();
            if !started {
                let _ = write!(lock, "{label} ");
                shared.line_open.store(true, Ordering::SeqCst);
                started = true;
            }
            let _ = write!(lock, "{t}");
            let _ = lock.flush();
            drop(lock);
            buf.push_str(t);
            if let Some(v) = &mut voice {
                for sentence in take_sentences(&mut buf) {
                    if speak_after {
                        deferred.push(sentence);
                    } else {
                        v.say(&sentence);
                    }
                }
                v.pump();
            }
            true
        };
        let result = produce(&mut on_text);
        drop(on_text);
        let cut = self.interrupted(id);
        if let Some(v) = &mut voice {
            if !cut {
                for sentence in deferred.drain(..) {
                    v.say(&sentence);
                }
                if !buf.trim().is_empty() {
                    v.say(&buf); // the last sentence, or a reply without punctuation
                }
            }
        }
        if started && shared.line_open.swap(false, Ordering::SeqCst) {
            println!();
        }
        // The voice's trace is printed once the reply line is closed, not in the middle of it.
        if let Some(v) = &mut voice {
            if let Some(note) = v.take_note() {
                eprintln!("{note}");
            }
        }
        self.voice = voice;
        match result {
            Ok(text) => text,
            Err(e) => {
                eprintln!("[brain: {}]", brain::brief(&e));
                String::new()
            }
        }
    }

    /// Decide who answers and produce the reply (printed and spoken as it comes).
    fn answer(&mut self, id: u64, text: &str) -> String {
        if let Some((prompt, style)) = pictures::imagine_request(text) {
            return self.imagine(id, &prompt, style);
        }
        if let Some((source, question)) = pictures::image_reference(text) {
            return self.describe_picture(id, &source, &question);
        }
        // "AI, ..." is an explicit request for the cloud.
        if let Some(q) = strip_ai_prefix(text) {
            if self.cloud.is_some() {
                return self.cloud(id, &q, "you asked for AI");
            }
            eprintln!("[\"AI\" request, but no XAI_API_KEY is set: answering on device]");
            return self.think(id, &q);
        }
        if eyes::wants_screen(text) {
            return self.look(id, text);
        }
        // Otherwise the cloud is used only when the on-device router says it is needed.
        if self.cloud.is_some() {
            if let Some(router) = &mut self.apple {
                match router.needs_cloud(text) {
                    Ok(true) => {
                        eprintln!("[ai local: Apple on-device model (router): cloud needed = yes]");
                        return self.cloud(id, text, "needs live information");
                    }
                    Ok(false) => eprintln!("[ai local: Apple on-device model (router): cloud needed = no]"),
                    Err(e) => eprintln!("[router: {}]", brain::brief(&e)),
                }
            }
        }
        self.think(id, text)
    }

    fn think(&mut self, id: u64, prompt: &str) -> String {
        if std::env::var_os("JARVIS_TEST_FAIL_BRAIN").is_none() {
            eprintln!("[ai local: {} (brain) · {}]", self.brain.name(), self.brain_info(false));
            let mut brain = std::mem::replace(&mut self.brain, Brain::Apple);
            let on_gpu = matches!(brain, Brain::Mlx(_));
            let mut apple = self.apple.take();
            let mut note = None;
            let mut failure = None;
            let out = self.stream_out_with(id, "AI:", on_gpu, |on_text| {
                let (text, n) = brain.reply(prompt, apple.as_mut(), on_text)?;
                note = n;
                Ok(text)
            });
            if let Some(n) = note {
                eprintln!("{n}");
            }
            self.brain = brain;
            self.apple = apple;
            if !out.is_empty() || self.interrupted(id) {
                return out;
            }
            // An empty reply means the brain failed (stream_out printed the error); Apple's model
            // can fail when a system model is missing. Try the fallback below.
            failure = Some(());
            if failure.is_none() || matches!(self.brain, Brain::Mlx(_)) {
                return out;
            }
        }
        // Fall back to Qwen3.5 on MLX, loaded on this thread on first need.
        if self.fallback.is_none() {
            if let BrainSpec::Mlx { model, temp, top_k, max_tokens, instructions } = &self.fallback_spec {
                eprintln!("[brain: the Apple on-device model failed; loading Qwen3.5 on MLX as the local fallback]");
                match Brain::load_mlx(model, *temp, *top_k, *max_tokens, instructions) {
                    Ok(b) => self.fallback = Some(b),
                    Err(e) => {
                        eprintln!("[brain: fallback failed too: {}]", brain::brief(&e));
                        return String::new();
                    }
                }
            }
        }
        let Some(mut fallback) = self.fallback.take() else { return String::new() };
        eprintln!("[ai local: {} (brain, fallback) · {}]", fallback.name(), match &fallback {
            Brain::Mlx(chat) => chat.engine.info_short.clone(),
            Brain::Apple => String::new(),
        });
        let mut note = None;
        let out = self.stream_out_with(id, "AI:", true, |on_text| {
            let (text, n) = fallback.reply(prompt, None, on_text)?;
            note = n;
            Ok(text)
        });
        if let Some(n) = note {
            eprintln!("{n}");
        }
        self.fallback = Some(fallback);
        out
    }

    fn look(&mut self, id: u64, question: &str) -> String {
        let png = match eyes::screenshot() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[screen: {e}]");
                return self.think(id, question);
            }
        };
        let seen = match &mut self.eyes {
            Some(eyes) => match eyes.look(&png, question) {
                Ok(reply) => Some(reply),
                Err(e) => {
                    eprintln!("[eyes: {e}; reading the screen with Apple Vision instead]");
                    None
                }
            },
            None => None,
        };
        let reply = match seen {
            Some(reply) => self.stream_out(id, "AI:", |on_text| {
                on_text(&reply);
                Ok(reply.clone())
            }),
            None => self.look_with_ocr(id, &png, question),
        };
        let _ = std::fs::remove_file(&png);
        reply
    }

    /// No vision model: OCR the screenshot with Apple Vision and let the brain answer from the text.
    fn look_with_ocr(&mut self, id: u64, png: &Path, question: &str) -> String {
        let Some(apple) = &self.apple else {
            eprintln!("[screen: no vision model and no apple-ai helper]");
            return self.think(id, question);
        };
        match apple.describe_image(png, 1500) {
            Ok(desc) => {
                eprintln!("[screen: read with Apple Vision]");
                self.think(id, &format!("{question}\n\n(Answer using this description of the user's screen right now:)\n{desc}"))
            }
            Err(e) => {
                eprintln!("[screen: {e}]");
                self.think(id, question)
            }
        }
    }

    /// Make a picture, show it in the terminal, and say so.
    fn imagine(&mut self, id: u64, prompt: &str, style: &str) -> String {
        eprintln!("[imagining: {prompt} ({style})]");
        let out = pictures::picture_path(prompt);
        let apple = self.apple.as_ref().filter(|_| self.playground != Some(false));
        let made = pictures::generate(apple, self.flux.as_mut(), prompt, style, &out);
        match made {
            Ok(engine) => {
                if self.playground.is_none() {
                    self.playground = Some(engine == "Image Playground");
                }
                if let Err(e) = show::show(self.protocol, &out, self.image_size) {
                    eprintln!("[show: {e}]");
                }
                println!("Saved {}", out.display());
                let reply = format!("Here is your picture of {prompt}.");
                self.stream_out(id, "AI:", |on_text| {
                    on_text(&reply);
                    Ok(reply.clone())
                })
            }
            Err(e) => {
                eprintln!("[imagine: {e}]");
                let reply = "Sorry, I could not make that picture.".to_string();
                self.stream_out(id, "AI:", |on_text| {
                    on_text(&reply);
                    Ok(reply.clone())
                })
            }
        }
    }

    /// Look at a picture from a path or URL with Apple Vision, show it, and answer about it.
    fn describe_picture(&mut self, id: u64, source: &str, question: &str) -> String {
        let (path, downloaded) = match pictures::fetch(source) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("[picture: {e}]");
                let reply = format!("I could not open {source}.");
                return self.stream_out(id, "AI:", |on_text| {
                    on_text(&reply);
                    Ok(reply.clone())
                });
            }
        };
        if let Err(e) = show::show(self.protocol, &path, self.image_size) {
            eprintln!("[show: {e}]");
        }
        let seen = match &self.apple {
            Some(apple) => pictures::describe_with_vision(apple, &path),
            None => Err(anyhow::anyhow!("the apple-ai helper is not running")),
        };
        if downloaded {
            let _ = std::fs::remove_file(&path);
        }
        match seen {
            Ok(desc) => {
                eprintln!("[vision: {desc}]");
                self.think(
                    id,
                    &format!(
                        "{question}\n\n(The user shared a picture. Apple Vision found: {desc} Answer from this in one or \
                         two spoken sentences; if asked to describe it, describe what it shows.)"
                    ),
                )
            }
            Err(e) => {
                eprintln!("[vision: {e}]");
                self.think(id, question)
            }
        }
    }

    fn cloud(&mut self, id: u64, prompt: &str, why: &str) -> String {
        let shared = self.shared.clone();
        let mut grok = self.cloud.take().expect("cloud() needs a key");
        eprintln!("[ai cloud: {} via xAI with web search · api.x.ai/v1/responses ({why})]", grok.model);
        let cancelled = || shared.latest.load(Ordering::SeqCst) != id;
        let out = match grok.ask(prompt, &cancelled) {
            Ok(reply) => self.stream_out(id, "CLOUD AI:", |on_text| {
                on_text(&reply);
                Ok(reply.clone())
            }),
            Err(e) => {
                eprintln!("[cloud: {e}]");
                String::new()
            }
        };
        self.cloud = Some(grok);
        out
    }
}

/// Pop the complete sentences off the front of `buf` (a sentence ends with . ! ? or their
/// East Asian forms followed by whitespace), leaving the unfinished rest in place.
fn take_sentences(buf: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes: Vec<(usize, char)> = buf.char_indices().collect();
    let mut i = 0;
    while i < bytes.len() {
        let (pos, c) = bytes[i];
        let ends = matches!(c, '.' | '!' | '?' | '。' | '！' | '？');
        let next_is_space = bytes.get(i + 1).is_none_or(|(_, n)| n.is_whitespace());
        // Only cut after whitespace: "3.5" or "e.g." must not be split, and the final sentence
        // waits for what follows (or the end of the reply).
        if ends && next_is_space && bytes.get(i + 1).is_some() {
            let sentence = buf[start..pos + c.len_utf8()].trim();
            if sentence.chars().filter(|c| c.is_alphanumeric()).count() >= 2 {
                out.push(sentence.to_string());
                start = pos + c.len_utf8();
            }
        }
        i += 1;
    }
    if start > 0 {
        buf.drain(..start);
    }
    out
}

/// Words of `text`, lower-cased, letters and digits only; small number words become digits so
/// "two plus two is four" (as spoken) matches "2 plus 2 is 4" (as Whisper writes it).
fn words(text: &str) -> Vec<String> {
    const NUMBERS: [(&str, &str); 21] = [
        ("zero", "0"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"), ("six", "6"),
        ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"), ("eleven", "11"), ("twelve", "12"),
        ("thirteen", "13"), ("fourteen", "14"), ("fifteen", "15"), ("sixteen", "16"), ("seventeen", "17"),
        ("eighteen", "18"), ("nineteen", "19"), ("twenty", "20"),
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let w = w.to_lowercase();
            NUMBERS.iter().find(|(name, _)| *name == w).map(|(_, d)| d.to_string()).unwrap_or(w)
        })
        .collect()
}

/// Is `heard` just the microphone picking up `spoken` (Jarvis's own reply)? True when most of
/// the heard words occur in what was spoken. A real interruption shares few words with it.
fn is_echo(heard: &str, spoken: &str) -> bool {
    let heard = words(heard);
    if heard.is_empty() {
        return true;
    }
    let spoken = words(spoken);
    let hits = heard.iter().filter(|w| spoken.contains(w)).count();
    let threshold = if heard.len() <= 3 { heard.len() } else { heard.len() * 6 / 10 };
    hits >= threshold
}

/// "AI, what's the weather?" -> Some("what's the weather?"). Speech recognition writes the wake
/// word as "AI", "A.I.", "A I" or "Hey AI", so compare on letters only.
fn strip_ai_prefix(text: &str) -> Option<String> {
    for wake in ["heyai", "ai"] {
        let mut matched = 0;
        let mut end = None;
        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() {
                let want = wake.as_bytes()[matched] as char;
                if c.to_lowercase().next() != Some(want) {
                    break;
                }
                matched += 1;
                if matched == wake.len() {
                    end = Some(i + c.len_utf8());
                    break;
                }
            }
        }
        let Some(end) = end else { continue };
        let rest = &text[end..];
        // The wake word must be a whole word ("AI," yes, "Air" no).
        if rest.chars().next().is_some_and(char::is_alphanumeric) {
            continue;
        }
        let rest = rest.trim_start_matches(|c: char| !c.is_alphanumeric()).trim();
        if !rest.is_empty() {
            return Some(rest.to_string());
        }
    }
    None
}

fn main() -> Result<()> {
    let args = Args::parse();
    listen::quit_cleanly_on_signals();
    if args.list_devices {
        return listen::list_devices();
    }
    let apple_ai = args.apple_ai.clone().unwrap_or_else(default_apple_ai);
    let python = args.python.clone().unwrap_or_else(default_python);
    let scripts = args.scripts.clone().unwrap_or_else(default_scripts);

    // Helpers first, so their model loading overlaps with ours.
    let mut eyes = if args.no_eyes {
        None
    } else {
        match Eyes::start(&python, &scripts.join("see.py"), &args.vlm, PERSONA, args.verbose) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("[eyes unavailable: {e}]");
                None
            }
        }
    };
    let kokoro = if args.no_speak {
        None
    } else {
        Voice::start_kokoro(&python, &scripts.join("speak.py"), &args.tts, args.verbose)
            .map_err(|e| eprintln!("[Kokoro unavailable, using say: {e}]"))
            .ok()
    };
    let protocol = Protocol::parse(&args.image_protocol)?;
    let flux = if args.no_imagine {
        None
    } else {
        Helper::start(&python, &scripts.join("imagine.py"), &["--model".to_string(), args.imagine_model.clone()], args.verbose)
            .map_err(|e| eprintln!("[FLUX image model unavailable: {e}]"))
            .ok()
    };
    // The cloud is decided first: the brain's instructions tell it whether the cloud exists.
    let cloud = if args.local_only { None } else { Grok::from_env(args.grok_model.clone()) };
    let instructions = brain::persona(cloud.as_ref().map(|g| g.model.as_str()));
    // One apple-ai process serves as the brain, the cloud router and the OCR fallback.
    let apple = match AppleChat::start(&apple_ai, &instructions) {
        Ok(a) => Some(a),
        Err(e) if args.brain == "apple" => return Err(e.context("the Apple brain needs the apple-ai helper")),
        Err(e) => {
            eprintln!("[apple-ai unavailable, so no cloud routing: {e}]");
            None
        }
    };

    let brain_spec = match args.brain.as_str() {
        "apple" => BrainSpec::Apple,
        "mlx" => BrainSpec::Mlx {
            model: args.llm.clone(),
            temp: args.temp,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
            instructions: instructions.clone(),
        },
        other => bail!("--brain must be apple or mlx, not {other:?}"),
    };
    if let Some(e) = &mut eyes {
        if let Err(err) = e.wait_ready() {
            eprintln!("[eyes unavailable: {err}]");
            eyes = None;
        }
    }
    let shared = Arc::new(Shared {
        phase: AtomicU8::new(IDLE),
        latest: AtomicU64::new(0),
        generated: AtomicU64::new(0),
        finished: AtomicU64::new(0),
        heard: AtomicU64::new(0),
        transcribed: AtomicU64::new(0),
        line_open: AtomicBool::new(false),
        spoken: SpokenLog::default(),
    });
    let mut voice = (!args.no_speak).then(|| Voice::new(kokoro, &args.language, args.voice.clone(), shared.spoken.clone()));
    if let Some(v) = &mut voice {
        if let Err(e) = v.wait_ready() {
            eprintln!("[Kokoro unavailable, using say: {e}]");
            *v = Voice::new(None, &args.language, args.voice.clone(), shared.spoken.clone());
        }
        v.volume = args.volume.clamp(0.0, 1.0);
        v.verbose = args.verbose;
    }
    let mut flux = flux;
    let flux_info = match &mut flux {
        Some(f) => f
            .wait_ready()
            .map(|_| if args.verbose { f.info.clone() } else { f.info_short.clone() })
            .unwrap_or_else(|e| format!("unavailable: {e}")),
        None => "off (--no-imagine)".into(),
    };
    let apple_status = apple::status(&apple_ai).unwrap_or_default();
    let playground = apple_status["image_playground"].as_str().unwrap_or("unknown").to_string();
    let macos = voice::macos_version();
    // Facts for the startup summary, taken before the worker thread owns these.
    let has_apple = apple.is_some();
    let eyes_info = eyes.as_ref().map(|e| if args.verbose { e.info().to_string() } else { e.info_short().to_string() });
    let voice_name = voice.as_ref().map(|v| v.name());
    let cloud_model = cloud.as_ref().map(|g| g.model.clone());

    // The two model threads: ears (Whisper) and the worker (brain, eyes, voice, pictures, cloud).
    // Each loads its own MLX model on its own thread and reports back before the summary prints.
    let (tx, rx) = mpsc::channel();
    let (audio_tx, audio_rx) = mpsc::channel();
    let (ears_loaded_tx, ears_loaded_rx) = mpsc::channel();
    let stt_thread = {
        let (shared, tx) = (shared.clone(), tx.clone());
        let (whisper, language, verbose) = (args.whisper.clone(), args.language.clone(), args.verbose);
        std::thread::Builder::new()
            .name("jarvis-ears".into())
            .spawn(move || transcribe_loop(whisper, language, verbose, ears_loaded_tx, audio_rx, shared, tx))?
    };
    let (brain_loaded_tx, brain_loaded_rx) = mpsc::channel();
    let fallback_spec = BrainSpec::Mlx {
        model: args.llm.clone(),
        temp: args.temp,
        top_k: args.top_k,
        max_tokens: args.max_tokens,
        instructions: instructions.clone(),
    };
    let worker = Worker {
        brain: Brain::Apple,
        brain_spec,
        loaded: Some(brain_loaded_tx),
        fallback: None,
        fallback_spec,
        apple,
        eyes,
        voice,
        cloud,
        flux,
        playground: (playground == "available").then_some(true),
        protocol,
        image_size: args.image_size,
        verbose: args.verbose,
        shared: shared.clone(),
        rx,
    };
    let worker = std::thread::Builder::new().name("jarvis-worker".into()).spawn(move || worker.run())?;
    let ears_info = ears_loaded_rx.recv().map_err(|_| anyhow::anyhow!("the transcription thread died while loading"))??;
    let brain_info = brain_loaded_rx.recv().map_err(|_| anyhow::anyhow!("the assistant thread died while loading"))??;
    if let Some(key) = listen::quit_requested() {
        listen::announce_quit(key);
        listen::exit_now(0);
    }

    // Which model does what: the startup summary.
    eprintln!("Models (local = on this Mac{}):", if args.verbose { "" } else { "; --verbose for paths and sizes" });
    eprintln!("  ears     local  Whisper on MLX · {ears_info}");
    eprintln!("  brain    local  {} · {brain_info}", if args.brain == "apple" { "Apple on-device model" } else { "Qwen3.5 on MLX" });
    match (&cloud_model, has_apple) {
        (Some(_), true) => eprintln!("  router   local  Apple on-device model · decides when a question needs the cloud"),
        (Some(_), false) => eprintln!("  router   -      none (apple-ai unavailable): only \"AI ...\" reaches the cloud"),
        (None, _) => eprintln!("  router   -      not needed (cloud off)"),
    }
    match &eyes_info {
        Some(info) => eprintln!("  eyes     local  Qwen3-VL on MLX · {info}"),
        None => eprintln!("  eyes     -      off (screen questions use Apple Vision OCR)"),
    }
    eprintln!("  vision   local  Apple Vision framework · macOS {macos} · text, labels, faces, animals, barcodes");
    if playground == "available" {
        eprintln!("  imagine  local  Apple Image Playground · macOS {macos} (FLUX on MLX as fallback: {flux_info})");
    } else {
        eprintln!("  imagine  local  FLUX.1-schnell on MLX · {flux_info} (Image Playground: {playground})");
    }
    match &voice_name {
        Some(name) => eprintln!("  voice    local  {name}"),
        None => eprintln!("  voice    -      off (--no-speak)"),
    }
    match &cloud_model {
        Some(model) => eprintln!("  cloud    CLOUD  {model} via xAI with web search · only on \"AI ...\" or when the router asks"),
        None => eprintln!("  cloud    -      off (set XAI_API_KEY to enable Grok)"),
    }
    if !args.half_duplex && voice_name.is_some() {
        eprintln!("Full duplex: Jarvis keeps listening while it thinks and talks; just talk over it.");
    }

    // SAFETY: isatty only inspects a file descriptor.
    let stdin_is_tty = unsafe { libc::isatty(0) } == 1;
    if args.stdin || (stdin_is_tty && args.wav.is_none()) {
        eprintln!("You can also type requests here (e.g. \"imagine a red apple\", \"describe ~/Desktop/photo.jpg\").");
        let (shared, tx) = (shared.clone(), tx.clone());
        std::thread::Builder::new().name("jarvis-typed".into()).spawn(move || read_typed_input(shared, tx, stdin_is_tty))?;
    }
    drop(tx);
    let mut ears = Ears { audio_tx, shared: shared.clone(), half_duplex: args.half_duplex, replay: args.wav.is_some() };

    let result = match &args.wav {
        Some(wav) => listen::simulate(wav, &args.listen, |u| ears.handle(u)),
        None => listen::listen(&args.listen, args.half_duplex, |u| ears.handle(u)),
    };
    if listen::quit_requested().is_none() && args.wav.is_some() {
        // Replaying a file: let the last reply finish before leaving.
        ears.wait_until(|s| s.caught_up() && s.finished.load(Ordering::SeqCst) >= s.latest.load(Ordering::SeqCst));
    }
    // Closing the channels ends the transcription thread and then the worker, which stops speech
    // and kills the helpers; then leave without running global destructors (MLX and CoreAudio
    // teardown can hang the exit).
    drop(ears);
    let _ = stt_thread.join();
    let _ = worker.join();
    let code = if result.is_err() { 1 } else { 0 };
    if let Err(e) = &result {
        eprintln!("Error: {e:#}");
    }
    listen::exit_now(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_prefix_variants() {
        for t in ["AI, what time is it?", "A.I. what time is it", "a i what time is it?", "Hey AI! What time is it?"] {
            let q = strip_ai_prefix(t).unwrap();
            assert!(q.to_lowercase().starts_with("what time is it"), "{t:?} -> {q:?}");
        }
        assert_eq!(strip_ai_prefix("Air quality today?"), None);
        assert_eq!(strip_ai_prefix("What is AI?"), None);
    }

    #[test]
    fn sentences_are_cut_at_punctuation_followed_by_space() {
        let mut buf = String::from("Two plus two is 4. That is 3.5 more than");
        assert_eq!(take_sentences(&mut buf), vec!["Two plus two is 4."]);
        assert_eq!(buf, " That is 3.5 more than");
        buf.push_str(" nothing! Really? Yes.");
        assert_eq!(take_sentences(&mut buf), vec!["That is 3.5 more than nothing!", "Really?"]);
        assert_eq!(buf, " Yes."); // the final sentence waits for what follows
        let mut ko = String::from("안녕하세요. 햄릿은 셰익스피어가 썼어요. 더 궁금한");
        assert_eq!(take_sentences(&mut ko), vec!["안녕하세요.", "햄릿은 셰익스피어가 썼어요."]);
    }

    fn shared() -> Arc<Shared> {
        Arc::new(Shared {
            phase: AtomicU8::new(IDLE),
            latest: AtomicU64::new(0),
            generated: AtomicU64::new(0),
            finished: AtomicU64::new(0),
            heard: AtomicU64::new(0),
            transcribed: AtomicU64::new(0),
            line_open: AtomicBool::new(false),
            spoken: SpokenLog::default(),
        })
    }

    #[test]
    fn typed_requests_are_numbered_and_blank_lines_skipped() {
        let shared = shared();
        let (tx, rx) = mpsc::channel();
        let input = std::io::Cursor::new("first\n\n  second  \n");
        // Interactive: no waiting on the worker, no quit at the end.
        typed_requests(input, shared.clone(), tx, true);
        let got: Vec<(u64, String)> = rx.try_iter().collect();
        assert_eq!(got, vec![(1, "first".to_string()), (2, "second".to_string())]);
        assert_eq!(shared.latest.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn piped_requests_wait_for_each_reply() {
        let shared = shared();
        let (tx, rx) = mpsc::channel::<(u64, String)>();
        // A stand-in worker: marks each request finished 30 ms after it arrives.
        let s2 = shared.clone();
        let worker = std::thread::spawn(move || {
            let mut order = Vec::new();
            for (id, text) in rx {
                std::thread::sleep(Duration::from_millis(30));
                order.push(text);
                s2.generated.store(id, Ordering::SeqCst);
                s2.finished.store(id, Ordering::SeqCst);
            }
            order
        });
        let t0 = std::time::Instant::now();
        typed_requests(std::io::Cursor::new("a\nb\nc\n"), shared.clone(), tx, false);
        let order = worker.join().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
        assert!(t0.elapsed() >= Duration::from_millis(90), "each line waited for the previous reply");
        assert_eq!(shared.finished.load(Ordering::SeqCst), 3);
        // Note: in non-interactive mode the end of input requests a quit; that flag is process-wide,
        // so this test runs last-ish by name and other tests do not depend on it being unset.
    }

    #[test]
    fn echo_detection() {
        let spoken = "Tomorrow in Hong Kong will be fine, dry and hot, with temperatures from 26 to 32 degrees.";
        assert!(is_echo("Tomorrow in Hong Kong will be fine and dry.", spoken));
        assert!(is_echo("fine dry and hot", spoken));
        assert!(!is_echo("Stop, what about Tokyo instead?", spoken));
        assert!(!is_echo("AI, what is the news today?", spoken));
        // Whisper writes digits, the voice says number words: still the same sentence.
        assert!(is_echo("2 plus 2 equals 4.", "Two plus two equals four. Anything else?"));
    }
}
