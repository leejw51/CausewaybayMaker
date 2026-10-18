//! The voice: Kokoro on MLX (python/speak.py) for the languages it knows, macOS `say` otherwise
//! or when Kokoro is unavailable. Sentences are queued and played one after another in the
//! background, so Jarvis starts talking while the rest of its reply is still being written, and
//! can be cut off the moment the user speaks or quits.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::json;

use crate::helper::Helper;

/// One sentence Jarvis has said or is saying: text, when playback started, when it ended.
pub struct Spoken {
    pub text: String,
    pub start: Instant,
    pub end: Option<Instant>,
}

/// Log of recent speech, shared with the listening thread so it can recognise its own voice.
pub type SpokenLog = Arc<Mutex<Vec<Spoken>>>;

/// Everything said now or within the last `window`: what the microphone may be hearing.
pub fn recently_spoken(log: &SpokenLog, window: Duration) -> Option<String> {
    let mut log = log.lock().unwrap();
    log.retain(|s| s.end.is_none_or(|e| e.elapsed() < window));
    if log.is_empty() {
        return None;
    }
    Some(log.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" "))
}

enum Job {
    /// A synthesized WAV to play with afplay.
    Wav(PathBuf),
    /// Text for `say` to speak.
    Say(String),
}

pub struct Voice {
    kokoro: Option<Helper>,
    /// Kokoro language code for the spoken language, if Kokoro covers it.
    kokoro_lang: Option<&'static str>,
    kokoro_voice: String,
    /// `say -v VOICE` for the spoken language.
    say_voice: Option<String>,
    /// Playback volume for Kokoro speech, 0 to 1 (afplay -v).
    pub volume: f32,
    /// JARVIS_SPEAK_DIR: write replies as WAV files here instead of playing them (for tests).
    capture_dir: Option<PathBuf>,
    spoken: usize,
    queue: VecDeque<(String, Job)>,
    /// What is playing now: the player process, its WAV if any, and the log entry index.
    playing: Option<(Child, Option<PathBuf>, usize)>,
    log: SpokenLog,
    /// Synthesis done since the last `take_note`: (engine description, sentences, seconds).
    note: Option<(String, usize, f64)>,
    /// Startup `name()` with path and size (--verbose).
    pub verbose: bool,
}

impl Voice {
    /// `language` is the Whisper language code (en, ko, ja, ...).
    pub fn new(kokoro: Option<Helper>, language: &str, voice: Option<String>, log: SpokenLog) -> Self {
        let kokoro_lang = match language {
            "en" => Some("a"),
            "ja" => Some("j"),
            "zh" => Some("z"),
            "es" => Some("e"),
            "fr" => Some("f"),
            "it" => Some("i"),
            "pt" => Some("p"),
            "hi" => Some("h"),
            _ => None,
        };
        let say_voice = match language {
            "ko" => Some("Yuna"),
            "ja" => Some("Kyoko"),
            "zh" => Some("Tingting"),
            _ => None,
        }
        .map(String::from);
        let uses_kokoro = kokoro.is_some() && kokoro_lang.is_some();
        Self {
            kokoro,
            kokoro_lang,
            kokoro_voice: voice.clone().filter(|_| uses_kokoro).unwrap_or_else(|| "af_heart".into()),
            say_voice: voice.filter(|_| !uses_kokoro).or(say_voice),
            volume: 1.0,
            capture_dir: std::env::var_os("JARVIS_SPEAK_DIR").map(PathBuf::from).inspect(|d| {
                let _ = std::fs::create_dir_all(d);
            }),
            spoken: 0,
            queue: VecDeque::new(),
            playing: None,
            log,
            note: None,
            verbose: false,
        }
    }

    pub fn start_kokoro(python: &Path, script: &Path, model: &str, verbose: bool) -> Result<Helper> {
        Helper::start(python, script, &["--model".to_string(), model.to_string()], verbose)
    }

    pub fn wait_ready(&mut self) -> Result<f64> {
        match &mut self.kokoro {
            Some(k) => k.wait_ready(),
            None => Ok(0.0),
        }
    }

    pub fn name(&self) -> String {
        match (&self.kokoro, self.kokoro_lang) {
            (Some(k), Some(_)) => format!("Kokoro on MLX · voice {} · {}", self.kokoro_voice, if self.verbose { &k.info } else { &k.info_short }),
            _ => format!("macOS say · voice {} · macOS {}", self.say_voice.as_deref().unwrap_or("default"), macos_version()),
        }
    }

    /// Queue one sentence: synthesize it now (Kokoro) and start playing as soon as the previous
    /// sentence is over. Returns at once.
    pub fn say(&mut self, text: &str) {
        let text = speakable(text);
        let text = text.trim();
        if text.chars().any(char::is_alphanumeric) == false {
            return;
        }
        self.spoken += 1;
        let capture = self.capture_dir.as_ref().map(|d| d.join(format!("reply-{}.wav", self.spoken)));
        let job = if let (Some(kokoro), Some(lang)) = (&mut self.kokoro, self.kokoro_lang) {
            let wav = capture
                .clone()
                .unwrap_or_else(|| std::env::temp_dir().join(format!("jarvis-say-{}-{}.wav", std::process::id(), self.spoken)));
            let t0 = Instant::now();
            match kokoro.call(&json!({ "text": text, "voice": self.kokoro_voice, "lang": lang, "wav": wav })) {
                Ok(_) => {
                    let secs = t0.elapsed().as_secs_f64();
                    let engine = format!("Kokoro on MLX (voice) · {}", kokoro.info_short);
                    match &mut self.note {
                        Some((_, n, total)) => {
                            *n += 1;
                            *total += secs;
                        }
                        None => self.note = Some((engine, 1, secs)),
                    }
                    if capture.is_some() {
                        self.log_only(text);
                        return;
                    }
                    Job::Wav(wav)
                }
                Err(e) => {
                    eprintln!("[voice: {e}; using say]");
                    Job::Say(text.to_string())
                }
            }
        } else {
            let engine = format!("macOS say (voice) · voice {} · macOS {}", self.say_voice.as_deref().unwrap_or("default"), macos_version());
            match &mut self.note {
                Some((_, n, _)) => *n += 1,
                None => self.note = Some((engine, 1, 0.0)),
            }
            Job::Say(text.to_string())
        };
        if let (Job::Say(t), Some(path)) = (&job, &capture) {
            // Test mode: synthesize to a file and move on.
            let mut cmd = Command::new("say");
            if let Some(v) = &self.say_voice {
                cmd.arg("-v").arg(v);
            }
            let _ = cmd.arg("-o").arg(path).arg("--data-format=LEI16@22050").arg(t).status();
            self.log_only(text);
            return;
        }
        self.queue.push_back((text.to_string(), job));
        self.pump();
    }

    fn log_only(&self, text: &str) {
        let now = Instant::now();
        self.log.lock().unwrap().push(Spoken { text: text.to_string(), start: now, end: Some(now) });
    }

    /// The trace line for the speech synthesized since the last call, if any.
    pub fn take_note(&mut self) -> Option<String> {
        let (engine, n, secs) = self.note.take()?;
        let plural = if n == 1 { "sentence" } else { "sentences" };
        Some(if secs > 0.0 {
            format!("[ai local: {engine} · {n} {plural} · {secs:.1} s]")
        } else {
            format!("[ai local: {engine} · {n} {plural}]")
        })
    }

    /// Advance playback: notice a finished sentence, start the next queued one. Call often.
    pub fn pump(&mut self) {
        if let Some((child, wav, idx)) = &mut self.playing {
            match child.try_wait() {
                Ok(None) => return, // still playing
                _ => {
                    if let Some(w) = wav.take() {
                        let _ = std::fs::remove_file(w);
                    }
                    if let Some(s) = self.log.lock().unwrap().get_mut(*idx) {
                        s.end = Some(Instant::now());
                    }
                    self.playing = None;
                }
            }
        }
        let Some((text, job)) = self.queue.pop_front() else { return };
        let (child, wav) = match job {
            Job::Wav(wav) => {
                let child = Command::new("afplay")
                    .arg("-v")
                    .arg(format!("{}", self.volume))
                    .arg(&wav)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                (child, Some(wav))
            }
            Job::Say(t) => {
                let mut cmd = Command::new("say");
                if let Some(v) = &self.say_voice {
                    cmd.arg("-v").arg(v);
                }
                (cmd.arg(t).spawn(), None)
            }
        };
        match child {
            Ok(child) => {
                let mut log = self.log.lock().unwrap();
                log.push(Spoken { text, start: Instant::now(), end: None });
                let idx = log.len() - 1;
                self.playing = Some((child, wav, idx));
            }
            Err(e) => {
                eprintln!("[voice: cannot play: {e}]");
                if let Some(w) = wav {
                    let _ = std::fs::remove_file(w);
                }
            }
        }
    }

    /// Is anything playing or waiting to be played?
    pub fn busy(&mut self) -> bool {
        self.pump();
        self.playing.is_some() || !self.queue.is_empty()
    }

    /// Cut the current speech off and forget what was queued.
    pub fn stop(&mut self) {
        if let Some((mut child, wav, idx)) = self.playing.take() {
            let _ = child.kill();
            let _ = child.wait();
            if let Some(w) = wav {
                let _ = std::fs::remove_file(w);
            }
            if let Some(s) = self.log.lock().unwrap().get_mut(idx) {
                s.end = Some(Instant::now());
            }
        }
        for (_, job) in self.queue.drain(..) {
            if let Job::Wav(w) = job {
                let _ = std::fs::remove_file(w);
            }
        }
    }
}

impl Drop for Voice {
    fn drop(&mut self) {
        self.stop();
    }
}

/// "27.0.1"
pub fn macos_version() -> String {
    std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "?".into())
}

/// Make text sound natural when read aloud: no markdown marks, URLs, bracketed notes, emoji or
/// list bullets; "e.g." style abbreviations are left to the synthesizer.
pub fn speakable(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skip_until: Option<char> = None;
    for word in text.split_whitespace() {
        // Drop URLs and file paths wholesale.
        if word.starts_with("http://") || word.starts_with("https://") || word.starts_with("www.") {
            continue;
        }
        let mut w = String::new();
        for c in word.chars() {
            if let Some(close) = skip_until {
                if c == close {
                    skip_until = None;
                }
                continue;
            }
            match c {
                '[' => skip_until = Some(']'),
                '(' if word.starts_with('(') => skip_until = Some(')'),
                '*' | '_' | '#' | '`' | '~' | '>' | '|' => {}
                c if is_emoji(c) => {}
                c => w.push(c),
            }
        }
        // Nothing but punctuation left (a bullet, a bracket's trailing dot): drop it.
        if !w.chars().any(char::is_alphanumeric) {
            continue;
        }
        let w = w.as_str();
        if !w.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(w);
        }
    }
    out
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32, 0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F000..=0x1F2FF | 0xFE0F | 0x200D)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> SpokenLog {
        Arc::default()
    }

    #[test]
    fn voice_choice_by_language() {
        // Without Kokoro everything goes to `say`, with a language-specific voice where one exists.
        assert!(Voice::new(None, "en", None, log()).name().starts_with("macOS say · voice default"));
        assert!(Voice::new(None, "ko", None, log()).name().starts_with("macOS say · voice Yuna"));
        assert!(Voice::new(None, "ja", None, log()).name().starts_with("macOS say · voice Kyoko"));
        assert!(Voice::new(None, "ko", Some("Jian".into()), log()).name().starts_with("macOS say · voice Jian"));
    }

    #[test]
    fn nothing_spoken_yet() {
        let l = log();
        let mut v = Voice::new(None, "en", None, l.clone());
        assert!(!v.busy());
        assert!(recently_spoken(&l, Duration::from_secs(5)).is_none());
        v.say("   "); // empty text is ignored
        assert!(!v.busy());
    }

    #[test]
    fn speakable_strips_what_should_not_be_read_aloud() {
        assert_eq!(speakable("**Two** plus _two_ is `4`."), "Two plus two is 4.");
        assert_eq!(speakable("See https://example.com/x for details [1] (source: web)."), "See for details");
        assert_eq!(speakable("- first\n- second 🎉"), "first second");
        assert_eq!(speakable("It's 3.5 degrees, e.g. mild."), "It's 3.5 degrees, e.g. mild.");
        assert_eq!(speakable("안녕하세요! 햄릿은 셰익스피어가 썼어요."), "안녕하세요! 햄릿은 셰익스피어가 썼어요.");
    }

    #[test]
    fn voice_note_summarises_sentences() {
        let l = log();
        let mut v = Voice::new(None, "en", None, l.clone());
        assert!(v.take_note().is_none());
        // Capture mode: `say` writes files instead of playing, so this is silent.
        let dir = std::env::temp_dir().join(format!("jarvis-voice-note-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        v.capture_dir = Some(dir.clone());
        v.say("One.");
        v.say("Two.");
        let note = v.take_note().unwrap();
        assert!(note.starts_with("[ai local: macOS say (voice) · voice default · macOS "), "{note}");
        assert!(note.ends_with(" · 2 sentences]"), "{note}");
        assert!(v.take_note().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn spoken_log_window() {
        let l = log();
        let now = Instant::now();
        l.lock().unwrap().push(Spoken { text: "old".into(), start: now, end: Some(now - Duration::from_secs(10)) });
        l.lock().unwrap().push(Spoken { text: "fresh".into(), start: now, end: Some(now) });
        l.lock().unwrap().push(Spoken { text: "playing".into(), start: now, end: None });
        assert_eq!(recently_spoken(&l, Duration::from_secs(3)).as_deref(), Some("fresh playing"));
        assert_eq!(l.lock().unwrap().len(), 2, "stale entries are dropped");
    }
}
