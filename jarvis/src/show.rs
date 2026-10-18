//! Show a picture right in the terminal, with the Kitty graphics protocol (kitty, iTerm2 3.6+,
//! WezTerm, Ghostty, ...) or iTerm2's own inline-image protocol.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Protocol {
    Kitty,
    Iterm,
    None,
}

impl Protocol {
    /// `--image-protocol auto|kitty|iterm|none`. Auto: iTerm2 gets its own inline-image protocol
    /// (its Kitty support acknowledges but does not reliably draw), Terminal.app gets nothing,
    /// everything else (kitty, WezTerm, Ghostty, ...) gets the Kitty graphics protocol; nothing
    /// when output is not a terminal.
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "kitty" => Self::Kitty,
            "iterm" => Self::Iterm,
            "none" => Self::None,
            "auto" => {
                // SAFETY: isatty only inspects a file descriptor.
                if unsafe { libc::isatty(1) } == 0 {
                    Self::None
                } else {
                    match std::env::var("TERM_PROGRAM").as_deref() {
                        Ok("Apple_Terminal") => Self::None,
                        Ok("iTerm.app") => Self::Iterm,
                        _ => Self::Kitty,
                    }
                }
            }
            other => bail!("--image-protocol must be auto, kitty, iterm or none, not {other:?}"),
        })
    }
}

/// Remove terminal control replies (a Kitty `ESC _ G ... ESC \\` acknowledgement, OSC or CSI
/// sequences) that can land in typed input after a picture was drawn.
pub fn strip_terminal_replies(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match chars.next() {
            // APC (ESC _) and OSC (ESC ]) run until ESC \\ or BEL.
            Some('_') | Some(']') => {
                let mut prev = ' ';
                for d in chars.by_ref() {
                    if d == '\x07' || (prev == '\x1b' && d == '\\') {
                        break;
                    }
                    prev = d;
                }
            }
            // CSI (ESC [) runs until a letter.
            Some('[') => {
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// An animated progress line on stderr, redrawn ten times a second from another thread:
/// `🎨 ▓▓▓▓▓▓▓░░░░░░░ 2/3 steps · 6.4 s` once steps are known, a spinner before that.
/// Dropping it (or calling `finish`) clears the line.
pub struct Progress {
    state: std::sync::Arc<(std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64, std::sync::atomic::AtomicBool)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Progress {
    pub fn start(label: &'static str) -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let state = std::sync::Arc::new((AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false)));
        let s = state.clone();
        let t0 = std::time::Instant::now();
        let thread = std::thread::spawn(move || {
            const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let mut tick = 0usize;
            while !s.2.load(Ordering::Relaxed) {
                let (done, total) = (s.0.load(Ordering::Relaxed), s.1.load(Ordering::Relaxed));
                let line = render(label, done, total, t0.elapsed().as_secs_f64(), SPIN[tick % SPIN.len()]);
                eprint!("\r\x1b[K{line}");
                tick += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            eprint!("\r\x1b[K");
        });
        Self { state, thread: Some(thread) }
    }

    pub fn set(&self, done: u64, total: u64) {
        use std::sync::atomic::Ordering;
        self.state.0.store(done, Ordering::Relaxed);
        self.state.1.store(total, Ordering::Relaxed);
    }

    pub fn finish(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        self.state.2.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The text of the progress line.
pub fn render(label: &str, done: u64, total: u64, secs: f64, spin: char) -> String {
    if total == 0 {
        return format!("{label} {spin} {secs:.1} s");
    }
    let width = 20;
    let filled = ((done.min(total) * width) / total) as usize;
    let bar: String = std::iter::repeat_n('▓', filled).chain(std::iter::repeat_n('░', width as usize - filled)).collect();
    format!("{label} {bar} {done}/{total} steps · {secs:.1} s")
}

/// Size of one terminal cell in pixels (width, height), from the terminal if it reports it,
/// otherwise a typical 10 x 20.
fn cell_pixels() -> (u32, u32) {
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: ioctl TIOCGWINSZ fills the winsize struct for a terminal fd.
    let ok = unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } == 0;
    if ok && ws.ws_col > 0 && ws.ws_row > 0 && ws.ws_xpixel > 0 && ws.ws_ypixel > 0 {
        ((ws.ws_xpixel / ws.ws_col).max(1) as u32, (ws.ws_ypixel / ws.ws_row).max(1) as u32)
    } else {
        (10, 20)
    }
}

/// PNG bytes of an image file, converting other formats with `sips` (JPEG, HEIC, WebP, ...).
pub fn png_bytes(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.starts_with(b"\x89PNG") {
        return Ok(bytes);
    }
    let tmp = std::env::temp_dir().join(format!("jarvis-show-{}.png", std::process::id()));
    let status = Command::new("sips").args(["-s", "format", "png"]).arg(path).arg("--out").arg(&tmp).stdout(std::process::Stdio::null()).status()?;
    if !status.success() {
        bail!("sips could not convert {}", path.display());
    }
    let png = std::fs::read(&tmp)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(png)
}

/// The escape sequences that draw `png` inline in a box of `px` x `px` pixels (aspect ratio
/// kept). `cell` is the terminal's cell size in pixels, used to turn pixels into cells for Kitty.
pub fn encode(protocol: Protocol, png: &[u8], px: u32, cell: (u32, u32)) -> Vec<u8> {
    let data = B64.encode(png);
    let mut out = Vec::new();
    match protocol {
        Protocol::None => {}
        Protocol::Iterm => {
            out.extend_from_slice(
                format!("\x1b]1337;File=inline=1;width={px}px;height={px}px;preserveAspectRatio=1;size={}:", png.len()).as_bytes(),
            );
            out.extend_from_slice(data.as_bytes());
            out.extend_from_slice(b"\x07\n");
        }
        Protocol::Kitty => {
            let cols = px.div_ceil(cell.0.max(1)).max(1);
            let rows = px.div_ceil(cell.1.max(1)).max(1);
            // f=100: PNG; a=T: transmit and display; c/r: the box in cells; q=2: no replies;
            // m=1: more chunks follow.
            let chunks: Vec<&[u8]> = data.as_bytes().chunks(4096).collect();
            for (i, chunk) in chunks.iter().enumerate() {
                let more = if i + 1 < chunks.len() { 1 } else { 0 };
                if i == 0 {
                    out.extend_from_slice(format!("\x1b_Gf=100,a=T,i=1,c={cols},r={rows},q=2,m={more};").as_bytes());
                } else {
                    out.extend_from_slice(format!("\x1b_Gm={more};").as_bytes());
                }
                out.extend_from_slice(chunk);
                out.extend_from_slice(b"\x1b\\");
            }
            out.extend_from_slice(b"\n");
        }
    }
    out
}

/// Draw the image file in the terminal as a `px` x `px` picture (no-op for `Protocol::None`).
pub fn show(protocol: Protocol, path: &Path, px: u32) -> Result<()> {
    if protocol == Protocol::None {
        return Ok(());
    }
    let png = png_bytes(path)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&encode(protocol, &png, px.max(16), cell_pixels()))?;
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n_fake_png_payload_";

    #[test]
    fn kitty_sequence_is_chunked_and_terminated() {
        // 256 px in 8 x 16 px cells is 32 columns by 16 rows.
        let out = encode(Protocol::Kitty, PNG, 256, (8, 16));
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("\x1b_Gf=100,a=T,i=1,c=32,r=16,q=2,m=0;"), "{s:?}");
        assert!(s.ends_with("\x1b\\\n"));
        // A large payload is split into 4096-byte chunks, all but the last marked m=1.
        let big = vec![0u8; 10_000];
        let s = String::from_utf8(encode(Protocol::Kitty, &big, 256, (8, 16))).unwrap();
        assert_eq!(s.matches("\x1b_G").count(), (B64.encode(&big).len() + 4095) / 4096);
        assert_eq!(s.matches("m=1;").count(), s.matches("\x1b_G").count() - 1);
        assert!(s.contains("m=0;"));
    }

    #[test]
    fn iterm_sequence_carries_pixel_size_and_base64() {
        let s = String::from_utf8(encode(Protocol::Iterm, PNG, 256, (10, 20))).unwrap();
        assert!(s.starts_with(&format!("\x1b]1337;File=inline=1;width=256px;height=256px;preserveAspectRatio=1;size={}:", PNG.len())));
        assert!(s.contains(&B64.encode(PNG)));
        assert!(s.ends_with("\x07\n"));
        assert!(encode(Protocol::None, PNG, 256, (10, 20)).is_empty());
    }

    /// A valid 1x1 red PNG.
    const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

    #[test]
    fn non_png_files_are_converted_with_sips() {
        let dir = std::env::temp_dir().join(format!("jarvis-show-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("dot.png");
        std::fs::write(&png, B64.decode(TINY_PNG).unwrap()).unwrap();
        assert!(png_bytes(&png).unwrap().starts_with(b"\x89PNG"), "PNG passes through");
        let jpg = dir.join("dot.jpg");
        let ok = Command::new("sips").args(["-s", "format", "jpeg"]).arg(&png).arg("--out").arg(&jpg).stdout(std::process::Stdio::null()).status().unwrap().success();
        assert!(ok);
        assert!(!std::fs::read(&jpg).unwrap().starts_with(b"\x89PNG"));
        assert!(png_bytes(&jpg).unwrap().starts_with(b"\x89PNG"), "JPEG is converted to PNG");
        assert!(png_bytes(&dir.join("missing.png")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn terminal_replies_are_stripped_from_typed_lines() {
        assert_eq!(strip_terminal_replies("\x1b_Gi=1;OK\x1b\\imagine a cat"), "imagine a cat");
        assert_eq!(strip_terminal_replies("hello \x1b[31mred\x1b[0m"), "hello red");
        assert_eq!(strip_terminal_replies("plain"), "plain");
    }

    #[test]
    fn progress_line_renders_spinner_then_bar() {
        assert_eq!(render("🎨", 0, 0, 1.25, '⠋'), "🎨 ⠋ 1.2 s");
        let line = render("🎨", 2, 3, 6.4, '⠋');
        assert!(line.starts_with("🎨 ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ 2/3 steps · 6.4 s"), "{line}");
        assert!(render("🎨", 3, 3, 9.0, '⠋').contains("▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 3/3 steps"));
        let p = Progress::start("🎨");
        p.set(1, 4);
        std::thread::sleep(std::time::Duration::from_millis(150));
        p.finish(); // clears the line and joins the thread
    }

    #[test]
    fn protocol_names() {
        assert_eq!(Protocol::parse("kitty").unwrap(), Protocol::Kitty);
        assert_eq!(Protocol::parse("iterm").unwrap(), Protocol::Iterm);
        assert_eq!(Protocol::parse("none").unwrap(), Protocol::None);
        assert!(Protocol::parse("sixel").is_err());
    }
}
