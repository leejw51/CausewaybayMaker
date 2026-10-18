//! Pictures: making them ("imagine a red apple") and looking at them ("describe ~/pic.jpg",
//! "what is in https://.../photo.png"). Generation uses Apple's Image Playground when the Mac
//! supports it, otherwise FLUX.1-schnell on MLX (python/imagine.py). Looking uses Apple's Vision
//! framework (text, labels, faces, animals, barcodes), narrated by the brain.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::apple::AppleChat;
use crate::helper::Helper;

/// "imagine a red apple" -> Some(("a red apple", style)). Also draw / paint / sketch /
/// generate (an image|a picture) of / make (an image|a picture) of / create an image of.
pub fn imagine_request(text: &str) -> Option<(String, &'static str)> {
    let t = text.trim().trim_end_matches(['.', '!', '?']).to_string();
    let lower = t.to_lowercase();
    let style = if lower.contains("sketch") {
        "sketch"
    } else if lower.contains("cartoon") || lower.contains("animation") || lower.contains("animated") {
        "animation"
    } else {
        "illustration"
    };
    let starters = [
        "imagine", "draw", "paint", "sketch", "generate", "create", "make", "show me", "give me",
    ];
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut i = 0;
    // Optional "jarvis," / "please" prefix.
    while i < words.len() && matches!(words[i].trim_matches(','), "jarvis" | "please" | "hey" | "can" | "you" | "could") {
        i += 1;
    }
    let rest = words.get(i..)?;
    let start = starters.iter().find(|s| {
        let sw: Vec<&str> = s.split_whitespace().collect();
        rest.len() >= sw.len() && rest[..sw.len()] == sw[..]
    })?;
    let mut rest = &rest[start.split_whitespace().count()..];
    let media = |w: &str| matches!(w, "image" | "picture" | "drawing" | "painting" | "sketch" | "illustration" | "cartoon" | "photo" | "pic");
    let is_media_request = *start == "imagine" || *start == "draw" || *start == "paint" || *start == "sketch"
        || rest.iter().take(4).any(|w| media(w));
    if !is_media_request {
        return None;
    }
    // Drop "me", "please" and a media phrase such as "a picture of" / "an image showing", but keep
    // the subject's own words ("a red apple on a table").
    while rest.first().is_some_and(|w| matches!(*w, "me" | "please" | "us")) {
        rest = &rest[1..];
    }
    let mut j = 0;
    if rest.get(j).is_some_and(|w| matches!(*w, "a" | "an" | "the" | "new" | "nice" | "quick")) {
        j += 1;
        if rest.get(j).is_some_and(|w| matches!(*w, "new" | "nice" | "quick" | "little")) {
            j += 1;
        }
    }
    if rest.get(j).is_some_and(|w| media(w)) {
        j += 1;
        if rest.get(j).is_some_and(|w| matches!(*w, "of" | "showing" | "with" | "depicting")) {
            j += 1;
        }
        rest = &rest[j..];
    }
    if rest.is_empty() {
        return None;
    }
    // Keep the user's own casing for the prompt.
    let skip = words.len() - rest.len();
    let prompt: Vec<&str> = t.split_whitespace().skip(skip).collect();
    Some((prompt.join(" "), style))
}

/// A file path or URL in the text that looks like an image -> Some((source, question)).
/// "describe ~/Desktop/cat.jpg" -> ("~/Desktop/cat.jpg", "describe this image").
pub fn image_reference(text: &str) -> Option<(String, String)> {
    const EXT: [&str; 9] = ["png", "jpg", "jpeg", "heic", "webp", "gif", "bmp", "tiff", "tif"];
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c: char| matches!(c, ',' | ';' | '"' | '\'' | '(' | ')' | '<' | '>'));
        let is_url = token.starts_with("http://") || token.starts_with("https://");
        let is_path = token.starts_with('/') || token.starts_with("~/") || token.starts_with("./");
        let lower = token.to_lowercase();
        let ends_like_image = EXT.iter().any(|e| lower.trim_end_matches(['.', '?', '!']).ends_with(&format!(".{e}")));
        if is_url || (is_path && (ends_like_image || Path::new(&expand(token)).is_file())) || (ends_like_image && Path::new(&expand(token)).is_file()) {
            let source = token.trim_end_matches(['.', '?', '!']).to_string();
            let question = text.replace(raw, "").split_whitespace().collect::<Vec<_>>().join(" ");
            let question = question.trim().trim_matches(',').trim().to_string();
            let question = if question.chars().filter(|c| c.is_alphabetic()).count() < 3 { "Describe this image.".to_string() } else { question };
            return Some((source, question));
        }
    }
    None
}

/// `~/x` -> `$HOME/x`.
pub fn expand(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path.to_string(),
    }
}

/// Bring the image local: a URL is downloaded with curl to a temp file.
pub fn fetch(source: &str) -> Result<(PathBuf, bool)> {
    if source.starts_with("http://") || source.starts_with("https://") {
        let ext = source.rsplit('.').next().filter(|e| e.len() <= 4 && e.chars().all(char::is_alphanumeric)).unwrap_or("img");
        let tmp = std::env::temp_dir().join(format!("jarvis-fetch-{}.{ext}", std::process::id()));
        // -f: an HTTP error (404, 500) is a failure, not a saved error page.
        let status = Command::new("curl").args(["-sSLf", "--max-time", "60", "-o"]).arg(&tmp).arg(source).status()?;
        if !status.success() || std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0) == 0 {
            bail!("could not download {source}");
        }
        return Ok((tmp, true));
    }
    let p = PathBuf::from(expand(source));
    if !p.is_file() {
        bail!("no such file: {}", p.display());
    }
    Ok((p, false))
}

/// What Apple Vision found in the image, as a description the brain can narrate.
pub fn describe_with_vision(apple: &AppleChat, image: &Path) -> Result<String> {
    eprintln!("[ai local: Apple Vision (picture) · Vision framework · macOS {}]", crate::voice::macos_version());
    let v = apple.see(image)?;
    let names = |key: &str| -> Vec<String> {
        v[key].as_array().map(|a| a.iter().filter_map(|x| x.as_str().or_else(|| x["name"].as_str()).map(str::to_string)).collect()).unwrap_or_default()
    };
    let mut d = format!("Image {}x{} pixels.", v["width"], v["height"]);
    let labels = names("labels");
    if !labels.is_empty() {
        d.push_str(&format!(" Looks like: {}.", labels.join(", ")));
    }
    let animals = names("animals");
    if !animals.is_empty() {
        d.push_str(&format!(" Animals: {}.", animals.join(", ")));
    }
    match v["faces"].as_u64().unwrap_or(0) {
        0 => {}
        1 => d.push_str(" One person's face is visible."),
        n => d.push_str(&format!(" {n} faces are visible.")),
    }
    let codes = names("barcodes");
    if !codes.is_empty() {
        d.push_str(&format!(" Barcode/QR content: {}.", codes.join(" | ")));
    }
    let text = names("text");
    if !text.is_empty() {
        let joined = text.join(" / ");
        let cut: String = joined.chars().take(800).collect();
        d.push_str(&format!(" Text in the image: {cut}"));
    }
    Ok(d)
}

/// Where generated pictures are kept (JARVIS_PICTURES or ~/Pictures/Jarvis).
pub fn pictures_dir() -> PathBuf {
    let dir = std::env::var_os("JARVIS_PICTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Pictures/Jarvis"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// A fresh output path for a generated picture, named after the prompt.
pub fn picture_path(prompt: &str) -> PathBuf {
    let slug: String = prompt
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    pictures_dir().join(format!("{stamp}-{slug}.png"))
}

/// Generate a picture: Image Playground first, FLUX on MLX as the fallback. Returns the file.
pub fn generate(apple: Option<&AppleChat>, flux: Option<&mut Helper>, prompt: &str, style: &str, out: &Path) -> Result<&'static str> {
    let bar = crate::show::Progress::start("🎨");
    if let Some(apple) = apple {
        match apple.imagine(prompt, out, style) {
            Ok(()) => {
                bar.finish();
                return Ok("Image Playground");
            }
            Err(e) => eprintln!("\r\x1b[K[imagine: {e}]"),
        }
    }
    let Some(flux) = flux else {
        bar.finish();
        bail!("no image model: Image Playground is unavailable and FLUX is off")
    };
    let steps = 3u64;
    let v = flux.call_with_progress(&json!({ "prompt": prompt, "out": out, "steps": steps, "width": 768, "height": 768 }), |p| {
        bar.set(p["progress"].as_u64().unwrap_or(0), p["total"].as_u64().unwrap_or(steps));
    })?;
    bar.set(steps, steps);
    std::thread::sleep(std::time::Duration::from_millis(120)); // let the full bar show once
    bar.finish();
    let info = if flux.info_short.is_empty() { String::from("FLUX.1-schnell") } else { flux.info_short.clone() };
    eprintln!("[ai local: FLUX.1-schnell on MLX (imagine) · {info} · {:.1} s]", v["seconds"].as_f64().unwrap_or(0.0));
    Ok("FLUX.1-schnell on MLX")
}

pub fn value_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imagine_requests() {
        assert_eq!(imagine_request("Imagine a red apple on a table."), Some(("a red apple on a table".into(), "illustration")));
        assert_eq!(imagine_request("draw me a picture of a cat in space"), Some(("a cat in space".into(), "illustration")));
        assert_eq!(imagine_request("Jarvis, generate an image of a lighthouse at night"), Some(("a lighthouse at night".into(), "illustration")));
        assert_eq!(imagine_request("sketch a dog"), Some(("a dog".into(), "sketch")));
        assert_eq!(imagine_request("show me a picture of the Eiffel Tower"), Some(("the Eiffel Tower".into(), "illustration")));
        assert_eq!(imagine_request("make a cartoon of a robot chef").map(|r| r.1), Some("animation"));
        assert_eq!(imagine_request("make a reservation for two"), None);
        assert_eq!(imagine_request("What is two plus two?"), None);
        assert_eq!(imagine_request("create a new file"), None);
    }

    #[test]
    fn image_references() {
        let (src, q) = image_reference("describe https://example.com/pics/cat.jpg please").unwrap();
        assert_eq!(src, "https://example.com/pics/cat.jpg");
        assert_eq!(q, "describe please");
        let (src, q) = image_reference("~/Desktop/photo.png").unwrap();
        assert_eq!(src, "~/Desktop/photo.png");
        assert_eq!(q, "Describe this image.");
        let (src, _) = image_reference("what text is in /tmp/scan.heic?").unwrap();
        assert_eq!(src, "/tmp/scan.heic");
        assert!(image_reference("what is on my screen").is_none());
        assert!(image_reference("read http://example.com/page").is_some(), "any URL is worth a look");
    }

    #[test]
    fn fetch_local_and_http() {
        let dir = std::env::temp_dir().join(format!("jarvis-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pic.png"), b"\x89PNG\r\n\x1a\nfake").unwrap();
        // Local file: used in place, not copied.
        let (p, downloaded) = fetch(dir.join("pic.png").to_str().unwrap()).unwrap();
        assert_eq!(p, dir.join("pic.png"));
        assert!(!downloaded);
        assert!(fetch("/nonexistent/x.png").is_err());
        // URL: served by a throwaway local HTTP server, downloaded to a temp file.
        let mut server = Command::new("python3")
            .args(["-u", "-m", "http.server", "0", "--bind", "127.0.0.1", "--directory"])
            .arg(&dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let mut port = None;
        {
            use std::io::BufRead;
            let out = std::io::BufReader::new(server.stdout.take().unwrap());
            for line in out.lines().map_while(Result::ok) {
                // "Serving HTTP on 127.0.0.1 port 54321 ..."
                if let Some(rest) = line.split(" port ").nth(1) {
                    port = rest.split_whitespace().next().and_then(|p| p.parse::<u16>().ok());
                    break;
                }
            }
        }
        let port = port.expect("http.server did not report its port");
        let (p, downloaded) = fetch(&format!("http://127.0.0.1:{port}/pic.png")).unwrap();
        assert!(downloaded);
        assert_eq!(std::fs::read(&p).unwrap(), b"\x89PNG\r\n\x1a\nfake");
        assert!(fetch(&format!("http://127.0.0.1:{port}/missing.png")).is_err());
        let _ = server.kill();
        let _ = server.wait();
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn picture_names() {
        let p = picture_path("A red apple, on a table!");
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with("-a-red-apple-on-a-table.png"), "{name}");
        assert_eq!(expand("~/x"), format!("{}/x", std::env::var("HOME").unwrap()));
    }
}
