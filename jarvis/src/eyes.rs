//! The eyes: a screenshot answered by Qwen3-VL on MLX (python/see.py), with Apple's Vision
//! framework (OCR) as the fallback when the vision model is not available.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};
use serde_json::json;

use crate::helper::Helper;

pub struct Eyes {
    vlm: Helper,
}

impl Eyes {
    pub fn start(python: &Path, script: &Path, model: &str, persona: &str, verbose: bool) -> Result<Self> {
        let system = format!(
            "{persona} You are shown a screenshot of the user's Mac screen; describe what is on it and answer \
             the question from what you see."
        );
        let args = vec!["--model".to_string(), model.to_string(), "--system".to_string(), system];
        Ok(Self { vlm: Helper::start(python, script, &args, verbose)? })
    }

    pub fn wait_ready(&mut self) -> Result<f64> {
        self.vlm.wait_ready()
    }

    pub fn info(&self) -> &str {
        &self.vlm.info
    }

    pub fn info_short(&self) -> &str {
        &self.vlm.info_short
    }

    /// Answer `question` about the image.
    pub fn look(&mut self, image: &Path, question: &str) -> Result<String> {
        let prompt = format!("This is a screenshot of the user's Mac screen. {question}");
        let v = self.vlm.call(&json!({ "image": image, "prompt": prompt, "max_tokens": 200 }))?;
        eprintln!("[ai local: Qwen3-VL on MLX (eyes) · {} · {:.1} s]", self.vlm.info_short, v["seconds"].as_f64().unwrap_or(0.0));
        Ok(v["text"].as_str().unwrap_or("").trim().to_string())
    }
}

/// Capture the whole screen to a PNG (needs Screen Recording permission for the terminal;
/// without it the picture shows only the wallpaper).
pub fn screenshot() -> Result<PathBuf> {
    let png = std::env::temp_dir().join(format!("jarvis-screen-{}.png", std::process::id()));
    let status = Command::new("screencapture").args(["-x", "-t", "png"]).arg(&png).status()?;
    if !status.success() {
        bail!("screencapture failed: {status}");
    }
    Ok(png)
}

/// Is the user asking about what is on the screen?
pub fn wants_screen(text: &str) -> bool {
    let t = text.to_lowercase();
    ["screen", "display", "monitor", "what do you see", "look at this", "read this", "what am i looking at"]
        .iter()
        .any(|k| t.contains(k))
}

#[cfg(test)]
mod tests {
    use super::wants_screen;

    #[test]
    fn screen_questions() {
        for t in ["What is on my screen right now?", "Read this for me", "what do you see?", "Look at this chart"] {
            assert!(wants_screen(t), "{t:?}");
        }
        for t in ["What is two plus two?", "Tell me a joke", "AI, what's the weather?"] {
            assert!(!wants_screen(t), "{t:?}");
        }
    }
}
