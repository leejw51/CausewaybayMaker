//! Apple's on-device foundation model and Vision framework, through the Swift helper
//! `swift/apple_ai.swift` (their APIs are Swift-only). Used as the router that decides when a
//! question needs the cloud, optionally as the brain, and as the fallback screen reader.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

pub struct AppleChat {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
    helper: PathBuf,
    /// "Apple Foundation Models · macOS 27.0.1 · system model, managed by macOS"
    pub info: String,
    /// "Apple Foundation Models · macOS 27.0.1"
    pub info_short: String,
}

impl AppleChat {
    pub fn start(helper: &Path, instructions: &str) -> Result<Self> {
        let mut child = Command::new(helper)
            .arg("chat")
            .arg(instructions)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            // Its own process group: terminal Ctrl-C / Ctrl-Z go to us, and we close it by EOF.
            .process_group(0)
            .spawn()
            .with_context(|| format!("start {} (run `make jarvis` to build it)", helper.display()))?;
        let stdin = child.stdin.take().unwrap();
        let out = BufReader::new(child.stdout.take().unwrap());
        let macos = crate::voice::macos_version();
        let info = format!("Apple Foundation Models · macOS {macos} · system model, managed by macOS");
        let info_short = format!("Apple Foundation Models · macOS {macos}");
        Ok(Self { child, stdin, out, helper: helper.to_path_buf(), info, info_short })
    }

    fn read(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.out.read_line(&mut line)? == 0 {
            bail!("apple-ai exited");
        }
        serde_json::from_str(&line).with_context(|| format!("apple-ai said {line:?}"))
    }

    /// Ask the model; `on_delta` gets the reply as it streams. Errors carry the helper's code.
    pub fn ask(&mut self, prompt: &str, mut on_delta: impl FnMut(&str)) -> Result<String> {
        writeln!(self.stdin, "{}", json!({ "prompt": prompt }))?;
        self.stdin.flush()?;
        let mut text = String::new();
        loop {
            let v = self.read()?;
            if let Some(d) = v["delta"].as_str() {
                text.push_str(d);
                on_delta(d);
            } else if v["done"].as_bool() == Some(true) {
                return Ok(text);
            } else if let Some(e) = v["error"].as_str() {
                bail!("{} ({})", e, v["code"].as_str().unwrap_or("other"));
            }
        }
    }

    /// Does this request need live or recent information? A guided-generation call, so the
    /// answer is a real boolean and does not depend on the conversation so far.
    pub fn needs_cloud(&mut self, question: &str) -> Result<bool> {
        writeln!(self.stdin, "{}", json!({ "route": question }))?;
        self.stdin.flush()?;
        let v = self.read()?;
        v["needs_cloud"].as_bool().ok_or_else(|| anyhow!("apple-ai said {v}"))
    }

    /// What the Vision framework finds in an image: text lines, scene labels, faces, animals,
    /// barcodes (the raw JSON from `apple-ai see`).
    pub fn see(&self, image: &Path) -> Result<Value> {
        let out = Command::new(&self.helper).arg("see").arg(image).output()?;
        if !out.status.success() {
            bail!("apple-ai see failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(serde_json::from_slice(&out.stdout)?)
    }

    /// Generate a picture with Image Playground (fails with `notSupported` where unavailable).
    pub fn imagine(&self, prompt: &str, out: &Path, style: &str) -> Result<()> {
        let o = Command::new(&self.helper).arg("imagine").arg(prompt).arg(out).arg(style).output()?;
        let v: Value = serde_json::from_slice(&o.stdout).with_context(|| format!("apple-ai imagine said {:?}", String::from_utf8_lossy(&o.stdout)))?;
        if let Some(e) = v["error"].as_str() {
            bail!("{e}");
        }
        eprintln!("[ai local: Image Playground (imagine) · macOS {} · {:.1} s]", crate::voice::macos_version(), v["seconds"].as_f64().unwrap_or(0.0));
        Ok(())
    }

    /// Read the text on an image with the Vision framework (OCR plus scene labels), as a
    /// description the brain can use.
    pub fn describe_image(&self, png: &Path, max_chars: usize) -> Result<String> {
        let v = self.see(png)?;
        let lines: Vec<&str> = v["text"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let labels: Vec<&str> = v["labels"]
            .as_array()
            .map(|a| a.iter().filter_map(|l| l["name"].as_str()).collect())
            .unwrap_or_default();
        let mut desc = format!("Scene labels: {}.\n", labels.join(", "));
        if lines.is_empty() {
            desc.push_str("No readable text.");
        } else {
            desc.push_str("Text on screen, top to bottom:\n");
            for l in lines {
                if desc.len() + l.len() > max_chars {
                    desc.push_str("...");
                    break;
                }
                desc.push_str(l);
                desc.push('\n');
            }
        }
        Ok(desc)
    }
}

/// Availability of Apple's on-device model and Image Playground, from `apple-ai status`.
pub fn status(helper: &Path) -> Result<Value> {
    let out = Command::new(helper).arg("status").output()?;
    serde_json::from_slice(&out.stdout).with_context(|| format!("apple-ai status said {:?}", String::from_utf8_lossy(&out.stdout)))
}

impl Drop for AppleChat {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
