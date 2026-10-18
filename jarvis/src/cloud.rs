//! The cloud: Grok through xAI's Responses API, with its web search tool. Only called when the
//! user says "AI, ..." or the on-device router decides the question needs live information.

use std::io::Write;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use qwen35::audio::listen::quit_requested;
use serde_json::{Value, json};

use crate::brain::PERSONA;

pub struct Grok {
    key: String,
    pub model: String,
    /// Last response id; xAI stores the conversation server-side, so this is our memory.
    previous: Option<String>,
    calls: usize,
}

impl Grok {
    pub fn from_env(model: Option<String>) -> Option<Self> {
        let key = std::env::var("XAI_API_KEY").ok().filter(|k| !k.trim().is_empty())?;
        let model = model.or_else(|| std::env::var("XAI_MODEL").ok()).unwrap_or_else(|| "grok-4.7".into());
        Some(Self { key, model, previous: None, calls: 0 })
    }

    /// Ask Grok; `cancelled` is polled while waiting (user quit or interrupted).
    pub fn ask(&mut self, prompt: &str, cancelled: &dyn Fn() -> bool) -> Result<String> {
        let mut body = json!({
            "model": self.model,
            "input": prompt,
            "tools": [{ "type": "web_search" }],
        });
        // xAI keeps the conversation (instructions included) under the previous response id;
        // it rejects `instructions` together with `previous_response_id`.
        match &self.previous {
            Some(id) => body["previous_response_id"] = json!(id),
            None => {
                body["instructions"] = json!(format!("{PERSONA} Use web search when the answer depends on current information."))
            }
        }
        let tag = format!("jarvis-{}-{}", std::process::id(), self.calls);
        self.calls += 1;
        let body_file = std::env::temp_dir().join(format!("{tag}-request.json"));
        let reply_file = std::env::temp_dir().join(format!("{tag}-reply.json"));
        std::fs::write(&body_file, body.to_string())?;
        // The key goes in through a config file on stdin so it never shows up in `ps`; the reply
        // goes to a file, so there is no pipe to drain while we wait.
        let mut curl = Command::new("curl")
            .args(["-sS", "--max-time", "180", "-K", "-", "-H", "Content-Type: application/json"])
            .arg("--data-binary")
            .arg(format!("@{}", body_file.display()))
            .arg("-o")
            .arg(&reply_file)
            .arg("https://api.x.ai/v1/responses")
            .stdin(Stdio::piped())
            // Own process group: Ctrl-Z must not stop curl under us; we kill it on quit instead.
            .process_group(0)
            .spawn()
            .context("curl not found")?;
        writeln!(curl.stdin.take().unwrap(), "header = \"Authorization: Bearer {}\"", self.key)?;
        // A spinner with the elapsed time while the cloud thinks (web searches take a while).
        let spinner = crate::show::Progress::start("☁️  cloud");
        // Wait, but give up at once when the user quits.
        let cleanup = || {
            let _ = std::fs::remove_file(&body_file);
            let _ = std::fs::remove_file(&reply_file);
        };
        let status = loop {
            if let Some(status) = curl.try_wait()? {
                break status;
            }
            if quit_requested().is_some() || cancelled() {
                let _ = curl.kill();
                let _ = curl.wait();
                cleanup();
                bail!("cancelled");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        spinner.finish();
        let body_out = std::fs::read(&reply_file).unwrap_or_default();
        cleanup();
        if !status.success() {
            bail!("curl failed: {status}");
        }
        let v: Value = serde_json::from_slice(&body_out).context("xAI reply is not JSON")?;
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            bail!("xAI: {}", e.get("message").and_then(Value::as_str).unwrap_or(&e.to_string()));
        }
        let text = v["output"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|o| o["type"] == "message")
            .flat_map(|o| o["content"].as_array().cloned().unwrap_or_default())
            .filter_map(|c| c["text"].as_str().map(str::to_string))
            .collect::<Vec<_>>()
            .join(" ");
        if text.trim().is_empty() {
            bail!("xAI reply has no text: {v}");
        }
        if let Some(id) = v["id"].as_str() {
            self.previous = Some(id.to_string());
        }
        Ok(strip_citations(&text))
    }
}

/// Grok cites sources as `[[1]](https://...)`; that is not for speaking aloud.
fn strip_citations(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("[[") {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        match after.find("]](").and_then(|j| after[j..].find(')').map(|k| j + k + 1)) {
            Some(end) => rest = &after[end..],
            None => {
                out.push_str("[[");
                rest = &after[2..];
            }
        }
    }
    out.push_str(rest);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::strip_citations;

    #[test]
    fn citations_are_removed_for_speech() {
        let t = "It will be hot. [[1]](https://www.hko.gov.hk/en/x.htm) Light winds.  [[2]](https://a.b/c)";
        assert_eq!(strip_citations(t), "It will be hot. Light winds.");
        assert_eq!(strip_citations("plain text"), "plain text");
        assert_eq!(strip_citations("odd [[ bracket"), "odd [[ bracket");
    }
}
