//! The brain: Apple's on-device foundation model (default, through the one `apple-ai` process
//! that also routes), or Qwen3.5 on MLX in this process with `--brain mlx`.

use anyhow::{Result, anyhow};
use qwen35::audio::listen::quit_requested;
use qwen35::llm::{Chat, Engine, Sampler};

use crate::apple::AppleChat;

/// Who Jarvis is. Shared by every model that speaks for it.
pub const PERSONA: &str = "You are Jarvis, a voice assistant running on the user's Mac. The user's words come \
    from speech recognition and may contain small errors. Reply in one to three short sentences of plain \
    spoken language, in the language the user speaks, with no markdown, lists, links or emoji. If a question \
    needs live or recent information you do not have, say so briefly.";

/// The persona plus what Jarvis can actually do, so questions like "can you use cloud AI?" or
/// "can you see my screen?" get true answers. `cloud` names the cloud model when a key is set.
pub fn persona(cloud: Option<&str>) -> String {
    let cloud_text = match cloud {
        Some(model) => format!(
            "Cloud AI is available: if the user starts a request with the word AI (for example \"AI, what is the \
             weather tomorrow\"), or a question needs live or recent information, Jarvis sends it to {model}, a \
             cloud model from xAI with web search. Everything else stays on this Mac."
        ),
        None => "Cloud AI is currently off (no XAI_API_KEY is set), so everything runs on this Mac.".to_string(),
    };
    format!(
        "{PERSONA} About yourself, Jarvis: you run on the user's Mac and can hear the user through the microphone \
         and talk back out loud; you can look at the user's screen when asked (\"what is on my screen\"); you can \
         make a picture when asked (\"imagine a red apple\") and describe a picture from a file or URL; you keep \
         listening while you talk, so the user can interrupt you. {cloud_text}"
    )
}

/// Start a fresh Qwen conversation once the cache holds this many tokens.
const RESET_AT_TOKENS: usize = 6000;

pub enum Brain {
    /// Apple's on-device model; the conversation lives in the apple-ai helper's session.
    Apple,
    /// Qwen3.5 on MLX; the conversation lives in the KV cache.
    Mlx(Chat),
}

// SAFETY: the Qwen model carries a raw MLX kernel handle (not `Send` by default). The brain is
// created on the main thread and then owned and used only by the worker thread, never shared.
unsafe impl Send for Brain {}

impl Brain {
    pub fn load_mlx(model: &str, temp: f32, top_k: i32, max_tokens: usize, instructions: &str) -> Result<Self> {
        let engine = Engine::load(model, Sampler { temp, top_k }, true)?;
        Ok(Self::Mlx(Chat::new(engine, Some(instructions.to_string()), false, max_tokens)))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Apple => "Apple on-device model",
            Self::Mlx(_) => "Qwen3.5 on MLX",
        }
    }

    /// Answer `prompt`, handing the reply to `on_text` as it streams; returns the whole text and
    /// an optional note (generation statistics) to print once the reply line is closed.
    /// `apple` is the shared apple-ai process (needed for the Apple brain). `on_text` returns
    /// false to stop generating (Qwen stops at once; Apple's short reply is read to the end).
    pub fn reply(
        &mut self,
        prompt: &str,
        apple: Option<&mut AppleChat>,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, Option<String>)> {
        let mut note = None;
        let text = match self {
            Self::Apple => {
                let apple = apple.ok_or_else(|| anyhow!("the apple-ai helper is not running"))?;
                let mut go_on = true;
                apple.ask(prompt, |t| {
                    if go_on {
                        go_on = on_text(t);
                    }
                })?
            }
            Self::Mlx(chat) => {
                if chat.tokens() > RESET_AT_TOKENS {
                    chat.reset();
                }
                let mut cut = false;
                let (text, stats) = chat.reply(prompt, |t| {
                    let go_on = on_text(t) && quit_requested().is_none();
                    cut |= !go_on;
                    go_on
                })?;
                note = Some(format!("[brain: {} tok @ {:.0} tok/s]", stats.gen_tokens, stats.gen_tps));
                if cut {
                    // The cache holds a half-finished turn; start clean next time.
                    chat.reset();
                }
                text
            }
        };
        Ok((text.trim().to_string(), note))
    }
}

/// The human part of an error. Apple's NSError dumps are hundreds of characters of nested
/// UserInfo; keep the quoted description: "The operation couldn’t be completed. (com.apple...)".
pub fn brief(error: &anyhow::Error) -> String {
    let text = error.to_string();
    if let Some(start) = text.find("Code=") {
        if let Some(q1) = text[start..].find('"') {
            let rest = &text[start + q1 + 1..];
            if let Some(q2) = rest.find('"') {
                return rest[..q2].to_string();
            }
        }
    }
    let first_line = text.lines().next().unwrap_or("").trim();
    if first_line.chars().count() > 160 {
        let cut: String = first_line.chars().take(157).collect();
        format!("{cut}...")
    } else {
        first_line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::brief;

    #[test]
    fn apple_errors_are_condensed() {
        let e = anyhow::anyhow!(
            "Error Domain=FoundationModels.LanguageModelError Code=-1 \"The operation couldn’t be completed. \
             (com.apple.SensitiveContentAnalysisML error 15.)\" UserInfo={{NSMultipleUnderlyingErrorsKey=(\n  \"Error ...\"\n)}} (other)"
        );
        assert_eq!(brief(&e), "The operation couldn’t be completed. (com.apple.SensitiveContentAnalysisML error 15.)");
        assert_eq!(brief(&anyhow::anyhow!("apple-ai exited")), "apple-ai exited");
    }
}
