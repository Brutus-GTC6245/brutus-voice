use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    /// Base URL of the OpenAI-compatible chat API (e.g. "http://localhost:18789/v1")
    pub api_url: String,
    /// Bearer token for the chat API
    pub api_key: String,
    /// URL of the Whisper transcription endpoint
    pub whisper_url: String,
    /// TTS command: program + args; reply text is appended as the final argument
    pub tts_command: Vec<String>,
    /// Seconds of audio to record per turn (default: 6)
    #[serde(default = "default_record_seconds")]
    pub record_seconds: u64,
    /// RMS amplitude below which audio is treated as silence (default: 0.01)
    #[serde(default = "default_silence_threshold")]
    pub silence_threshold: f32,
}

fn default_record_seconds() -> u64 { 6 }
fn default_silence_threshold() -> f32 { 0.01 }

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config: {}", path.display()))?;
        serde_json::from_str(&text).context("parsing config JSON")
    }
}
