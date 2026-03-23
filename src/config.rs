use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    /// Base URL of the OpenAI-compatible chat API
    pub api_url: String,
    /// Bearer token for the chat API
    pub api_key: String,
    /// URL of the Whisper transcription endpoint
    pub whisper_url: String,

    /// TTS command used for simple (non-pipelined) fallback:
    /// program + args; reply text is appended as the final argument
    pub tts_command: Vec<String>,

    /// Path to the Piper binary (used for chunked/pipelined TTS)
    /// If absent, chunked TTS falls back to tts_command
    #[serde(default)]
    pub piper_binary: Option<String>,

    /// Path to the Piper ONNX voice model
    #[serde(default)]
    pub piper_model: Option<String>,

    /// Minimum seconds to record before checking for end-of-speech (default: 0.5)
    #[serde(default = "default_min_record_seconds")]
    pub min_record_seconds: f32,

    /// Maximum seconds to record before forcing a cutoff (default: 30)
    #[serde(default = "default_max_record_seconds")]
    pub max_record_seconds: f32,

    /// Seconds of silence that signals end-of-speech (default: 2.0)
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout: f32,

    /// RMS amplitude below which audio is treated as silence (default: 0.01)
    #[serde(default = "default_silence_threshold")]
    pub silence_threshold: f32,

}

fn default_min_record_seconds() -> f32 { 0.5 }
fn default_max_record_seconds() -> f32 { 30.0 }
fn default_silence_timeout() -> f32 { 2.0 }
fn default_silence_threshold() -> f32 { 0.01 }

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config: {}", path.display()))?;
        serde_json::from_str(&text).context("parsing config JSON")
    }
}
