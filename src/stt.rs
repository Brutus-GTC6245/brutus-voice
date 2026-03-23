use anyhow::{bail, Context, Result};
use reqwest::multipart;
use std::path::Path;
use tracing::debug;

/// Send a WAV file to the Whisper HTTP server and return the transcript.
///
/// The server is expected to accept `POST /asr` as `multipart/form-data`
/// with a field named `audio_file`, and respond with `{"text": "..."}`.
pub async fn transcribe(
    client: &reqwest::Client,
    url: &str,
    wav_path: &Path,
) -> Result<String> {
    let bytes = tokio::fs::read(wav_path)
        .await
        .with_context(|| format!("reading WAV: {}", wav_path.display()))?;

    let part = multipart::Part::bytes(bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = multipart::Form::new().part("audio_file", part);

    let resp = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .context("POST to Whisper server")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body   = resp.text().await.unwrap_or_default();
        bail!("Whisper returned {status}: {body}");
    }

    let body = resp.text().await.context("reading Whisper response body")?;
    debug!("whisper raw: {body}");

    // Parse {"text": "..."} — fall back to raw body if not JSON
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
        if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
            return Ok(t.trim().to_string());
        }
    }

    Ok(body.trim().to_string())
}
