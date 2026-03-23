use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::config::Config;

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

/// Send a single-turn prompt to the OpenAI-compatible chat completions endpoint
/// and return the assistant's reply text.
pub async fn complete(
    client: &reqwest::Client,
    cfg: &Config,
    prompt: &str,
) -> Result<String> {
    let url = format!("{}/chat/completions", cfg.api_url.trim_end_matches('/'));

    let body = ChatRequest {
        model: "default".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    debug!("sending to LLM: {url}");

    let resp = client
        .post(&url)
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .await
        .context("POST to chat completions")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text   = resp.text().await.unwrap_or_default();
        bail!("chat API returned {status}: {text}");
    }

    let parsed: ChatResponse = resp.json().await.context("parsing chat response")?;

    Ok(parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default()
        .trim()
        .to_string())
}
