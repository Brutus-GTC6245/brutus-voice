mod audio_capture;
mod config;
mod llm;
mod state;
mod stt;
mod tts;
mod wake_word;
mod webserver;

use anyhow::{Context, Result};
use clap::Parser;
use std::{path::PathBuf, sync::{Arc, Mutex}, time::Duration};
use tracing::{debug, error, info};

use audio_capture::{open_input_stream, resample, rms, write_wav, SAMPLE_RATE};
use config::Config;
use state::AppState;
use wake_word::WakeDetector;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "brutus-voice", about = "Voice: wake word → Whisper STT → LLM → TTS")]
struct Cli {
    /// Path to config JSON file
    #[arg(short, long, default_value = "config.json")]
    config: PathBuf,

    /// Directory containing the ONNX wake-word models
    #[arg(long, default_value = "models")]
    models: PathBuf,

    /// Wake-word detection threshold (0.0–1.0)
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Disable wake word — record every N seconds continuously
    #[arg(long)]
    no_wake: bool,

    /// Web UI port (default: 8080)
    #[arg(long, default_value = "8080")]
    web_port: u16,

    /// Disable the web UI entirely
    #[arg(long)]
    no_web: bool,

    /// Enable debug/verbose logging
    #[arg(short, long)]
    verbose: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(if cli.verbose { "debug" } else { "info" })
            }),
        )
        .init();

    let cfg   = Config::load(&cli.config)?;
    let state = AppState::new();

    // ── Web server ────────────────────────────────────────────────────────────
    if !cli.no_web {
        webserver::serve(cli.web_port, state.clone()).await?;
    }

    // ── HTTP client (shared by STT and LLM) ──────────────────────────────────
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    // ── Microphone stream → shared ring buffer ────────────────────────────────
    let audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let (_stream, native_sr) = open_input_stream(Arc::clone(&audio_buf))?;
    // `_stream` must remain alive for the duration of the program

    // ── Wake-word detector ────────────────────────────────────────────────────
    let mut detector: Option<WakeDetector> = if cli.no_wake {
        info!("wake word disabled — recording continuously");
        None
    } else {
        info!("loading wake-word models from {}", cli.models.display());
        Some(WakeDetector::new(&cli.models)?)
    };

    info!("ready — {}", if cli.no_wake { "recording continuously" } else { "say 'Hey Jarvis'" });

    let record_dur     = Duration::from_secs(cfg.record_seconds);
    let tick           = Duration::from_millis(20);
    let record_samples = (native_sr as u64 * cfg.record_seconds) as usize;

    // ── Main voice loop ───────────────────────────────────────────────────────
    loop {
        // Phase 1 — wait for wake word (or fill buffer in no-wake mode)
        if let Some(det) = &mut detector {
            info!("listening for wake word…");
            loop {
                tokio::time::sleep(tick).await;
                let chunk: Vec<f32> = audio_buf.lock().unwrap().drain(..).collect();
                if chunk.is_empty() { continue; }

                let chunk16 = resample(chunk, native_sr);
                match det.process(&chunk16) {
                    Ok(Some(score)) if score >= cli.threshold => {
                        info!("wake word! score={score:.3}");
                        audio_buf.lock().unwrap().clear();
                        break;
                    }
                    Err(e) => error!("wake detector: {e:#}"),
                    _ => {}
                }
            }
        } else {
            loop {
                tokio::time::sleep(tick).await;
                if audio_buf.lock().unwrap().len() >= record_samples { break; }
            }
        }

        // Phase 2 — record for `record_seconds`
        info!("recording {}s…", cfg.record_seconds);
        audio_buf.lock().unwrap().clear();
        tokio::time::sleep(record_dur).await;

        let raw     = audio_buf.lock().unwrap().drain(..).collect::<Vec<_>>();
        let samples = resample(raw, native_sr);
        let amp     = rms(&samples);
        debug!("RMS={amp:.5}");

        if amp < cfg.silence_threshold {
            info!("silence (RMS {amp:.5}), skipping");
            continue;
        }

        // Phase 3 — speech-to-text
        let wav = match tokio::task::spawn_blocking(
            move || write_wav(&samples, SAMPLE_RATE)
        ).await? {
            Ok(w)  => w,
            Err(e) => { error!("WAV write: {e:#}"); continue; }
        };

        let transcript = match stt::transcribe(&http, &cfg.whisper_url, wav.path()).await {
            Ok(t) if t.is_empty() => { info!("empty transcript, skipping"); continue; }
            Ok(t)  => t,
            Err(e) => { error!("STT: {e:#}"); continue; }
        };
        info!("you: {transcript}");
        state.push("user", &transcript);

        // Phase 4 — LLM
        let reply = match llm::complete(&http, &cfg, &transcript).await {
            Ok(r)  => r,
            Err(e) => { error!("LLM: {e:#}"); continue; }
        };
        info!("brutus: {reply}");
        state.push("assistant", &reply);

        // Phase 5 — text-to-speech
        let cmd       = cfg.tts_command.clone();
        let reply_tts = reply.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || tts::speak(&cmd, &reply_tts))
            .await
            .context("TTS thread panicked")?
        {
            error!("TTS: {e:#}");
        }
    }
}
