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

use audio_capture::{open_input_stream, record_vad, resample, rms, write_wav, SAMPLE_RATE};
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

    /// Disable wake word — record every turn using VAD only
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

    // ── Wake-word detector ────────────────────────────────────────────────────
    let mut detector: Option<WakeDetector> = if cli.no_wake {
        info!("wake word disabled — VAD-only mode");
        None
    } else {
        info!("loading wake-word models from {}", cli.models.display());
        Some(WakeDetector::new(&cli.models)?)
    };

    info!(
        "ready — {} | VAD: min={:.1}s silence={:.1}s max={:.0}s",
        if cli.no_wake { "VAD-only" } else { "say 'Hey Jarvis'" },
        cfg.min_record_seconds,
        cfg.silence_timeout,
        cfg.max_record_seconds,
    );

    let tick = Duration::from_millis(20);

    // ── Main voice loop ───────────────────────────────────────────────────────
    loop {
        // Phase 1 — wait for wake word (or skip in no-wake mode)
        if let Some(det) = &mut detector {
            info!("listening for wake word…");
            // Drain buffer so wake detector only sees fresh audio
            audio_buf.lock().unwrap().clear();
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
        }

        // Phase 2 — VAD recording: capture until silence or max duration
        info!("recording (VAD)…");
        let raw = record_vad(
            &audio_buf,
            native_sr,
            cfg.silence_threshold,
            cfg.silence_timeout,
            cfg.min_record_seconds,
            cfg.max_record_seconds,
        ).await;

        let samples = resample(raw, native_sr);
        let amp     = rms(&samples);
        debug!("recorded {:.2}s, RMS={amp:.5}",
            samples.len() as f32 / SAMPLE_RATE as f32);

        if amp < cfg.silence_threshold {
            info!("all silence, skipping");
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

        // Phase 5 — pipelined chunked TTS with barge-in
        let cfg_tts   = cfg.clone();
        let reply_tts = reply.clone();
        let buf_tts   = Arc::clone(&audio_buf);
        let barged_in = match tokio::task::spawn_blocking(
            move || tts::speak_chunked(&cfg_tts, &reply_tts, buf_tts)
        ).await.context("TTS thread panicked")? {
            Ok(v)  => v,
            Err(e) => { error!("TTS: {e:#}"); false }
        };

        if barged_in {
            // User spoke while we were talking — their voice is already in the
            // mic buffer from the barge-in monitor. Go straight to VAD.
            info!("barge-in: recording follow-up immediately");
            audio_buf.lock().unwrap().clear();
        }
        // Either way, loop continues to Phase 1/2 — if barged_in the buffer
        // already has speech so VAD will trigger immediately.
    }
}
