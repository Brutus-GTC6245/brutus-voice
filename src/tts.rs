use anyhow::{bail, Context, Result};
use std::{
    io::Write,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tracing::{debug, info, warn};

use crate::config::Config;

// ── Simple (fallback) TTS ─────────────────────────────────────────────────────

/// Speak `text` synchronously via `tts_command`.
pub fn speak_simple(tts_command: &[String], text: &str) -> Result<()> {
    if text.trim().is_empty() { return Ok(()); }
    let (prog, args) = tts_command.split_first()
        .context("tts_command must have at least one element")?;
    let status = Command::new(prog).args(args).arg(text).status()
        .with_context(|| format!("running TTS `{prog}`"))?;
    if !status.success() { warn!("TTS exited {status}"); }
    Ok(())
}

// ── Chunk splitting ───────────────────────────────────────────────────────────

const WORDS_PER_CHUNK: usize = 8;

pub fn split_chunks(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let s = current.trim().to_string();
            if !s.is_empty() { sentences.push(s); }
            current.clear();
        }
    }
    if !current.trim().is_empty() { sentences.push(current.trim().to_string()); }

    let mut chunks: Vec<String> = Vec::new();
    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for group in words.chunks(WORDS_PER_CHUNK) {
            let c = group.join(" ");
            if !c.is_empty() { chunks.push(c); }
        }
    }
    if chunks.is_empty() {
        let s = text.trim().to_string();
        if !s.is_empty() { chunks.push(s); }
    }
    chunks
}

// ── Piper synthesis ───────────────────────────────────────────────────────────

fn piper_synthesise(
    piper_bin: &str,
    piper_model: &str,
    text: &str,
) -> Result<tempfile::NamedTempFile> {
    let wav = tempfile::Builder::new().suffix(".wav").tempfile()
        .context("creating temp WAV")?;
    let mut child = Command::new(piper_bin)
        .args(["--model", piper_model, "--output_file",
               wav.path().to_str().unwrap_or("")])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawning piper `{piper_bin}`"))?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    let status = child.wait()?;
    if !status.success() { bail!("piper exited {status}"); }
    Ok(wav)
}

// ── Barge-in afplay wrapper ───────────────────────────────────────────────────

/// Play a WAV file via afplay. While playing, monitor `audio_buf` for speech.
/// Kills playback and returns `true` if the user starts speaking (barge-in).
/// Returns `false` if playback completed normally.
fn play_wav_with_barge_in(
    path: &std::path::Path,
    audio_buf: &Arc<Mutex<Vec<f32>>>,
    barge_in_threshold: f32,
) -> Result<bool> {
    let mut afplay = Command::new("afplay")
        .arg(path)
        .spawn()
        .context("spawning afplay")?;

    // Shared flag: set by monitor thread when barge-in detected
    let interrupted = Arc::new(Mutex::new(false));
    let interrupted_clone = Arc::clone(&interrupted);
    let buf_clone = Arc::clone(audio_buf);
    let pid = afplay.id();

    // Background mic monitor thread
    let monitor = thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(40));

            // Check if afplay is still running (simple: try to send signal 0)
            // We detect completion via the interrupted flag staying false after join

            let chunk: Vec<f32> = buf_clone.lock().unwrap().drain(..).collect();
            if chunk.is_empty() { continue; }

            let rms = {
                let sum: f32 = chunk.iter().map(|s| s * s).sum();
                (sum / chunk.len() as f32).sqrt()
            };

            if rms > barge_in_threshold {
                info!("barge-in detected (RMS={rms:.5})");
                *interrupted_clone.lock().unwrap() = true;
                // Kill afplay
                let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
                return;
            }
        }
    });

    // Wait for afplay to finish (either naturally or killed)
    let _ = afplay.wait();

    // Signal monitor to stop by checking: if afplay exited and not interrupted,
    // the monitor will keep looping — we just detach it (daemon thread).
    // The important thing is we read the interrupted flag before returning.
    let was_interrupted = *interrupted.lock().unwrap();

    if was_interrupted {
        drop(monitor); // detach — it already killed afplay and returned
    }

    Ok(was_interrupted)
}

// ── Main pipelined TTS entry point ───────────────────────────────────────────

/// Speak `text` using pipelined Piper TTS with barge-in support.
///
/// Returns `true` if the user interrupted playback, `false` if it completed.
/// Falls back to `speak_simple` (no barge-in) if piper is not configured.
pub fn speak_chunked(
    cfg: &Config,
    text: &str,
    audio_buf: Arc<Mutex<Vec<f32>>>,
) -> Result<bool> {
    let (Some(piper_bin), Some(piper_model)) =
        (cfg.piper_binary.as_deref(), cfg.piper_model.as_deref())
    else {
        speak_simple(&cfg.tts_command, text)?;
        return Ok(false);
    };

    if text.trim().is_empty() { return Ok(false); }

    let chunks = split_chunks(text);
    debug!("TTS chunks ({}): {:?}", chunks.len(), chunks);
    if chunks.is_empty() { return Ok(false); }

    let barge_in_threshold = cfg.silence_threshold * 1.5; // slightly above ambient

    let piper_bin   = piper_bin.to_string();
    let piper_model = piper_model.to_string();

    // Synthesise first chunk immediately
    let first_wav = piper_synthesise(&piper_bin, &piper_model, &chunks[0])?;

    type WavSlot = Arc<Mutex<Option<Result<tempfile::NamedTempFile>>>>;
    let next_slot: WavSlot = Arc::new(Mutex::new(None));

    for i in 0..chunks.len() {
        // Pre-synthesise next chunk in background while current plays
        if i + 1 < chunks.len() {
            let slot   = Arc::clone(&next_slot);
            let text   = chunks[i + 1].clone();
            let bin    = piper_bin.clone();
            let model  = piper_model.clone();
            *slot.lock().unwrap() = None;
            thread::spawn(move || {
                *slot.lock().unwrap() = Some(piper_synthesise(&bin, &model, &text));
            });
        }

        // Get path for current chunk's WAV
        let wav_path = if i == 0 {
            first_wav.path().to_path_buf()
        } else {
            loop {
                if next_slot.lock().unwrap().is_some() { break; }
                thread::sleep(Duration::from_millis(5));
            }
            match next_slot.lock().unwrap().take().unwrap() {
                Ok(f)  => f.into_temp_path().keep()
                           .unwrap_or_else(|e| { warn!("keep failed: {e}"); std::path::PathBuf::new() }),
                Err(e) => { warn!("piper failed chunk {i}: {e:#}"); continue; }
            }
        };

        // Play with barge-in monitoring
        match play_wav_with_barge_in(&wav_path, &audio_buf, barge_in_threshold) {
            Ok(true) => {
                info!("playback interrupted at chunk {}/{}", i + 1, chunks.len());
                return Ok(true); // caller should go straight to recording
            }
            Ok(false) => {} // continue to next chunk
            Err(e) => warn!("playback error chunk {i}: {e:#}"),
        }
    }

    Ok(false)
}
