use anyhow::{bail, Context, Result};
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};
use tracing::{debug, warn};

use crate::config::Config;

// ── Simple (fallback) TTS ─────────────────────────────────────────────────────

/// Speak `text` synchronously via `tts_command` (e.g. `say` or `speak.sh`).
/// Text is appended as the final argument.
pub fn speak_simple(tts_command: &[String], text: &str) -> Result<()> {
    if text.trim().is_empty() { return Ok(()); }
    let (prog, args) = tts_command.split_first()
        .context("tts_command must have at least one element")?;
    debug!("TTS (simple): {text}");
    let status = Command::new(prog).args(args).arg(text).status()
        .with_context(|| format!("running TTS command `{prog}`"))?;
    if !status.success() { warn!("TTS exited {status}"); }
    Ok(())
}

// ── Chunked / pipelined Piper TTS ─────────────────────────────────────────────
//
// Strategy:
//   1. Split the reply into sentence-boundary chunks (≤ WORDS_PER_CHUNK words,
//      always splitting at sentence ends when possible).
//   2. Synthesise chunk[0] → audio bytes in a blocking thread.
//   3. While afplay plays chunk[0], synthesise chunk[1] in parallel.
//   4. Hand off each synthesised buffer to a single-slot "next audio" queue
//      so the play thread always has something ready.

const WORDS_PER_CHUNK: usize = 8;

/// Split reply text into playable chunks.
/// Splits on sentence endings first, then falls back to word-count groups.
pub fn split_chunks(text: &str) -> Vec<String> {
    // First split on sentence boundaries
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
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    // If sentences are very long, further split by word count
    let mut chunks: Vec<String> = Vec::new();
    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        for group in words.chunks(WORDS_PER_CHUNK) {
            let chunk = group.join(" ");
            if !chunk.is_empty() { chunks.push(chunk); }
        }
    }

    if chunks.is_empty() {
        // fallback: just return the whole thing
        let s = text.trim().to_string();
        if !s.is_empty() { chunks.push(s); }
    }

    chunks
}

/// Synthesise `text` to a temporary WAV file using Piper (blocking).
/// Returns the NamedTempFile so it stays alive until the caller drops it.
fn piper_synthesise(
    piper_bin: &str,
    piper_model: &str,
    text: &str,
) -> Result<tempfile::NamedTempFile> {
    let wav_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .context("creating temp WAV for piper")?;

    let mut child = Command::new(piper_bin)
        .args(["--model", piper_model, "--output_file",
               wav_file.path().to_str().unwrap_or("")])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawning piper `{piper_bin}`"))?;

    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    let status = child.wait()?;
    if !status.success() {
        bail!("piper exited {status}");
    }

    Ok(wav_file)
}

/// Play a WAV file via afplay (blocks until playback complete).
fn play_wav(path: &std::path::Path) -> Result<()> {
    let status = Command::new("afplay")
        .arg(path)
        .status()?;
    if !status.success() { warn!("afplay exited {status}"); }
    Ok(())
}

/// Speak `text` using pipelined Piper TTS:
/// synthesise and play chunks concurrently to minimise latency.
///
/// Falls back to `speak_simple` if piper_binary / piper_model are not configured.
pub fn speak_chunked(cfg: &Config, text: &str) -> Result<()> {
    let (Some(piper_bin), Some(piper_model)) =
        (cfg.piper_binary.as_deref(), cfg.piper_model.as_deref())
    else {
        return speak_simple(&cfg.tts_command, text);
    };

    if text.trim().is_empty() { return Ok(()); }

    let chunks = split_chunks(text);
    debug!("TTS chunks ({}): {:?}", chunks.len(), chunks);

    if chunks.is_empty() { return Ok(()); }

    // Shared queue: Option<Vec<u8>> = next pre-synthesised audio
    // None = not ready yet; Some(bytes) = ready to play


    let piper_bin   = piper_bin.to_string();
    let piper_model = piper_model.to_string();

    // Synthesise first chunk synchronously so we can start playing immediately
    let first_wav = piper_synthesise(&piper_bin, &piper_model, &chunks[0])?;

    // Slot holds the pre-synthesised WAV for the next chunk
    type WavSlot = Arc<Mutex<Option<Result<tempfile::NamedTempFile>>>>;
    let next_slot: WavSlot = Arc::new(Mutex::new(None));

    for i in 0..chunks.len() {
        // Kick off synthesis of chunk i+1 in background while current plays
        let has_next = i + 1 < chunks.len();
        if has_next {
            let slot_clone  = Arc::clone(&next_slot);
            let next_text   = chunks[i + 1].clone();
            let bin_clone   = piper_bin.clone();
            let model_clone = piper_model.clone();
            *slot_clone.lock().unwrap() = None;
            thread::spawn(move || {
                let result = piper_synthesise(&bin_clone, &model_clone, &next_text);
                *slot_clone.lock().unwrap() = Some(result);
            });
        }

        // Play current chunk (blocks until audio finishes)
        let wav = if i == 0 {
            first_wav.path().to_path_buf()
        } else {
            // Wait for pre-synthesised chunk to be ready
            loop {
                if next_slot.lock().unwrap().is_some() { break; }
                thread::sleep(std::time::Duration::from_millis(5));
            }
            match next_slot.lock().unwrap().take().unwrap() {
                Ok(f)  => f.into_temp_path().keep()
                           .unwrap_or_else(|e| { warn!("keep failed: {e}"); std::path::PathBuf::new() }),
                Err(e) => { warn!("piper synthesis failed for chunk {i}: {e:#}"); continue; }
            }
        };

        if let Err(e) = play_wav(&wav) {
            warn!("playback error chunk {i}: {e:#}");
        }
    }

    Ok(())
}
