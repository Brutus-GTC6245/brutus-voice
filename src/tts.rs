use anyhow::{bail, Context, Result};
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tracing::{debug, info, warn};

use crate::config::Config;

// ── Simple (fallback) TTS ─────────────────────────────────────────────────────

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
//
// Split on punctuation that marks a natural speech pause.
// The delimiter itself is stripped — Piper should not say "comma" or "dash".
// Minimum chunk length avoids tiny fragments like a single word after a colon.

const MIN_CHUNK_WORDS: usize = 3;

/// Characters that act as chunk delimiters (stripped from output).
fn is_delimiter(ch: char) -> bool {
    matches!(ch, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '–' | '—' | '\n')
}

pub fn split_chunks(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if is_delimiter(ch) {
            let candidate = current.trim().to_string();
            // Only emit if long enough — avoids tiny orphan fragments
            let word_count = candidate.split_whitespace().count();
            if word_count >= MIN_CHUNK_WORDS {
                chunks.push(candidate);
                current.clear();
            } else if !candidate.is_empty() {
                // Too short: keep accumulating (append a space for readability)
                current.push(' ');
            }
        } else {
            current.push(ch);
        }
    }

    // Flush remainder
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        if let Some(last) = chunks.last_mut() {
            // Merge short tail into previous chunk
            if tail.split_whitespace().count() < MIN_CHUNK_WORDS {
                last.push(' ');
                last.push_str(&tail);
                return chunks;
            }
        }
        chunks.push(tail);
    }

    // Fallback: if nothing split, return whole text
    if chunks.is_empty() {
        let s = text.trim().to_string();
        if !s.is_empty() { chunks.push(s); }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::split_chunks;
    #[test]
    fn test_split_basic() {
        let chunks = split_chunks("Hello there. How are you doing today? I'm doing great, thanks!");
        println!("{chunks:?}");
        assert!(chunks.len() >= 2);
        // No chunk should contain a raw delimiter at the end
        for c in &chunks { assert!(!c.ends_with('.') && !c.ends_with(',')) }
    }
}

// ── Piper synthesis ───────────────────────────────────────────────────────────

fn piper_synthesise(bin: &str, model: &str, text: &str) -> Result<tempfile::NamedTempFile> {
    let wav = tempfile::Builder::new().suffix(".wav").tempfile()
        .context("creating temp WAV")?;
    let mut child = Command::new(bin)
        .args(["--model", model, "--output_file",
               wav.path().to_str().unwrap_or("")])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawning piper `{bin}`"))?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    let status = child.wait()?;
    if !status.success() { bail!("piper exited {status}"); }
    Ok(wav)
}

// ── Wake-word barge-in monitor ────────────────────────────────────────────────
//
// During TTS playback we run the wake detector in a background thread.
// Shared stop flag: set → true when wake word fires.
// afplay PID: stored so the monitor can kill it.

pub type StopFlag = Arc<Mutex<bool>>;

/// Spawn a background thread that monitors `audio_buf` with Silero VAD.
/// Sets `stop_flag` and kills afplay when speech is detected above `threshold`.
///
/// Silero VAD is pre-loaded before the thread starts, so there is no startup
/// delay once TTS begins playing.
pub fn spawn_barge_in_monitor(
    audio_buf: Arc<Mutex<Vec<f32>>>,
    native_sr: u32,
    threshold: f32,   // Silero speech probability, e.g. 0.5
    stop_flag: StopFlag,
    afplay_pid: Arc<Mutex<Option<u32>>>,
    models_dir: std::path::PathBuf,
) -> Result<thread::JoinHandle<()>> {
    use crate::audio_capture::resample;
    use crate::wake_word::SileroVad;

    // Load VAD synchronously BEFORE spawning so the thread starts ready
    let mut vad = SileroVad::new(&models_dir)?;
    vad.reset();

    const CHUNK: usize = SileroVad::CHUNK_SAMPLES; // 512 samples @ 16 kHz
    let mut remainder: Vec<f32> = Vec::new();

    Ok(thread::spawn(move || {
        loop {
            if *stop_flag.lock().unwrap() { return; }

            thread::sleep(Duration::from_millis(15));

            let new: Vec<f32> = audio_buf.lock().unwrap().drain(..).collect();
            if new.is_empty() { continue; }

            // Resample to 16 kHz if needed
            let resampled = resample(new, native_sr);
            remainder.extend_from_slice(&resampled);

            while remainder.len() >= CHUNK {
                let chunk: Vec<f32> = remainder.drain(..CHUNK).collect();
                match vad.process(&chunk) {
                    Ok(score) => {
                        debug!("silero vad: {score:.3}");
                        if score >= threshold {
                            info!("barge-in: speech detected (silero={score:.3})");
                            *stop_flag.lock().unwrap() = true;
                            if let Some(pid) = *afplay_pid.lock().unwrap() {
                                let _ = Command::new("kill")
                                    .args(["-9", &pid.to_string()])
                                    .status();
                            }
                            return;
                        }
                    }
                    Err(e) => warn!("silero vad error: {e:#}"),
                }
            }
        }
    }))
}

// ── Play WAV with barge-in watch ─────────────────────────────────────────────

/// Play a WAV via afplay, watching `stop_flag`. Returns true if interrupted.
fn play_wav(path: &Path, stop_flag: &StopFlag, afplay_pid: &Arc<Mutex<Option<u32>>>) -> bool {
    let mut child = match Command::new("afplay").arg(path).spawn() {
        Ok(c)  => c,
        Err(e) => { warn!("afplay spawn: {e}"); return false; }
    };

    // Store pid so monitor can kill it
    *afplay_pid.lock().unwrap() = Some(child.id());

    // Poll until afplay exits or stop_flag is set
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                *afplay_pid.lock().unwrap() = None;
                return *stop_flag.lock().unwrap(); // done — was it interrupted?
            }
            Ok(None) => {
                if *stop_flag.lock().unwrap() {
                    let _ = child.kill();
                    let _ = child.wait();
                    *afplay_pid.lock().unwrap() = None;
                    return true;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => { warn!("afplay wait: {e}"); return false; }
        }
    }
}

// ── Main pipelined TTS entry point ───────────────────────────────────────────

/// Speak `text` using pipelined Piper TTS.
///
/// If `models_dir` is Some, runs the wake-word detector in a background thread
/// during playback for barge-in support. Returns `true` if interrupted.
pub fn speak_chunked(
    cfg: &Config,
    text: &str,
    audio_buf: Arc<Mutex<Vec<f32>>>,
    native_sr: u32,
    models_dir: Option<std::path::PathBuf>,
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

    // Shared state for barge-in
    let stop_flag: StopFlag = Arc::new(Mutex::new(false));
    let afplay_pid: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

    // Start Silero VAD barge-in monitor if models_dir provided
    let _monitor = if let Some(dir) = models_dir {
        match spawn_barge_in_monitor(
            Arc::clone(&audio_buf),
            native_sr,
            0.5, // Silero speech probability threshold
            Arc::clone(&stop_flag),
            Arc::clone(&afplay_pid),
            dir,
        ) {
            Ok(h)  => Some(h),
            Err(e) => { warn!("barge-in monitor failed to start: {e:#}"); None }
        }
    } else {
        None
    };

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
            let txt    = chunks[i + 1].clone();
            let bin    = piper_bin.clone();
            let model  = piper_model.clone();
            *slot.lock().unwrap() = None;
            thread::spawn(move || {
                *slot.lock().unwrap() = Some(piper_synthesise(&bin, &model, &txt));
            });
        }

        // Get current WAV path
        let wav_path = if i == 0 {
            first_wav.path().to_path_buf()
        } else {
            loop {
                if next_slot.lock().unwrap().is_some() { break; }
                thread::sleep(Duration::from_millis(5));
            }
            match next_slot.lock().unwrap().take().unwrap() {
                Ok(f)  => f.into_temp_path().keep()
                           .unwrap_or_else(|e| { warn!("keep: {e}"); std::path::PathBuf::new() }),
                Err(e) => { warn!("piper chunk {i}: {e:#}"); continue; }
            }
        };

        // Play — returns true if barge-in interrupted
        let interrupted = play_wav(&wav_path, &stop_flag, &afplay_pid);
        if interrupted {
            info!("barge-in: stopped at chunk {}/{}", i + 1, chunks.len());
            // Signal monitor to exit
            *stop_flag.lock().unwrap() = true;
            return Ok(true);
        }
    }

    // Signal monitor to exit cleanly
    *stop_flag.lock().unwrap() = true;
    Ok(false)
}
