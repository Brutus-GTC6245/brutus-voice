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

/// Spawn a background thread that feeds `audio_buf` through `detector`
/// and sets `stop_flag` when the wake word fires above `threshold`.
///
/// Returns a join handle (can be dropped — thread is detached on interrupt).
pub fn spawn_barge_in_monitor(
    audio_buf: Arc<Mutex<Vec<f32>>>,
    native_sr: u32,
    threshold: f32,
    stop_flag: StopFlag,
    afplay_pid: Arc<Mutex<Option<u32>>>,
    models_dir: std::path::PathBuf,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        use crate::audio_capture::resample;
        use crate::wake_word::WakeDetector;

        let mut det = match WakeDetector::new(&models_dir) {
            Ok(d)  => d,
            Err(e) => { warn!("barge-in detector failed to load: {e:#}"); return; }
        };

        loop {
            // Exit if stop already set (TTS finished normally)
            if *stop_flag.lock().unwrap() { return; }

            thread::sleep(Duration::from_millis(20));

            let chunk: Vec<f32> = audio_buf.lock().unwrap().drain(..).collect();
            if chunk.is_empty() { continue; }

            let chunk16 = resample(chunk, native_sr);
            match det.process(&chunk16) {
                Ok(Some(score)) if score >= threshold => {
                    info!("barge-in: wake word detected (score={score:.3})");
                    *stop_flag.lock().unwrap() = true;
                    // Kill afplay if running
                    if let Some(pid) = *afplay_pid.lock().unwrap() {
                        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
                    }
                    return;
                }
                Err(e) => { warn!("barge-in detector: {e:#}"); }
                _ => {}
            }
        }
    })
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
    wake_threshold: f32,
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

    // Start barge-in monitor if models_dir provided
    let _monitor = models_dir.map(|dir| {
        spawn_barge_in_monitor(
            Arc::clone(&audio_buf),
            native_sr,
            wake_threshold,
            Arc::clone(&stop_flag),
            Arc::clone(&afplay_pid),
            dir,
        )
    });

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
