use anyhow::{bail, Context, Result};
use std::process::Command;
use tracing::{debug, warn};

/// Speak `text` by running the configured TTS command.
///
/// The command is specified as `["program", "arg1", ...]` in `config.tts_command`;
/// the reply text is appended as the final argument — compatible with both
/// `say` (macOS) and the `speak.sh` Piper wrapper.
pub fn speak(tts_command: &[String], text: &str) -> Result<()> {
    let (program, args) = tts_command
        .split_first()
        .context("tts_command must have at least one element")?;

    if text.trim().is_empty() {
        return Ok(());
    }

    debug!("TTS via `{program}`: {text}");

    let status = Command::new(program)
        .args(args)
        .arg(text)
        .status()
        .with_context(|| format!("running TTS command `{program}`"))?;

    if !status.success() {
        warn!("TTS command exited with status: {status}");
    }

    Ok(())
}
