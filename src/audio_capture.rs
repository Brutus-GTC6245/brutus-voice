use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::{debug, error, info};

pub const SAMPLE_RATE: u32 = 16000;

/// Open the default input device and stream mono f32 samples into `audio_buf`.
/// Returns the stream (must be kept alive) and the device's native sample rate.
pub fn open_input_stream(
    audio_buf: Arc<Mutex<Vec<f32>>>,
) -> Result<(cpal::Stream, u32)> {
    let host   = cpal::default_host();
    let device = host.default_input_device().context("no default input device")?;
    info!("mic: {}", device.name().unwrap_or_default());

    let cfg = device.default_input_config()?;
    let sr  = cfg.sample_rate().0;
    let ch  = cfg.channels() as usize;
    debug!("native sr={sr} channels={ch}");

    let err_fn = |e| error!("audio stream error: {e}");

    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let buf = Arc::clone(&audio_buf);
            device.build_input_stream(
                &cfg.into(),
                move |data: &[f32], _| {
                    let mut b = buf.lock().unwrap();
                    for chunk in data.chunks(ch) {
                        b.push(chunk.iter().sum::<f32>() / ch as f32);
                    }
                },
                err_fn, None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let buf = Arc::clone(&audio_buf);
            device.build_input_stream(
                &cfg.into(),
                move |data: &[i16], _| {
                    let mut b = buf.lock().unwrap();
                    for chunk in data.chunks(ch) {
                        let mono = chunk.iter()
                            .map(|&x| x as f32 / i16::MAX as f32)
                            .sum::<f32>() / ch as f32;
                        b.push(mono);
                    }
                },
                err_fn, None,
            )?
        }
        fmt => bail!("unsupported sample format: {fmt:?}"),
    };

    stream.play().context("failed to start audio stream")?;
    Ok((stream, sr))
}

/// Linear-interpolation resample from `src_rate` to `SAMPLE_RATE`.
pub fn resample(samples: Vec<f32>, src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE { return samples; }
    let ratio   = SAMPLE_RATE as f64 / src_rate as f64;
    let out_len = (samples.len() as f64 * ratio) as usize;
    (0..out_len).map(|i| {
        let src  = i as f64 / ratio;
        let lo   = src.floor() as usize;
        let hi   = (lo + 1).min(samples.len().saturating_sub(1));
        let frac = (src - lo as f64) as f32;
        samples[lo] * (1.0 - frac) + samples[hi] * frac
    }).collect()
}

/// Root-mean-square amplitude of a sample slice.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Write mono f32 samples to a temporary 16-bit PCM WAV file.
pub fn write_wav(samples: &[f32], sample_rate: u32) -> Result<tempfile::NamedTempFile> {
    let tmp = tempfile::Builder::new().suffix(".wav").tempfile()
        .context("creating temp WAV file")?;
    let spec = hound::WavSpec {
        channels: 1, sample_rate, bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(tmp.path(), spec)
        .context("creating WAV writer")?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .context("writing WAV sample")?;
    }
    writer.finalize().context("finalizing WAV")?;
    Ok(tmp)
}

/// Voice-activity-detection recording.
///
/// Records until one of:
/// - `silence_timeout` seconds of continuous silence after at least
///   `min_seconds` of audio has been captured, OR
/// - `max_seconds` hard cap is reached.
///
/// Polls the shared `audio_buf` (filled by the cpal stream) at a fixed tick.
/// Returns the accumulated samples at the device's native sample rate.
pub async fn record_vad(
    audio_buf: &Arc<Mutex<Vec<f32>>>,
    native_sr: u32,
    silence_threshold: f32,
    silence_timeout: f32,
    min_seconds: f32,
    max_seconds: f32,
) -> Vec<f32> {
    let tick              = Duration::from_millis(50);
    let _silence_samples  = (native_sr as f32 * silence_timeout) as usize; // reserved for future use
    let min_samples       = (native_sr as f32 * min_seconds) as usize;
    let max_samples       = (native_sr as f32 * max_seconds) as usize;

    let mut captured: Vec<f32> = Vec::new();
    let mut silent_since: Option<Instant> = None;

    // Clear stale audio before we start
    audio_buf.lock().unwrap().clear();

    loop {
        tokio::time::sleep(tick).await;

        let new: Vec<f32> = audio_buf.lock().unwrap().drain(..).collect();
        if new.is_empty() { continue; }

        captured.extend_from_slice(&new);

        // Check silence on the latest chunk
        let chunk_rms = rms(&new);
        let is_silent = chunk_rms < silence_threshold;

        if is_silent {
            if silent_since.is_none() {
                silent_since = Some(Instant::now());
            }
        } else {
            silent_since = None; // reset on any speech
        }

        let total = captured.len();

        // Hard cap
        if total >= max_samples {
            debug!("VAD: max duration reached ({max_seconds}s)");
            break;
        }

        // End-of-speech: minimum captured AND silence timeout elapsed
        if total >= min_samples {
            if let Some(t) = silent_since {
                if t.elapsed().as_secs_f32() >= silence_timeout {
                    debug!("VAD: silence timeout after {:.1}s of audio",
                        total as f32 / native_sr as f32);
                    break;
                }
            }
        }
    }

    captured
}
