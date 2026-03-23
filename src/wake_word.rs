use anyhow::{bail, Context, Result};
use ort::{session::Session, value::Tensor};
use std::{collections::VecDeque, path::Path};
use tracing::debug;

// ── Pipeline constants ────────────────────────────────────────────────────────
/// Audio chunk size fed to the melspectrogram model (80 ms at 16 kHz)
const CHUNK_SAMPLES: usize = 1280;
/// Mel frames required by the embedding model
const EMB_WINDOW: usize = 76;
/// Embedding frames required by the wake-word model
const WAKE_WINDOW: usize = 16;
/// Embedding vector width
const EMB_DIM: usize = 96;

/// Three-stage ONNX wake-word detector (openWakeWord pipeline):
///
/// ```
/// raw audio (1280 samples / 80 ms)
///   → melspectrogram.onnx   → mel features  [time × mel_cols]
///   → embedding_model.onnx  → 96-dim vector per 76-frame window
///   → hey_jarvis_v0.1.onnx  → score ∈ [0, 1]
/// ```
pub struct WakeDetector {
    mel_session:  Session,
    emb_session:  Session,
    wake_session: Session,
    // Rolling buffer of mel feature rows (each row = mel_cols floats)
    mel_buf:  VecDeque<f32>,
    mel_rows: usize,
    mel_cols: usize, // set after first inference (model-determined)
    // Rolling buffer of embedding vectors (each = EMB_DIM floats)
    emb_buf:  VecDeque<f32>,
    emb_rows: usize,
    // Leftover raw samples that don't fill a full CHUNK_SAMPLES block
    remainder: Vec<f32>,
}

impl WakeDetector {
    /// Load all three ONNX models from `models_dir`.
    pub fn new(models_dir: &Path) -> Result<Self> {
        let mel_path  = models_dir.join("melspectrogram.onnx");
        let emb_path  = models_dir.join("embedding_model.onnx");
        let wake_path = models_dir.join("hey_jarvis_v0.1.onnx");

        for p in [&mel_path, &emb_path, &wake_path] {
            if !p.exists() {
                bail!("wake-word model not found: {}", p.display());
            }
        }

        Ok(Self {
            mel_session:  Session::builder()?.commit_from_file(&mel_path)
                .context("loading melspectrogram model")?,
            emb_session:  Session::builder()?.commit_from_file(&emb_path)
                .context("loading embedding model")?,
            wake_session: Session::builder()?.commit_from_file(&wake_path)
                .context("loading wake-word model")?,
            mel_buf:   VecDeque::new(),
            mel_rows:  0,
            mel_cols:  0,
            emb_buf:   VecDeque::new(),
            emb_rows:  0,
            remainder: Vec::new(),
        })
    }

    /// Feed raw 16 kHz mono samples. Returns the latest wake-word score
    /// whenever a full prediction is produced, or `None` if more audio is needed.
    pub fn process(&mut self, audio: &[f32]) -> Result<Option<f32>> {
        // Prepend any leftover from the previous call
        let mut buf = self.remainder.clone();
        buf.extend_from_slice(audio);

        let n_chunks      = buf.len() / CHUNK_SAMPLES;
        self.remainder    = buf[n_chunks * CHUNK_SAMPLES..].to_vec();

        let mut last_score: Option<f32> = None;

        for i in 0..n_chunks {
            let chunk = &buf[i * CHUNK_SAMPLES..(i + 1) * CHUNK_SAMPLES];
            last_score = self.process_chunk(chunk)?.or(last_score);
        }

        Ok(last_score)
    }

    // ── Internal: one 80 ms chunk through the full pipeline ──────────────────

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<Option<f32>> {
        // Stage 1 — melspectrogram
        let input = Tensor::<f32>::from_array(
            (vec![1i64, CHUNK_SAMPLES as i64], chunk.to_vec()))?;
        let mel_out = self.mel_session.run(ort::inputs!["input" => input])?;
        let (mel_shape, mel_data) = mel_out[0].try_extract_tensor::<f32>()?;

        let time_steps = mel_shape[0] as usize;
        let cols       = mel_shape[mel_shape.len() - 1] as usize;
        if self.mel_cols == 0 { self.mel_cols = cols; }

        let stride = mel_data.len() / time_steps;
        for t in 0..time_steps {
            let row = &mel_data[t * stride..(t + 1) * stride];
            self.mel_buf.extend(&row[row.len() - self.mel_cols..]);
            self.mel_rows += 1;
        }
        // Keep ring buffer bounded
        while self.mel_rows > EMB_WINDOW + 16 {
            for _ in 0..self.mel_cols { self.mel_buf.pop_front(); }
            self.mel_rows -= 1;
        }

        if self.mel_rows < EMB_WINDOW {
            return Ok(None);
        }

        // Stage 2 — speech embedding
        let mel_start = (self.mel_rows - EMB_WINDOW) * self.mel_cols;
        let mel_slice: Vec<f32> = self.mel_buf
            .range(mel_start..mel_start + EMB_WINDOW * self.mel_cols)
            .copied()
            .collect();
        let emb_in = Tensor::<f32>::from_array(
            (vec![1i64, EMB_WINDOW as i64, self.mel_cols as i64, 1i64], mel_slice))?;
        let emb_out = self.emb_session.run(ort::inputs!["input_1" => emb_in])?;
        let (_emb_shape, emb_data) = emb_out[0].try_extract_tensor::<f32>()?;

        self.emb_buf.extend(emb_data.iter().copied());
        self.emb_rows += 1;
        while self.emb_rows > WAKE_WINDOW + 8 {
            for _ in 0..EMB_DIM { self.emb_buf.pop_front(); }
            self.emb_rows -= 1;
        }

        if self.emb_rows < WAKE_WINDOW {
            return Ok(None);
        }

        // Stage 3 — wake-word classifier
        let emb_start = (self.emb_rows - WAKE_WINDOW) * EMB_DIM;
        let wake_slice: Vec<f32> = self.emb_buf
            .range(emb_start..emb_start + WAKE_WINDOW * EMB_DIM)
            .copied()
            .collect();
        let wake_in = Tensor::<f32>::from_array(
            (vec![1i64, WAKE_WINDOW as i64, EMB_DIM as i64], wake_slice))?;
        let wake_out = self.wake_session.run(ort::inputs!["x.1" => wake_in])?;
        let (_ws, score_data) = wake_out[0].try_extract_tensor::<f32>()?;
        let score = score_data.first().copied().unwrap_or(0.0);

        if score > 0.01 { debug!("wake score: {score:.3}"); }
        Ok(Some(score))
    }
}
