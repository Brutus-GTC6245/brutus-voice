# brutus-voice

A Rust CLI voice loop: microphone → Whisper (STT) → LLM (chat) → TTS, running in a continuous loop.

```
mic ──▶ cpal ──▶ WAV ──▶ Whisper API ──▶ transcript
                                               │
                                     LLM chat completions
                                               │
                                          TTS command
```

## Dependencies

### System (macOS)

cpal uses CoreAudio on macOS — no extra system libraries are needed.

If you want to use a Whisper server locally, options include:
- [whisper.cpp server](https://github.com/ggerganov/whisper.cpp) — `./server -m models/ggml-base.en.bin`
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server)

For TTS the default config uses macOS `say`. Any command that accepts text as its final argument works.

### Rust

Requires Rust 1.75+ (edition 2024). Install via [rustup](https://rustup.rs).

## Setup

```bash
# Clone / enter the repo
cd brutus-voice

# Copy and edit config
cp config.json.example config.json
$EDITOR config.json

# Build
cargo build --release
```

## Config (`config.json`)

| Field | Description | Default |
|---|---|---|
| `api_url` | Base URL of OpenAI-compatible chat API | — |
| `api_key` | Bearer token for the chat API | — |
| `whisper_url` | URL of Whisper transcription endpoint | — |
| `tts_command` | Array: program + args; text is appended as last arg | — |
| `record_seconds` | Seconds to record per turn | `6` |
| `silence_threshold` | RMS level below which audio is skipped | `0.01` |

Example:

```json
{
  "api_url": "http://localhost:3000/v1",
  "api_key": "sk-...",
  "whisper_url": "http://localhost:9000/asr",
  "tts_command": ["say", "-v", "Samantha", "-r", "185"],
  "record_seconds": 6,
  "silence_threshold": 0.01
}
```

## Usage

```bash
# Continuous loop (default)
./target/release/brutus-voice

# Custom config path
./target/release/brutus-voice --config /path/to/config.json

# Single turn and exit
./target/release/brutus-voice --once

# Verbose/debug logging
./target/release/brutus-voice --verbose

# Override log level via env
RUST_LOG=debug ./target/release/brutus-voice
```

## How it works

1. **Record** — `cpal` captures from the default input device for `record_seconds`. Multichannel audio is downmixed to mono.
2. **Silence detection** — RMS amplitude is computed; if below `silence_threshold` the turn is skipped.
3. **WAV** — samples are written to a temp file (16-bit PCM, mono) via `hound`.
4. **Transcription** — the WAV is POST'd as multipart to `whisper_url`. The response `{"text": "..."}` is parsed.
5. **Chat** — the transcript is sent to `api_url/chat/completions` with a Bearer token.
6. **TTS** — the reply text is passed as the final argument to `tts_command`.
7. **Loop** — repeats from step 1 unless `--once` was passed.

Errors in transcription or chat are logged and the loop continues (in loop mode).
