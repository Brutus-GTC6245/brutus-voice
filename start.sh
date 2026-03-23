#!/bin/bash
# Start brutus-voice with all required env vars
cd "$(dirname "$0")"

export ORT_DYLIB_PATH=/Users/brutus/.openclaw/voice-env/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.24.3.dylib

# Start whisper server if not already running
if ! curl -s http://127.0.0.1:9000/asr > /dev/null 2>&1; then
  echo "Starting Whisper server..."
  /Users/brutus/.openclaw/voice-env/bin/python3 whisper_server.py \
    --model base.en --port 9000 >> /tmp/whisper-server.log 2>&1 &
  sleep 4
fi

echo "Starting brutus-voice (web UI → http://localhost:8080)"
exec ./target/release/brutus-voice \
  --config config.json \
  --models models \
  --threshold 0.5 \
  "$@"
