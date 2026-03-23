#!/bin/bash
# speak.sh <text>
# Wrapper so brutus-voice can use Piper the same way as `say`.
# Piper reads text from stdin; brutus-voice passes it as the last argument.
echo "$*" | /Users/brutus/.openclaw/voice-env/bin/piper \
  --model /Users/brutus/.openclaw/voice-models/ryan-high.onnx \
  --output_file /tmp/brutus-tts.wav \
  2>/dev/null
afplay /tmp/brutus-tts.wav
