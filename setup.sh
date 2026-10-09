#!/bin/bash
# Downloads the Piper voices and OpenVoice checkpoint that .gitignore excludes from the repo
# (they're ~950MB combined, over GitHub's size limits). Run once after cloning.
set -e
cd "$(dirname "$0")"

mkdir -p voices
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
declare -A VOICES=(
  ["en_GB-alba-medium"]="en/en_GB/alba/medium"
  ["en_US-amy-medium"]="en/en_US/amy/medium"
  ["en_US-hfc_female-medium"]="en/en_US/hfc_female/medium"
  ["en_US-lessac-medium"]="en/en_US/lessac/medium"
  ["hi_IN-pratham-medium"]="hi/hi_IN/pratham/medium"
  ["hi_IN-priyamvada-medium"]="hi/hi_IN/priyamvada/medium"
  ["hi_IN-rohan-medium"]="hi/hi_IN/rohan/medium"
  ["bn_BD-google-medium"]="bn/bn_BD/google/medium"
  ["mr_IN-google-medium"]="mr/mr_IN/google/medium"
  ["te_IN-padmavathi-medium"]="te/te_IN/padmavathi/medium"
  ["ur_PK-aegis_female-medium"]="ur/ur_PK/aegis_female/medium"
  ["ml_IN-meera-medium"]="ml/ml_IN/meera/medium"
  ["ne_NP-google-medium"]="ne/ne_NP/google/medium"
)
for name in "${!VOICES[@]}"; do
  path="${VOICES[$name]}"
  if [ ! -f "voices/$name.onnx" ]; then
    echo "downloading $name"
    curl -sL -o "voices/$name.onnx" "$BASE/$path/$name.onnx"
    curl -sL -o "voices/$name.onnx.json" "$BASE/$path/$name.onnx.json"
  fi
done

mkdir -p openvoice_ckpt/converter
if [ ! -f openvoice_ckpt/converter/checkpoint.pth ]; then
  echo "downloading OpenVoice V2 converter checkpoint"
  curl -sL -o openvoice_ckpt/converter/checkpoint.pth \
    "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth"
  curl -sL -o openvoice_ckpt/converter/config.json \
    "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json"
fi

echo "done. Now: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
