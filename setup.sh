#!/bin/bash
# Downloads the model files that .gitignore excludes from the repo (large binaries, some over
# GitHub's size limits). Run once after cloning. MMS voices (Punjabi/Gujarati/Kannada/Odia) aren't
# listed here - they're fetched automatically by transformers on first use and cached by Hugging Face.
set -e
cd "$(dirname "$0")"

mkdir -p voices
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
declare -A PIPER_VOICES=(
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
  ["te_IN-maya-medium"]="te/te_IN/maya/medium"
  ["te_IN-venkatesh-medium"]="te/te_IN/venkatesh/medium"
  ["ur_PK-aegis_female-medium"]="ur/ur_PK/aegis_female/medium"
  ["ur_PK-fasih-medium"]="ur/ur_PK/fasih/medium"
  ["ml_IN-meera-medium"]="ml/ml_IN/meera/medium"
  ["ml_IN-arjun-medium"]="ml/ml_IN/arjun/medium"
  ["ne_NP-google-medium"]="ne/ne_NP/google/medium"
  ["ne_NP-chitwan-medium"]="ne/ne_NP/chitwan/medium"
)
for name in "${!PIPER_VOICES[@]}"; do
  path="${PIPER_VOICES[$name]}"
  if [ ! -f "voices/$name.onnx" ]; then
    echo "downloading $name"
    curl -sL -o "voices/$name.onnx" "$PIPER_BASE/$path/$name.onnx"
    curl -sL -o "voices/$name.onnx.json" "$PIPER_BASE/$path/$name.onnx.json"
  fi
done

TAMIL_BASE="https://huggingface.co/Jeyaram-K/piper-tamil-voices/resolve/main"
for name in "ta_IN-HemaLatha-medium" "ta_IN-ValluvarNeural-medium"; do
  if [ ! -f "voices/$name.onnx" ]; then
    echo "downloading $name"
    curl -sL -o "voices/$name.onnx" "$TAMIL_BASE/$name/$name.onnx"
    curl -sL -o "voices/$name.onnx.json" "$TAMIL_BASE/$name/$name.onnx.json"
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

mkdir -p wav2lip_model
if [ ! -f wav2lip_model/wav2lip_gan.onnx ]; then
  echo "downloading Wav2Lip ONNX model"
  curl -sL -o wav2lip_model/wav2lip_gan.onnx \
    "https://huggingface.co/bluefoxcreation/Wav2lip-Onnx/resolve/main/wav2lip_gan.onnx"
fi
if [ ! -f wav2lip_model/face_detector.onnx ]; then
  echo "downloading face detector ONNX model"
  curl -sL -o wav2lip_model/face_detector.onnx \
    "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/main/version-RFB-320.onnx"
fi

echo "done. Now: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
