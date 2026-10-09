# Locally — AI Text-to-Speech & Voice Studio

Free, offline, open-source text-to-speech. Runs entirely on your own machine using
[Piper](https://github.com/OHF-voice/piper1-gpl) for synthesis and
[OpenVoice V2](https://github.com/myshell-ai/OpenVoice) for local voice cloning. No API keys
required for the core app.

## Features
- Text-to-speech in English, Hindi, Bengali, Marathi, Telugu, Urdu, Malayalam, Nepali, Tamil
  (Piper), plus Punjabi, Gujarati, Kannada, Odia (Meta's MMS - CC-BY-NC, see note below)
- Local voice cloning (upload a clip or record from the mic)
- Multi-speaker dialogue builder
- Searchable voice library with previews
- Talking avatar: upload a photo + text, get a local CPU lip-synced video (Wav2Lip ONNX)
- Optional registration + Razorpay payment gate (off by default, see below)

No Piper voice exists for Assamese anywhere yet - `training/train_assamese_piper.ipynb` is a
ready-to-run Colab notebook for training one (fine-tuned from Bengali, multi-speaker).

## Setup
```bash
./setup.sh                                  # downloads voice models (~1GB)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```
Open http://127.0.0.1:8900

Punjabi/Gujarati/Kannada/Odia voices and the OpenVoice cloning engine need `torch` - installed via
`requirements.txt`. If `opencv-python` ever shows up as a dependency for anything added later,
avoid it on older macOS: it has no prebuilt wheel there and triggers a multi-hour cmake source
build. The talking-avatar feature deliberately uses Pillow + a tiny ONNX face detector instead.

## Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `LOCALLY_GATE` | `0` | Set to `1` to enforce the 2-free-use → register → pay limit. Off by default for local use. |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | unset | Required only if `LOCALLY_GATE=1` and you want real payments. |
| `FFMPEG` | `/usr/local/bin/ffmpeg` | Used to normalize uploaded/recorded audio for cloning. |

## Admin
`/admin/users` lists registered users (only populated if the gate is in use) with simple
mark-paid / delete controls.

## Notes
- `voices/*.onnx`, `openvoice_ckpt/`, `users.json`, and `clones/` are gitignored — the first two
  are large binaries (`setup.sh` re-downloads them), the last two are private per-install data.
- Email sending isn't wired up yet; verification codes are printed to the server log.
