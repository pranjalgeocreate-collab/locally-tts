# Locally — AI Text-to-Speech & Voice Studio

Free, offline, open-source text-to-speech. Runs entirely on your own machine using
[Piper](https://github.com/OHF-voice/piper1-gpl) for synthesis and
[OpenVoice V2](https://github.com/myshell-ai/OpenVoice) for local voice cloning. No API keys
required for the core app.

## Features
- Text-to-speech in English, Hindi, Bengali, Marathi, Telugu, Urdu, Malayalam, Nepali
- Local voice cloning (upload a clip or record from the mic)
- Multi-speaker dialogue builder
- Searchable voice library with previews
- Optional registration + Razorpay payment gate (off by default, see below)

## Setup
```bash
./setup.sh                                  # downloads voice models (~950MB)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```
Open http://127.0.0.1:8900

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
