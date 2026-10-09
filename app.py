#!/usr/bin/env python3
"""A tiny free, offline, open-source text-to-speech app: type text, pick a voice, hear it.
Wraps Piper (https://github.com/OHF-Voice/piper1-gpl), the same engine Dikkey's offline
voice uses. No API keys, no per-character cost, runs entirely on this machine.
"""
import io
import json
import os
import tempfile
import threading
import time
import uuid
import wave

from flask import Flask, jsonify, request, send_file
from piper import PiperVoice, SynthesisConfig

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voices")
CLONE_DIR = os.path.join(HERE, "clones")
os.makedirs(CLONE_DIR, exist_ok=True)

PIPER_SLOWDOWN = 1.15   # length_scale multiplier: Piper's medium voices default to a rushed cadence

# ---------- accounts: free-use gate -> email verify -> Razorpay payment ----------
USERS_PATH = os.path.join(HERE, "users.json")
_users_lock = threading.Lock()
FREE_USES = 2
# The free-use/payment gate is for when this is actually deployed publicly. Running locally it's off
# by default so it never gets in your own way - set LOCALLY_GATE=1 to turn it back on for testing.
GATE_ENABLED = os.environ.get("LOCALLY_GATE", "0") == "1"
PRICE_PAISE = 34900  # Rs 349
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
CODE_TTL = 600  # seconds


def load_users():
    with _users_lock:
        if not os.path.exists(USERS_PATH):
            return {}
        with open(USERS_PATH) as f:
            return json.load(f)


def save_users(users):
    with _users_lock:
        with open(USERS_PATH, "w") as f:
            json.dump(users, f, indent=2)


def get_or_create_user(users, anon_id):
    if anon_id not in users:
        users[anon_id] = {"anon_id": anon_id, "unique_id": "LC" + uuid.uuid4().hex[:6].upper(),
                           "email": None, "name": None, "state": None, "phone": None,
                           "verified": False, "paid": False,
                           "uses": 0, "pending_code": None, "code_expires": None,
                           "order_id": None, "created": time.time()}
    return users[anon_id]


def account_summary(u):
    return {"unique_id": u.get("unique_id"), "email": u["email"], "name": u.get("name"),
            "state": u.get("state"), "phone": u.get("phone"), "verified": u["verified"], "paid": u["paid"],
            "uses": u["uses"], "free_remaining": max(0, FREE_USES - u["uses"]) if not u["paid"] else None}

app = Flask(__name__)
_cache = {}

# Local voice cloning (OpenVoice V2 tone-color conversion): reshapes Piper's output to match a
# reference clip's timbre, rather than being a separate TTS engine. Loading torch + the converter
# takes ~70s one time, so it's warmed in a background thread rather than blocking the first request.
_converter = None
_converter_lock = threading.Lock()
_tgt_se_cache = {}


def warm_converter():
    global _converter
    from openvoice.api import ToneColorConverter
    conv = ToneColorConverter(os.path.join(HERE, "openvoice_ckpt/converter/config.json"), device="cpu")
    conv.load_ckpt(os.path.join(HERE, "openvoice_ckpt/converter/checkpoint.pth"))
    with _converter_lock:
        _converter = conv
    print("voice cloning engine ready")


def clones():
    out = []
    for cid in sorted(os.listdir(CLONE_DIR)):
        meta_path = os.path.join(CLONE_DIR, cid, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            out.append({"id": cid, "name": meta["name"]})
    return out


def get_tgt_se(clone_id):
    if clone_id not in _tgt_se_cache:
        import torch
        se_path = os.path.join(CLONE_DIR, clone_id, "tgt_se.pt")
        if not os.path.exists(se_path):
            return None
        _tgt_se_cache[clone_id] = torch.load(se_path, map_location="cpu")
    return _tgt_se_cache[clone_id]


# Punjabi, Gujarati, Kannada, Odia: no free local Piper voice exists for any of these (checked
# rhasspy/piper-voices and the open community repos on HF). Meta's MMS-TTS fills the gap - free,
# VITS-based like Piper, runs fully local with no API/key/cost. License is CC-BY-NC 4.0
# (non-commercial) - fine for this app as long as GATE_ENABLED stays off; flag this again if the
# payment gate is ever turned on.
MMS_MODELS = {
    "pa_IN-mms-medium": "facebook/mms-tts-pan",
    "gu_IN-mms-medium": "facebook/mms-tts-guj",
    "kn_IN-mms-medium": "facebook/mms-tts-kan",
    "or_IN-mms-medium": "facebook/mms-tts-ory",
}
_mms_cache = {}


def get_mms(name):
    if name not in _mms_cache:
        import torch
        from transformers import VitsModel, AutoTokenizer
        from huggingface_hub import hf_hub_download
        repo = MMS_MODELS[name]
        model = VitsModel.from_pretrained(repo)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        # This torch version's weight_norm uses the new parametrizations.weight.original0/1 naming,
        # but the published checkpoint uses the old weight_g/weight_v naming - remap or most of the
        # flow/posterior_encoder weights silently fail to load (confirmed: 0 missing/unexpected after this).
        path = hf_hub_download(repo, "pytorch_model.bin")
        sd = torch.load(path, map_location="cpu")
        new_sd = {}
        for k, v in sd.items():
            if k.endswith(".weight_g"):
                new_sd[k[:-len("weight_g")] + "parametrizations.weight.original0"] = v
            elif k.endswith(".weight_v"):
                new_sd[k[:-len("weight_v")] + "parametrizations.weight.original1"] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        _mms_cache[name] = (model, tokenizer)
    return _mms_cache[name]


def mms_speak(name, text):
    import torch
    model, tokenizer = get_mms(name)
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].numel() == 0:
        raise ValueError("none of that text is in this voice's script/alphabet")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(model.config.sampling_rate)
        pcm = (waveform.squeeze().clamp(-1, 1).numpy() * 32767).astype("int16")
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf


def voices():
    out = []
    for f in sorted(os.listdir(VOICE_DIR)):
        if f.endswith(".onnx") and os.path.exists(os.path.join(VOICE_DIR, f + ".json")):
            name = f[:-5]
            lang, speaker = name.split("-")[0], name.split("-")[1]
            out.append({"name": name, "lang": lang, "speaker": speaker.replace("_", " ").title()})
    for name in MMS_MODELS:
        lang = name.split("-")[0]
        out.append({"name": name, "lang": lang, "speaker": "MMS (Meta, free, non-commercial)"})
    return out


def get_voice(name):
    if name not in _cache:
        _cache[name] = PiperVoice.load(os.path.join(VOICE_DIR, name + ".onnx"))
    return _cache[name]


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/admin/users")
def admin_users():
    users = load_users()
    rows = sorted(users.values(), key=lambda u: u.get("created", 0), reverse=True)
    import datetime
    def fmt(ts):
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
    body = "".join(f"""
      <tr>
        <td><code style="font-size:11px">{u.get('unique_id','—')}</code></td>
        <td>{(u.get('name') or '—')}</td>
        <td>{(u.get('email') or '(not registered)')}</td>
        <td>{(u.get('phone') or '—')}</td>
        <td>{(u.get('state') or '—')}</td>
        <td>{'✓' if u.get('verified') else '—'}</td>
        <td>{'✓ PAID' if u.get('paid') else '—'}</td>
        <td>{u.get('uses', 0)}</td>
        <td>{fmt(u.get('created'))}</td>
        <td>
          <form method="post" action="/admin/users/{u.get('anon_id')}/toggle-paid" style="display:inline">
            <button>{'Unmark paid' if u.get('paid') else 'Mark paid'}</button>
          </form>
          <form method="post" action="/admin/users/{u.get('anon_id')}/delete" style="display:inline">
            <button style="color:#b3261e">Delete</button>
          </form>
        </td>
      </tr>""" for u in rows)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Locally - Users</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; margin: 24px; color: #201C3D; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #DAD7EA; font-size: 14px; }}
      th {{ color: #5B577A; font-weight: 600; }}
      button {{ font-size: 12px; padding: 4px 8px; border-radius: 6px; border: 1px solid #DAD7EA; background: #F6F5FB; cursor: pointer; }}
    </style></head><body>
    <h2>Locally &mdash; registered users ({len(rows)})</h2>
    <p style="color:#5B577A;font-size:13px">Free limit: {FREE_USES} generations. Price: ₹{PRICE_PAISE // 100}.</p>
    <table><tr><th>User ID</th><th>Name</th><th>Email</th><th>Phone</th><th>State</th><th>Verified</th><th>Paid</th><th>Uses</th><th>Created</th><th>Actions</th></tr>
    {body}
    </table></body></html>"""


@app.post("/admin/users/<anon_id>/toggle-paid")
def admin_toggle_paid(anon_id):
    users = load_users()
    if anon_id in users:
        users[anon_id]["paid"] = not users[anon_id]["paid"]
        save_users(users)
    return admin_redirect()


@app.post("/admin/users/<anon_id>/delete")
def admin_delete_user(anon_id):
    users = load_users()
    users.pop(anon_id, None)
    save_users(users)
    return admin_redirect()


def admin_redirect():
    from flask import redirect
    return redirect("/admin/users")


@app.get("/api/voices")
def api_voices():
    return jsonify(voices=voices())


@app.get("/api/clones")
def api_clones():
    return jsonify(clones=clones(), ready=_converter is not None)


FFMPEG = os.environ.get("FFMPEG", "/usr/local/bin/ffmpeg")


@app.post("/api/clone")
def api_clone():
    if _converter is None:
        return jsonify(error="voice cloning engine is still warming up, try again in a bit"), 503
    name = (request.form.get("name") or "").strip()[:60]
    audio_file = request.files.get("audio")
    if not name or not audio_file:
        return jsonify(error="need a name and an audio clip"), 400
    cid = uuid.uuid4().hex[:12]
    cdir = os.path.join(CLONE_DIR, cid)
    os.makedirs(cdir, exist_ok=True)
    raw_path = os.path.join(cdir, "raw" + (os.path.splitext(audio_file.filename or "")[1] or ".webm"))
    audio_file.save(raw_path)

    # Browser recordings come in as webm/opus (and uploads could be mp3/m4a/etc) - soundfile can't
    # read those directly, so normalize everything to a plain WAV with ffmpeg first.
    import subprocess
    ref_path = os.path.join(cdir, "ref.wav")
    try:
        subprocess.run([FFMPEG, "-y", "-i", raw_path, "-ar", "24000", "-ac", "1", ref_path],
                        capture_output=True, check=True, timeout=30)
    except Exception as e:
        shutil_rmtree_quiet(cdir)
        detail = e.stderr.decode(errors="replace")[-300:] if hasattr(e, "stderr") and e.stderr else str(e)
        return jsonify(error=f"couldn't read that audio clip: {detail}"), 400
    os.remove(raw_path)

    try:
        tgt_se = _converter.extract_se(ref_path)
    except Exception as e:
        shutil_rmtree_quiet(cdir)
        return jsonify(error=f"couldn't process that clip: {e}"), 400
    import torch
    torch.save(tgt_se, os.path.join(cdir, "tgt_se.pt"))
    with open(os.path.join(cdir, "meta.json"), "w") as f:
        json.dump({"name": name, "created": time.time()}, f)
    return jsonify(id=cid, name=name)


def shutil_rmtree_quiet(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


@app.delete("/api/clone/<cid>")
def api_delete_clone(cid):
    cdir = os.path.join(CLONE_DIR, cid)
    if not os.path.isdir(cdir) or not os.path.exists(os.path.join(cdir, "meta.json")):
        return jsonify(error="unknown cloned voice"), 404
    shutil_rmtree_quiet(cdir)
    _tgt_se_cache.pop(cid, None)
    return jsonify(ok=True)


@app.get("/api/account")
def api_account():
    anon_id = request.args.get("anon") or ""
    if not anon_id:
        return jsonify(error="missing anon id"), 400
    users = load_users()
    u = get_or_create_user(users, anon_id)
    save_users(users)
    return jsonify(gate_enabled=GATE_ENABLED, **account_summary(u))


@app.post("/api/register")
def api_register():
    body = request.get_json(silent=True) or {}
    anon_id = body.get("anon_id") or ""
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()[:80]
    state = (body.get("state") or "").strip()[:40]
    phone = (body.get("phone") or "").strip()[:20]
    if not anon_id or "@" not in email:
        return jsonify(error="need a valid email"), 400
    if not name:
        return jsonify(error="name is required"), 400
    import random
    code = f"{random.randint(0, 999999):06d}"
    users = load_users()
    u = get_or_create_user(users, anon_id)
    u["email"] = email
    u["name"] = name
    u["state"] = state
    u["phone"] = phone
    u["verified"] = False
    u["pending_code"] = code
    u["code_expires"] = time.time() + CODE_TTL
    save_users(users)
    # No email service wired up yet (waiting on hosting/SMTP decision) - print the code here so the
    # flow is fully testable locally. Swap this for a real send once that's sorted.
    print(f"[locally] verification code for {email}: {code} (expires in {CODE_TTL}s)", flush=True)
    return jsonify(ok=True, dev_note="email sending isn't wired up yet - check the server terminal for your code")


@app.post("/api/verify-email")
def api_verify_email():
    body = request.get_json(silent=True) or {}
    anon_id = body.get("anon_id") or ""
    code = (body.get("code") or "").strip()
    users = load_users()
    u = users.get(anon_id)
    if not u or not u.get("pending_code"):
        return jsonify(error="no pending verification for this session"), 400
    if time.time() > (u.get("code_expires") or 0):
        return jsonify(error="code expired, request a new one"), 400
    if code != u["pending_code"]:
        return jsonify(error="wrong code"), 400
    u["verified"] = True
    u["pending_code"] = None
    u["code_expires"] = None
    save_users(users)
    return jsonify(**account_summary(u))


@app.post("/api/create-order")
def api_create_order():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return jsonify(error="payment gateway isn't configured yet"), 503
    body = request.get_json(silent=True) or {}
    anon_id = body.get("anon_id") or ""
    users = load_users()
    u = users.get(anon_id)
    if not u or not u["verified"]:
        return jsonify(error="verify your email first"), 400
    import razorpay
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    order = client.order.create({"amount": PRICE_PAISE, "currency": "INR", "receipt": anon_id,
                                  "notes": {"email": u["email"]}})
    u["order_id"] = order["id"]
    save_users(users)
    return jsonify(order_id=order["id"], amount=PRICE_PAISE, currency="INR", key_id=RAZORPAY_KEY_ID)


@app.post("/api/verify-payment")
def api_verify_payment():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return jsonify(error="payment gateway isn't configured yet"), 503
    body = request.get_json(silent=True) or {}
    anon_id = body.get("anon_id") or ""
    users = load_users()
    u = users.get(anon_id)
    if not u or u.get("order_id") != body.get("razorpay_order_id"):
        return jsonify(error="order mismatch"), 400
    import razorpay
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    try:
        client.utility.verify_payment_signature({
            "razorpay_order_id": body.get("razorpay_order_id"),
            "razorpay_payment_id": body.get("razorpay_payment_id"),
            "razorpay_signature": body.get("razorpay_signature"),
        })
    except razorpay.errors.SignatureVerificationError:
        return jsonify(error="payment signature didn't verify"), 400
    u["paid"] = True
    save_users(users)
    return jsonify(**account_summary(u))


@app.post("/api/speak")
def api_speak():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()[:2000]
    name = body.get("voice") or ""
    speed = float(body.get("speed") or 1.0)
    clone_id = body.get("clone_id") or ""
    anon_id = body.get("anon_id") or ""
    if not text:
        return jsonify(error="type something first"), 400
    if name not in {v["name"] for v in voices()}:
        return jsonify(error="unknown voice"), 400

    users = load_users()
    u = get_or_create_user(users, anon_id) if anon_id else None
    if GATE_ENABLED and u and not u["paid"] and u["uses"] >= FREE_USES:
        return jsonify(error="free_limit", message=f"You've used your {FREE_USES} free generations. "
                       f"Register and pay ₹{PRICE_PAISE // 100} to keep going."), 402

    if name in MMS_MODELS:
        try:
            buf = mms_speak(name, text)
        except Exception as e:
            return jsonify(error=f"MMS synthesis failed: {e}"), 502
    else:
        buf = io.BytesIO()
        # Piper's medium models default to a rushed cadence; PIPER_SLOWDOWN pulls the baseline back to natural speech.
        length_scale = max(0.5, min(2.0, PIPER_SLOWDOWN / speed))
        with wave.open(buf, "wb") as wf:
            get_voice(name).synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=length_scale))
        buf.seek(0)

    def record_use():
        if u:
            u["uses"] += 1
            save_users(users)

    if not clone_id:
        record_use()
        return send_file(buf, mimetype="audio/wav", download_name="speech.wav")

    if _converter is None:
        return jsonify(error="voice cloning engine is still warming up, try again in a bit"), 503
    tgt_se = get_tgt_se(clone_id)
    if tgt_se is None:
        return jsonify(error="unknown cloned voice"), 400

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.wav")
        out_path = os.path.join(tmp, "out.wav")
        with open(src_path, "wb") as f:
            f.write(buf.getvalue())
        src_se = _converter.extract_se(src_path)
        _converter.convert(src_path, src_se, tgt_se, output_path=out_path)
        with open(out_path, "rb") as f:
            out_bytes = f.read()
    record_use()
    return send_file(io.BytesIO(out_bytes), mimetype="audio/wav", download_name="speech.wav")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Locally — Clean AI Text-to-Speech Studio</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="icon" href="/static/logo.png">
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: { extend: {
      colors: {
        claude: { bg:'#F6F5FB', card:'#EEEDF6', border:'#DAD7EA', text:'#201C3D', subtext:'#5B577A',
                   darkBg:'#121026', darkCard:'#1C1A33', darkBorder:'#2E2B4A', darkText:'#EAE8F5', darkSubtext:'#A6A3BD' },
        terracotta: { 50:'#FDF5E9', 100:'#FBE8C9', 200:'#F5D08C', 500:'#DD9029', 600:'#C27D1E', 700:'#9C6416' }
      },
      fontFamily: { serif: ['"Newsreader"','serif'], sans: ['"Plus Jakarta Sans"','sans-serif'] }
    } }
  }
</script>
<style>
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(180, 172, 160, 0.4); border-radius: 999px; }

  .agent-orb { display: inline-flex; align-items: center; justify-content: center; gap: 3px;
               width: 30px; height: 30px; border-radius: 999px; background: #201C3D; flex-shrink: 0; }
  .dark .agent-orb { background: #DD9029; }
  .agent-bar { display: block; width: 3px; height: 8px; border-radius: 2px; background: #DD9029;
               animation: agentIdle 1.8s ease-in-out infinite; }
  .dark .agent-bar { background: #201C3D; }
  .agent-bar:nth-child(2) { animation-delay: 0.2s; }
  .agent-bar:nth-child(3) { animation-delay: 0.4s; }
  @keyframes agentIdle { 0%, 100% { height: 6px; opacity: .6; } 50% { height: 12px; opacity: 1; } }
  @keyframes agentWorking { 0%, 100% { height: 5px; } 25% { height: 18px; } 50% { height: 9px; } 75% { height: 20px; } }
  #agent.working .agent-bar { animation: agentWorking 0.5s ease-in-out infinite; }
  #agent.working .agent-orb { box-shadow: 0 0 0 4px rgba(221,144,41,0.25); }
</style>
</head>
<body class="font-sans bg-claude-bg dark:bg-claude-darkBg text-claude-text dark:text-claude-darkText min-h-screen transition-colors">
<div class="w-[80%] mx-auto px-4 py-8">

  <div class="flex items-center justify-between mb-2">
    <div class="flex items-center gap-3">
      <img src="/static/logo.png" alt="Locally" class="w-10 h-10 flex-shrink-0 rounded-xl">
      <div>
        <div class="font-serif text-2xl font-semibold leading-none">Locally</div>
        <div class="text-xs text-claude-subtext dark:text-claude-darkSubtext mt-0.5">AI Text-to-Speech &amp; Voice Studio</div>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <span id="accountBadge" class="hidden text-xs px-2.5 py-1 rounded-full bg-terracotta-100 text-terracotta-700"></span>
      <button id="loginBtn" class="px-3.5 py-2 rounded-lg border border-claude-border dark:border-claude-darkBorder hover:bg-claude-card dark:hover:bg-claude-darkCard text-sm font-medium transition-colors">Sign in</button>
      <button id="themeToggle" class="p-2.5 rounded-lg border border-claude-border dark:border-claude-darkBorder hover:bg-claude-card dark:hover:bg-claude-darkCard text-sm transition-colors">
        <span id="themeIcon">&#9789;</span>
      </button>
    </div>
  </div>
  <p class="font-serif italic text-claude-subtext dark:text-claude-darkSubtext mb-6">Your voice, native &amp; clear.</p>

  <div class="flex gap-1 mb-5 border-b border-claude-border dark:border-claude-darkBorder overflow-x-auto" id="tabbar">
    <button data-tab="speak" class="tabbtn px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors hover:text-claude-text dark:hover:text-claude-darkText">Speak</button>
    <button data-tab="clone" class="tabbtn px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors hover:text-claude-text dark:hover:text-claude-darkText">Clone a Voice</button>
    <button data-tab="dialogue" class="tabbtn px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors hover:text-claude-text dark:hover:text-claude-darkText">Dialogue</button>
    <button data-tab="library" class="tabbtn px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors hover:text-claude-text dark:hover:text-claude-darkText">Library</button>
  </div>

  <div id="panel-speak" class="tabpanel space-y-4">
    <h2 class="font-serif text-3xl font-medium mb-1">Hi&hellip; What should we making today?</h2>
    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow space-y-4">
      <textarea id="text" rows="4" class="w-full rounded-xl border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2.5 text-[15px] resize-y focus:outline-none focus:ring-2 focus:ring-terracotta-500/40">I am Locally. You can use it from your mobile, no API, no key, use unlimited.</textarea>
      <div class="flex justify-end -mt-2">
        <span id="charCount" class="text-xs text-claude-subtext dark:text-claude-darkSubtext">0 / 2000</span>
      </div>

      <div class="flex flex-nowrap gap-3 items-center text-sm overflow-x-auto">
        <label class="flex items-center gap-2 text-claude-subtext dark:text-claude-darkSubtext flex-shrink-0">Language
          <select id="lang" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1.5"></select>
        </label>
        <select id="voice" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1.5 flex-shrink-0"></select>
        <label class="flex items-center gap-2 text-claude-subtext dark:text-claude-darkSubtext flex-shrink-0">Speed
          <input type="range" id="speed" min="0.5" max="1.8" step="0.1" value="1.0" class="accent-terracotta-500">
          <span id="speedval" class="w-9 inline-block">1.0x</span>
        </label>
        <label class="flex items-center gap-2 text-claude-subtext dark:text-claude-darkSubtext flex-shrink-0">Cloned voice
          <select id="clone" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1.5"><option value="">None (use voice above)</option></select>
        </label>
        <span id="cloneStatus" class="text-xs text-claude-subtext dark:text-claude-darkSubtext flex-shrink-0"></span>
      </div>

      <div id="agent" class="flex items-center gap-2 flex-shrink-0">
        <span class="agent-orb"><span class="agent-bar"></span><span class="agent-bar"></span><span class="agent-bar"></span></span>
        <span id="agentLabel" class="text-xs text-claude-subtext dark:text-claude-darkSubtext">Locally agent idle</span>
      </div>

      <canvas id="wave" height="56" class="w-full rounded-lg bg-claude-bg dark:bg-claude-darkBg"></canvas>
      <audio id="audio" controls class="w-full hidden"></audio>

      <div class="flex items-center gap-3">
        <button id="play" class="bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-5 py-2.5 rounded-xl text-sm">&#9654; Play</button>
        <button id="dl" class="border border-terracotta-500 text-terracotta-600 hover:bg-terracotta-50 dark:hover:bg-terracotta-500/10 transition-colors px-5 py-2.5 rounded-xl text-sm">&#8595; Download</button>
        <span id="usageNote" class="text-xs text-claude-subtext dark:text-claude-darkSubtext"></span>
        <span id="err" class="text-red-600 text-sm"></span>
      </div>
    </div>

    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow">
      <div class="text-sm font-semibold mb-3">History</div>
      <div id="history" class="space-y-2 text-sm text-claude-subtext dark:text-claude-darkSubtext">No clips yet this session.</div>
    </div>
  </div>

  <div id="panel-clone" class="tabpanel hidden space-y-4">
    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow space-y-4">
      <div class="font-serif text-lg">Clone a voice</div>
      <p class="text-sm text-claude-subtext dark:text-claude-darkSubtext">Upload a short, clean clip (5-20s, one speaker, little background noise) or record one with your mic. Runs locally.</p>
      <div class="flex flex-wrap gap-3 items-center">
        <input id="cloneName" placeholder="Name this voice" class="flex-1 min-w-[160px] rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
        <input type="file" id="cloneFile" accept="audio/*" class="text-sm">
        <span class="text-xs text-claude-subtext dark:text-claude-darkSubtext">or</span>
        <button id="cloneRecord" class="border border-terracotta-500 text-terracotta-600 hover:bg-terracotta-50 dark:hover:bg-terracotta-500/10 transition-colors px-3 py-2 rounded-lg text-sm">&#127908; Record</button>
        <span id="recordTime" class="text-xs text-claude-subtext dark:text-claude-darkSubtext"></span>
      </div>
      <canvas id="recLevel" height="30" class="w-full rounded-lg bg-claude-bg dark:bg-claude-darkBg hidden"></canvas>
      <audio id="recordPreview" controls class="w-full hidden"></audio>
      <div class="flex items-center gap-3">
        <button id="cloneAdd" class="bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-5 py-2.5 rounded-xl text-sm">Add voice</button>
        <span id="cloneErr" class="text-red-600 text-sm"></span>
      </div>
    </div>
    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow">
      <div class="text-sm font-semibold mb-3">Your cloned voices</div>
      <div id="cloneListUI" class="space-y-2 text-sm">None yet.</div>
    </div>
  </div>

  <div id="panel-dialogue" class="tabpanel hidden space-y-4">
    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow space-y-3">
      <div class="font-serif text-lg">Multi-speaker dialogue</div>
      <p class="text-sm text-claude-subtext dark:text-claude-darkSubtext">Build a short script with a different voice per speaker, then play it start to finish.</p>
      <div id="turns" class="space-y-3"></div>
      <button id="addTurn" class="border border-claude-border dark:border-claude-darkBorder px-3 py-2 rounded-lg text-sm">+ Add turn</button>
    </div>
    <div class="flex items-center gap-3">
      <button id="playDialogue" class="bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-5 py-2.5 rounded-xl text-sm">&#9654; Play dialogue</button>
      <span id="dialogueStatus" class="text-sm text-claude-subtext dark:text-claude-darkSubtext"></span>
    </div>
    <audio id="dialogueAudio" class="hidden"></audio>
  </div>

  <div id="panel-library" class="tabpanel hidden space-y-4">
    <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow space-y-3">
      <input id="librarySearch" placeholder="Search voices by name or language…" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
      <div id="libraryList" class="divide-y divide-claude-border dark:divide-claude-darkBorder"></div>
    </div>
    <div class="text-sm text-claude-subtext dark:text-claude-darkSubtext">
      Don't see your language? No free local Piper voice exists yet for Assamese —
      <a href="https://colab.research.google.com/github/pranjalgeocreate-collab/locally-tts/blob/main/training/train_assamese_piper.ipynb" target="_blank" class="text-terracotta-600 underline">train one yourself on Colab (free)</a>.
      Punjabi, Gujarati, Kannada and Odia use Meta's MMS model instead of Piper — free, local, but non-commercial licensed.
    </div>
  </div>

</div>

<div id="gateOverlay" class="hidden fixed inset-0 bg-black/40 flex items-center justify-center p-4 z-50">
  <div class="bg-claude-card dark:bg-claude-darkCard border border-claude-border dark:border-claude-darkBorder rounded-2xl p-6 max-w-sm w-full space-y-4">
    <div class="font-serif text-xl">You've used your free generations</div>
    <p id="gateMsg" class="text-sm text-claude-subtext dark:text-claude-darkSubtext">Register your email, verify it, then unlock unlimited use for &#8377;349.</p>

    <div id="gateStepEmail" class="space-y-2">
      <input id="gateName" placeholder="Full name" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
      <input id="gateEmail" type="email" placeholder="you@email.com" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
      <input id="gatePhone" type="tel" placeholder="Phone number" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
      <select id="gateState" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
        <option value="">State (optional)</option>
        <option>Andhra Pradesh</option><option>Arunachal Pradesh</option><option>Assam</option><option>Bihar</option>
        <option>Chhattisgarh</option><option>Goa</option><option>Gujarat</option><option>Haryana</option>
        <option>Himachal Pradesh</option><option>Jharkhand</option><option>Karnataka</option><option>Kerala</option>
        <option>Madhya Pradesh</option><option>Maharashtra</option><option>Manipur</option><option>Meghalaya</option>
        <option>Mizoram</option><option>Nagaland</option><option>Odisha</option><option>Punjab</option>
        <option>Rajasthan</option><option>Sikkim</option><option>Tamil Nadu</option><option>Telangana</option>
        <option>Tripura</option><option>Uttar Pradesh</option><option>Uttarakhand</option><option>West Bengal</option>
        <option>Delhi</option><option>Jammu and Kashmir</option><option>Ladakh</option><option>Puducherry</option>
        <option>Chandigarh</option><option>Andaman and Nicobar Islands</option><option>Dadra and Nagar Haveli and Daman and Diu</option><option>Lakshadweep</option>
      </select>
      <button id="gateSendCode" class="w-full bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-4 py-2 rounded-xl text-sm">Send verification code</button>
      <span id="gateEmailErr" class="text-red-600 text-xs block"></span>
    </div>

    <div id="gateStepCode" class="hidden space-y-2">
      <p class="text-xs text-claude-subtext dark:text-claude-darkSubtext">Email sending isn't wired up yet &mdash; check the server terminal/log for your code.</p>
      <input id="gateCode" placeholder="6-digit code" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-3 py-2 text-sm">
      <button id="gateVerifyCode" class="w-full bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-4 py-2 rounded-xl text-sm">Verify</button>
      <span id="gateCodeErr" class="text-red-600 text-xs block"></span>
    </div>

    <div id="gateStepPay" class="hidden space-y-2">
      <p class="text-sm">Email verified. Unlock unlimited use for <strong>&#8377;349</strong>.</p>
      <button id="gatePay" class="w-full bg-terracotta-500 hover:bg-terracotta-600 text-white font-medium shadow-sm hover:shadow-md hover:-translate-y-px active:translate-y-0 transition-all px-4 py-2 rounded-xl text-sm">Pay &#8377;349</button>
      <span id="gatePayErr" class="text-red-600 text-xs block"></span>
    </div>

    <button id="gateClose" class="w-full text-xs text-claude-subtext dark:text-claude-darkSubtext">Close</button>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);

// ---------- account / free-use gate ----------
function getAnonId() {
  try {
    let id = localStorage.getItem('locally-anon-id');
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('locally-anon-id', id); }
    return id;
  } catch (e) { return 'anon'; }
}
const ANON_ID = getAnonId();
let account = {paid: false, verified: false, free_remaining: 2};
async function loadAccount() {
  try {
    const r = await fetch('/api/account?anon=' + encodeURIComponent(ANON_ID));
    account = await r.json();
    if (!account.gate_enabled) {
      $('usageNote').textContent = 'Unlimited (local)';
    } else {
      $('usageNote').textContent = account.paid ? 'Unlimited' :
        (account.free_remaining != null ? account.free_remaining + ' free generation' + (account.free_remaining === 1 ? '' : 's') + ' left' : '');
    }
    const badge = $('accountBadge');
    if (account.paid) {
      badge.textContent = '✓ ' + (account.name || account.email || 'Paid');
      badge.classList.remove('hidden');
      $('loginBtn').classList.add('hidden');
    } else if (account.verified) {
      badge.textContent = account.name || account.email;
      badge.classList.remove('hidden');
      $('loginBtn').textContent = 'Unlock full access';
    } else {
      badge.classList.add('hidden');
      $('loginBtn').textContent = 'Sign in';
    }
  } catch (e) {}
}
$('loginBtn').onclick = () => openGate();
function openGate(message) {
  $('gateMsg').textContent = message || 'Register your email, verify it, then unlock unlimited use for ₹349.';
  $('gateOverlay').classList.remove('hidden');
  $('gateStepEmail').classList.toggle('hidden', account.verified);
  $('gateStepCode').classList.add('hidden');
  $('gateStepPay').classList.toggle('hidden', !account.verified);
}
function closeGate() { $('gateOverlay').classList.add('hidden'); }
$('gateClose').onclick = closeGate;
$('gateSendCode').onclick = async () => {
  $('gateEmailErr').textContent = '';
  const name = $('gateName').value.trim();
  const email = $('gateEmail').value.trim();
  const phone = $('gatePhone').value.trim();
  const state = $('gateState').value;
  if (!name) { $('gateEmailErr').textContent = 'enter your name'; return; }
  if (!email.includes('@')) { $('gateEmailErr').textContent = 'enter a valid email'; return; }
  $('gateSendCode').disabled = true;
  try {
    const r = await fetch('/api/register', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({anon_id: ANON_ID, email, name, phone, state})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'failed');
    $('gateStepEmail').classList.add('hidden');
    $('gateStepCode').classList.remove('hidden');
  } catch (e) {
    $('gateEmailErr').textContent = e.message;
  } finally {
    $('gateSendCode').disabled = false;
  }
};
$('gateVerifyCode').onclick = async () => {
  $('gateCodeErr').textContent = '';
  const code = $('gateCode').value.trim();
  $('gateVerifyCode').disabled = true;
  try {
    const r = await fetch('/api/verify-email', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({anon_id: ANON_ID, code})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'failed');
    account = d;
    $('gateStepCode').classList.add('hidden');
    $('gateStepPay').classList.remove('hidden');
  } catch (e) {
    $('gateCodeErr').textContent = e.message;
  } finally {
    $('gateVerifyCode').disabled = false;
  }
};
function loadRazorpayScript() {
  return new Promise((resolve, reject) => {
    if (window.Razorpay) return resolve();
    const s = document.createElement('script');
    s.src = 'https://checkout.razorpay.com/v1/checkout.js';
    s.onload = resolve;
    s.onerror = () => reject(new Error('could not load payment widget'));
    document.head.appendChild(s);
  });
}
$('gatePay').onclick = async () => {
  $('gatePayErr').textContent = '';
  $('gatePay').disabled = true;
  try {
    const r = await fetch('/api/create-order', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({anon_id: ANON_ID})});
    const order = await r.json();
    if (!r.ok) throw new Error(order.error || 'failed');
    await loadRazorpayScript();
    const rp = new Razorpay({
      key: order.key_id, amount: order.amount, currency: order.currency, order_id: order.order_id,
      name: 'Locally', description: 'Unlimited Text-to-Speech',
      handler: async (resp) => {
        const vr = await fetch('/api/verify-payment', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({anon_id: ANON_ID, razorpay_order_id: resp.razorpay_order_id,
            razorpay_payment_id: resp.razorpay_payment_id, razorpay_signature: resp.razorpay_signature})});
        const vd = await vr.json();
        if (vr.ok && vd.paid) { account = vd; closeGate(); }
        else { $('gatePayErr').textContent = 'payment did not verify'; }
      },
      theme: {color: '#DD9029'},
    });
    rp.open();
  } catch (e) {
    $('gatePayErr').textContent = e.message;
  } finally {
    $('gatePay').disabled = false;
  }
};

// ---------- theme ----------
function applyTheme(dark) {
  document.documentElement.classList.toggle('dark', dark);
  $('themeIcon').textContent = dark ? '\\u2600' : '\\u263D';
  try { localStorage.setItem('locally-theme', dark ? 'dark' : 'light'); } catch (e) {}
}
try {
  const saved = localStorage.getItem('locally-theme');
  applyTheme(saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
} catch (e) { applyTheme(false); }
$('themeToggle').onclick = () => applyTheme(!document.documentElement.classList.contains('dark'));

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll('.tabpanel').forEach(p => p.classList.toggle('hidden', p.id !== 'panel-' + name));
  document.querySelectorAll('.tabbtn').forEach(b => {
    const on = b.dataset.tab === name;
    b.classList.toggle('border-terracotta-500', on);
    b.classList.toggle('text-claude-text', on);
    b.classList.toggle('dark:text-claude-darkText', on);
    b.classList.toggle('border-transparent', !on);
    b.classList.toggle('text-claude-subtext', !on);
    b.classList.toggle('dark:text-claude-darkSubtext', !on);
  });
}
document.querySelectorAll('.tabbtn').forEach(b => b.onclick = () => showTab(b.dataset.tab));
showTab('speak');

// ---------- voices ----------
let voiceList = [];
const LANG_NAMES = {
  en: 'English', hi: 'Hindi', mr: 'Marathi', bn: 'Bengali', te: 'Telugu', ur: 'Urdu', ml: 'Malayalam', ne: 'Nepali', ta: 'Tamil', pa: 'Punjabi', gu: 'Gujarati', kn: 'Kannada', or: 'Odia',
  ar: 'Arabic', bg: 'Bulgarian', ca: 'Catalan', cs: 'Czech', cy: 'Welsh', da: 'Danish', de: 'German', el: 'Greek',
  es: 'Spanish', et: 'Estonian', eu: 'Basque', fa: 'Persian', fi: 'Finnish', fr: 'French', he: 'Hebrew',
  hu: 'Hungarian', hy: 'Armenian', id: 'Indonesian', is: 'Icelandic', it: 'Italian', ja: 'Japanese',
  ka: 'Georgian', kk: 'Kazakh', ko: 'Korean', ku: 'Kurdish', lb: 'Luxembourgish', lt: 'Lithuanian',
  lv: 'Latvian', nl: 'Dutch', no: 'Norwegian', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian', ru: 'Russian',
  sk: 'Slovak', sl: 'Slovenian', sq: 'Albanian', sr: 'Serbian', sv: 'Swedish', sw: 'Swahili', th: 'Thai',
  tr: 'Turkish', uk: 'Ukrainian', vi: 'Vietnamese', zh: 'Chinese',
};
const SAMPLE_TEXT = {
  en: 'This is a preview of this voice.', hi: 'यह आवाज़ का एक नमूना है।', mr: 'ही आवाजाची एक झलक आहे.',
  bn: 'এটি এই কণ্ঠের একটি নমুনা।', te: 'ఇది ఈ వాయిస్ యొక్క నమూనా.', ur: 'یہ اس آواز کا نمونہ ہے۔',
  ml: 'ഇത് ഈ ശബ്ദത്തിന്റെ ഒരു സാമ്പിൾ ആണ്.', ne: 'यो यो आवाजको नमूना हो।', ta: 'இது இந்த குரலின் மாதிரி.', pa: 'ਇਹ ਇਸ ਆਵਾਜ਼ ਦਾ ਨਮੂਨਾ ਹੈ।',
  gu: 'આ આ અવાજનો એક નમૂનો છે.', kn: 'ಇದು ಈ ಧ್ವನಿಯ ಒಂದು ಮಾದರಿ.', or: 'ଏହା ଏହି ସ୍ୱରର ଏକ ନମୁନା।',
};
function langsSorted(list) {
  return [...new Set(list.map(v => v.baseLang))].sort((a, b) => (LANG_NAMES[a] || a).localeCompare(LANG_NAMES[b] || b));
}
function populateVoicesForLang(lang) {
  const opt = v => `<option value="${v.name}">${v.speaker}</option>`;
  $('voice').innerHTML = voiceList.filter(v => v.baseLang === lang).map(opt).join('');
}
async function loadVoices() {
  const r = await fetch('/api/voices');
  const d = await r.json();
  voiceList = d.voices.map(v => ({...v, baseLang: v.lang.split('_')[0].toLowerCase()}));
  const langs = langsSorted(voiceList);
  $('lang').innerHTML = langs.map(l => `<option value="${l}">${LANG_NAMES[l] || l.toUpperCase()}</option>`).join('');
  $('lang').value = langs.includes('en') ? 'en' : langs[0];
  populateVoicesForLang($('lang').value);
  renderLibrary();
}
$('lang').addEventListener('change', () => populateVoicesForLang($('lang').value));
$('speed').addEventListener('input', () => $('speedval').textContent = parseFloat($('speed').value).toFixed(1) + 'x');

const TEXT_MAX = 2000;
function updateCharCount() {
  const n = $('text').value.length;
  $('charCount').textContent = n + ' / ' + TEXT_MAX;
  $('charCount').classList.toggle('text-red-600', n > TEXT_MAX);
  $('charCount').classList.toggle('dark:text-claude-darkSubtext', n <= TEXT_MAX);
}
$('text').addEventListener('input', updateCharCount);
updateCharCount();

// ---------- clones ----------
let cloneList = [];
async function loadClones() {
  const r = await fetch('/api/clones');
  const d = await r.json();
  cloneList = d.clones;
  const prev = $('clone').value;
  $('clone').innerHTML = '<option value="">None (use voice above)</option>' +
    cloneList.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
  if (cloneList.some(c => c.id === prev)) $('clone').value = prev;
  $('cloneStatus').textContent = d.ready ? '' : 'cloning engine warming up…';
  $('cloneListUI').innerHTML = cloneList.length ? cloneList.map(c =>
    `<div class="flex items-center justify-between py-1.5">
       <span>${c.name}</span>
       <button data-del="${c.id}" class="text-red-600 text-xs">Remove</button>
     </div>`).join('') : 'None yet.';
  $('cloneListUI').querySelectorAll('[data-del]').forEach(btn => btn.onclick = async () => {
    await fetch('/api/clone/' + btn.dataset.del, {method: 'DELETE'});
    await loadClones();
  });
  renderLibrary();
  if (!d.ready) setTimeout(loadClones, 5000);
}

// ---------- recording (shared by Clone tab) ----------
let mediaRecorder = null, recordedChunks = [], recordedBlob = null, recordTimer = null, recordSeconds = 0;
let recAnalyser = null, recRAF = null;
const MAX_RECORD_SECONDS = 25;

function drawLevelMeter() {
  const canvas = $('recLevel'), ctx = canvas.getContext('2d');
  const data = new Uint8Array(recAnalyser.frequencyBinCount);
  recAnalyser.getByteFrequencyData(data);
  const level = data.reduce((a, b) => a + b, 0) / data.length / 255;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#DD9029';
  ctx.fillRect(0, 0, canvas.width * Math.min(1, level * 1.8), canvas.height);
  recRAF = requestAnimationFrame(drawLevelMeter);
}
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    recordedChunks = [];
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = actx.createMediaStreamSource(stream);
    recAnalyser = actx.createAnalyser();
    recAnalyser.fftSize = 256;
    src.connect(recAnalyser);
    $('recLevel').classList.remove('hidden');
    drawLevelMeter();
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(recRAF);
      $('recLevel').classList.add('hidden');
      actx.close();
      recordedBlob = new Blob(recordedChunks, {type: mediaRecorder.mimeType || 'audio/webm'});
      const url = URL.createObjectURL(recordedBlob);
      $('recordPreview').src = url;
      $('recordPreview').classList.remove('hidden');
      $('cloneFile').value = '';
    };
    mediaRecorder.start();
    recordSeconds = 0;
    $('recordTime').textContent = '0s';
    recordTimer = setInterval(() => {
      recordSeconds += 1;
      $('recordTime').textContent = recordSeconds + 's';
      if (recordSeconds >= MAX_RECORD_SECONDS) stopRecording();
    }, 1000);
    $('cloneRecord').textContent = '\\u25A0 Stop';
  } catch (e) {
    $('cloneErr').textContent = 'microphone access failed: ' + e.message;
  }
}
function stopRecording() {
  clearInterval(recordTimer);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  $('cloneRecord').textContent = '\\uD83C\\uDFA4 Record';
}
$('cloneRecord').addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});
$('cloneFile').addEventListener('change', () => {
  recordedBlob = null;
  $('recordPreview').classList.add('hidden');
});
$('cloneAdd').addEventListener('click', async () => {
  $('cloneErr').textContent = '';
  const name = $('cloneName').value.trim();
  const file = $('cloneFile').files[0];
  const source = recordedBlob || file;
  if (!name || !source) { $('cloneErr').textContent = 'need a name, and either a recording or a file'; return; }
  $('cloneAdd').disabled = true;
  $('cloneAdd').textContent = 'Adding…';
  setAgentWorking(true, 'Locally agent cloning voice…');
  try {
    const fd = new FormData();
    fd.append('name', name);
    fd.append('audio', source, recordedBlob ? 'recording.webm' : file.name);
    const r = await fetch('/api/clone', {method: 'POST', body: fd});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'failed');
    $('cloneName').value = '';
    $('cloneFile').value = '';
    recordedBlob = null;
    $('recordPreview').classList.add('hidden');
    await loadClones();
    $('clone').value = d.id;
  } catch (e) {
    $('cloneErr').textContent = e.message;
  } finally {
    $('cloneAdd').disabled = false;
    $('cloneAdd').textContent = 'Add voice';
    setAgentWorking(false);
  }
});

// ---------- waveform ----------
let waveAudioCtx = null, wavePlayRAF = null;
async function drawWaveformFromBlob(blob, canvasId) {
  const canvas = $(canvasId);
  const ctx2d = canvas.getContext('2d');
  canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  try {
    if (!waveAudioCtx) waveAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await waveAudioCtx.decodeAudioData(await blob.arrayBuffer());
    const raw = buf.getChannelData(0);
    const samples = canvas.width;
    const blockSize = Math.floor(raw.length / samples) || 1;
    const peaks = [];
    for (let i = 0; i < samples; i++) {
      let max = 0;
      for (let j = 0; j < blockSize; j++) max = Math.max(max, Math.abs(raw[i * blockSize + j] || 0));
      peaks.push(max);
    }
    return peaks;
  } catch (e) {
    return null;
  }
}
function paintWave(canvasId, peaks, progress) {
  const canvas = $(canvasId), ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!peaks) return;
  const mid = h / 2;
  for (let i = 0; i < peaks.length; i++) {
    const barH = Math.max(1.5, peaks[i] * h * 0.9);
    ctx.fillStyle = (progress != null && i / peaks.length <= progress) ? '#DD9029' : 'rgba(107,104,98,0.35)';
    ctx.fillRect(i, mid - barH / 2, 1, barH);
  }
}

// ---------- history ----------
let history = [];
function renderHistory() {
  $('history').innerHTML = history.length ? history.map((h, i) =>
    `<div class="flex items-center justify-between gap-2 py-1">
       <span class="truncate flex-1">${h.text}</span>
       <span class="text-xs">${h.voiceLabel}</span>
       <button data-h="${i}" class="text-terracotta-600 text-xs">&#9654; Play</button>
       <a href="${h.url}" download="speech.wav" class="text-terracotta-600 text-xs">&#8595;</a>
     </div>`).join('') : 'No clips yet this session.';
  $('history').querySelectorAll('[data-h]').forEach(btn => btn.onclick = () => {
    const h = history[parseInt(btn.dataset.h)];
    const a = $('audio'); a.src = h.url; a.classList.remove('hidden'); a.play();
  });
}

// ---------- speak ----------
function setAgentWorking(working, label) {
  $('agent').classList.toggle('working', working);
  $('agentLabel').textContent = label || (working ? 'Locally agent working…' : 'Locally agent idle');
}

async function speak() {
  $('err').textContent = '';
  $('play').disabled = true;
  $('play').textContent = 'Generating…';
  setAgentWorking(true);
  const ctrl = new AbortController();
  const cloneId = $('clone').value;
  const giveUp = setTimeout(() => ctrl.abort(), cloneId ? 40000 : 20000);
  try {
    const text = $('text').value;
    const voiceName = $('voice').value;
    const r = await fetch('/api/speak', {method:'POST', headers:{'Content-Type':'application/json'}, signal: ctrl.signal,
      body: JSON.stringify({text, voice: voiceName, speed: parseFloat($('speed').value), clone_id: cloneId, anon_id: ANON_ID})});
    clearTimeout(giveUp);
    if (r.status === 402) { const e = await r.json(); openGate(e.message); throw new Error(e.message || 'limit reached'); }
    if (!r.ok) { const e = await r.json(); throw new Error(e.error || 'failed'); }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = $('audio'); a.src = url; a.classList.remove('hidden');
    a.play();
    $('dl').onclick = () => { const link = document.createElement('a'); link.href = url; link.download = 'speech.wav'; link.click(); };

    const peaks = await drawWaveformFromBlob(blob, 'wave');
    paintWave('wave', peaks, 0);
    cancelAnimationFrame(wavePlayRAF);
    const tick = () => {
      if (!a.paused && a.duration) paintWave('wave', peaks, a.currentTime / a.duration);
      wavePlayRAF = requestAnimationFrame(tick);
    };
    tick();

    const voiceLabel = (voiceList.find(v => v.name === voiceName) || {}).speaker || voiceName;
    history.unshift({text: text.slice(0, 60), voiceLabel: cloneId ? (cloneList.find(c => c.id === cloneId) || {}).name || voiceLabel : voiceLabel, url});
    history = history.slice(0, 10);
    renderHistory();
    loadAccount();
  } catch (e) {
    $('err').textContent = e.name === 'AbortError' ? 'Took too long, try again' : e.message;
  } finally {
    $('play').disabled = false;
    $('play').textContent = '\\u25B6 Play';
    setAgentWorking(false);
  }
}
$('play').onclick = speak;

// ---------- dialogue ----------
let turns = [];
function addTurnRow() {
  turns.push({label: 'Speaker ' + String.fromCharCode(65 + turns.length), lang: 'en', voice: '', cloneId: '', text: ''});
  renderTurns();
}
function renderTurns() {
  $('turns').innerHTML = turns.map((t, i) => {
    const langs = langsSorted(voiceList);
    const voicesForLang = voiceList.filter(v => v.baseLang === t.lang);
    if (!t.voice && voicesForLang[0]) t.voice = voicesForLang[0].name;
    return `<div class="border border-claude-border dark:border-claude-darkBorder rounded-xl p-3 space-y-2">
      <div class="flex flex-wrap gap-2 items-center text-sm">
        <input data-f="label" data-i="${i}" value="${t.label}" class="w-28 rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1">
        <select data-f="lang" data-i="${i}" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1">
          ${langs.map(l => `<option value="${l}" ${l===t.lang?'selected':''}>${LANG_NAMES[l]||l}</option>`).join('')}
        </select>
        <select data-f="voice" data-i="${i}" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1">
          ${voicesForLang.map(v => `<option value="${v.name}" ${v.name===t.voice?'selected':''}>${v.speaker}</option>`).join('')}
        </select>
        <select data-f="cloneId" data-i="${i}" class="rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1">
          <option value="">(no clone)</option>
          ${cloneList.map(c => `<option value="${c.id}" ${c.id===t.cloneId?'selected':''}>${c.name}</option>`).join('')}
        </select>
        <button data-rm="${i}" class="text-red-600 text-xs ml-auto">Remove</button>
      </div>
      <textarea data-f="text" data-i="${i}" rows="2" placeholder="What does ${t.label} say?" class="w-full rounded-lg border border-claude-border dark:border-claude-darkBorder bg-claude-bg dark:bg-claude-darkBg px-2 py-1.5 text-sm">${t.text}</textarea>
    </div>`;
  }).join('');
  $('turns').querySelectorAll('[data-f]').forEach(el => el.addEventListener('change', onTurnEdit));
  $('turns').querySelectorAll('textarea[data-f]').forEach(el => el.addEventListener('input', onTurnEdit));
  $('turns').querySelectorAll('[data-rm]').forEach(btn => btn.onclick = () => { turns.splice(parseInt(btn.dataset.rm), 1); renderTurns(); });
}
function onTurnEdit(e) {
  const i = parseInt(e.target.dataset.i), f = e.target.dataset.f;
  turns[i][f] = e.target.value;
  if (f === 'lang') { turns[i].voice = ''; renderTurns(); }
}
$('addTurn').onclick = addTurnRow;
$('playDialogue').addEventListener('click', async () => {
  if (!turns.length) { $('dialogueStatus').textContent = 'add at least one turn first'; return; }
  $('playDialogue').disabled = true;
  setAgentWorking(true, 'Locally agent performing dialogue…');
  const audio = $('dialogueAudio');
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i];
    if (!t.text.trim()) continue;
    $('dialogueStatus').textContent = `Turn ${i+1} of ${turns.length} (${t.label})…`;
    try {
      const r = await fetch('/api/speak', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text: t.text, voice: t.voice, speed: 1.0, clone_id: t.cloneId, anon_id: ANON_ID})});
      if (r.status === 402) { const e = await r.json(); openGate(e.message); throw new Error(e.message || 'limit reached'); }
      if (!r.ok) { const e = await r.json(); throw new Error(e.error || 'failed'); }
      const url = URL.createObjectURL(await r.blob());
      await new Promise((resolve, reject) => {
        audio.src = url;
        audio.onended = resolve;
        audio.onerror = reject;
        audio.play();
      });
    } catch (e) {
      $('dialogueStatus').textContent = 'error on turn ' + (i+1) + ': ' + e.message;
      $('playDialogue').disabled = false;
      setAgentWorking(false);
      return;
    }
  }
  $('dialogueStatus').textContent = 'Done.';
  $('playDialogue').disabled = false;
  setAgentWorking(false);
});
addTurnRow();
addTurnRow();

// ---------- library ----------
async function previewVoice(name) {
  const v = voiceList.find(v => v.name === name);
  const text = SAMPLE_TEXT[v ? v.baseLang : 'en'] || SAMPLE_TEXT.en;
  const r = await fetch('/api/speak', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text, voice: name, speed: 1.0, clone_id: '', anon_id: ANON_ID})});
  if (r.status === 402) { const e = await r.json(); openGate(e.message); return; }
  if (!r.ok) return;
  const a = $('audio'); a.src = URL.createObjectURL(await r.blob()); a.classList.remove('hidden'); a.play();
}
async function previewClone(cloneId) {
  const base = voiceList.find(v => v.baseLang === 'en') || voiceList[0];
  if (!base) return;
  const r = await fetch('/api/speak', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text: SAMPLE_TEXT.en, voice: base.name, speed: 1.0, clone_id: cloneId, anon_id: ANON_ID})});
  if (r.status === 402) { const e = await r.json(); openGate(e.message); return; }
  if (!r.ok) return;
  const a = $('audio'); a.src = URL.createObjectURL(await r.blob()); a.classList.remove('hidden'); a.play();
}
function renderLibrary() {
  const q = ($('librarySearch').value || '').toLowerCase();
  const rows = [];
  voiceList.forEach(v => {
    const label = `${LANG_NAMES[v.baseLang] || v.baseLang} \\u2014 ${v.speaker}`;
    if (!q || label.toLowerCase().includes(q)) rows.push({label, tag: 'Piper', preview: () => previewVoice(v.name)});
  });
  cloneList.forEach(c => {
    if (!q || c.name.toLowerCase().includes(q)) rows.push({label: c.name, tag: 'Cloned', preview: () => previewClone(c.id)});
  });
  $('libraryList').innerHTML = rows.map((row, i) =>
    `<div class="flex items-center justify-between py-2.5">
       <div><span class="font-medium">${row.label}</span>
         <span class="ml-2 text-xs rounded-full px-2 py-0.5 ${row.tag === 'Cloned' ? 'bg-terracotta-100 text-terracotta-700' : 'bg-claude-border dark:bg-claude-darkBorder text-claude-subtext dark:text-claude-darkSubtext'}">${row.tag}</span>
       </div>
       <button data-prev="${i}" class="text-terracotta-600 text-sm">&#9654; Preview</button>
     </div>`).join('') || '<div class="py-4 text-sm text-claude-subtext dark:text-claude-darkSubtext">No matches.</div>';
  $('libraryList').querySelectorAll('[data-prev]').forEach(btn => btn.onclick = () => rows[parseInt(btn.dataset.prev)].preview());
}
$('librarySearch').addEventListener('input', renderLibrary);

loadVoices();
loadClones();
loadAccount();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Free TTS app -> http://127.0.0.1:8900")
    threading.Thread(target=warm_converter, daemon=True).start()
    app.run(host="0.0.0.0", port=8900)
