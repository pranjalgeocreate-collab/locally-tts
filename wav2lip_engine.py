"""Local CPU lip-sync: animate a single photo's mouth to match generated audio, via Wav2Lip (ONNX).

Reimplements Wav2Lip's exact mel-spectrogram pipeline (github.com/Rudrabha/Wav2Lip, audio.py/hparams.py)
in pure numpy/scipy instead of librosa - librosa's numba/llvmlite dependency has no prebuilt wheel for
this machine (same wall hit integrating OpenVoice earlier). Verified against Wav2Lip's hparams:
sample_rate=16000, n_fft=800, hop_size=200, win_size=800, num_mels=80, fmin=55, fmax=7600.

Also avoids OpenCV entirely - opencv-python-headless has no prebuilt wheel for this old macOS either
and triggers a multi-hour cmake/Ninja source build (confirmed, killed it). Face detection uses the
tiny (1.2MB) Ultra-Light-Fast-Generic-Face-Detector (RFB-320) ONNX model instead of cv2's Haar
cascade; image I/O uses Pillow; video assembly pipes PNG frames straight into ffmpeg.
"""
import os
import subprocess
import tempfile

import numpy as np
import onnxruntime
import soundfile as sf
from PIL import Image
from scipy.signal import lfilter

SR = 16000
N_FFT = 800
HOP = 200
WIN = 800
N_MELS = 80
FMIN = 55
FMAX = 7600
PREEMPHASIS = 0.97
REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100
MAX_ABS_VALUE = 4.0

FACE_SIZE = 96
MEL_STEP_SIZE = 16  # frames of mel per Wav2Lip inference chunk, matches the model's training window
FPS = 25

FFMPEG = os.environ.get("FFMPEG", "/usr/local/bin/ffmpeg")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "wav2lip_model", "wav2lip_gan.onnx")
FACE_MODEL_PATH = os.path.join(HERE, "wav2lip_model", "face_detector.onnx")


def _hz_to_mel(hz):
    # librosa's default (Slaney-style): linear below 1000Hz, log above - NOT the simpler HTK formula.
    hz = np.asarray(hz, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mel = (hz - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    is_log = hz >= min_log_hz
    mel = np.where(is_log, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    hz = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    is_log = mel >= min_log_mel
    hz = np.where(is_log, min_log_hz * np.exp(logstep * (mel - min_log_mel)), hz)
    return hz


_mel_basis_cache = None


def _mel_filterbank():
    global _mel_basis_cache
    if _mel_basis_cache is not None:
        return _mel_basis_cache
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SR / 2, n_freqs)
    mel_min, mel_max = _hz_to_mel(FMIN), _hz_to_mel(FMAX)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    weights = np.zeros((N_MELS, n_freqs))
    for i in range(N_MELS):
        lo, center, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / (center - lo)
        right = (hi - fft_freqs) / (hi - center)
        weights[i] = np.maximum(0, np.minimum(left, right))
    # Slaney-style area normalization (librosa's default norm='slaney')
    enorm = 2.0 / (hz_pts[2:N_MELS + 2] - hz_pts[:N_MELS])
    weights *= enorm[:, np.newaxis]
    _mel_basis_cache = weights
    return weights


def _stft(y):
    # Matches librosa.stft defaults: center=True (reflect-padded), hann window.
    pad = N_FFT // 2
    y_padded = np.pad(y, pad, mode="reflect")
    window = np.hanning(WIN + 1)[:-1]  # librosa/scipy periodic hann
    n_frames = 1 + (len(y_padded) - N_FFT) // HOP
    frames = np.empty((n_frames, N_FFT), dtype=np.float64)
    for i in range(n_frames):
        start = i * HOP
        seg = y_padded[start:start + N_FFT]
        windowed = np.zeros(N_FFT)
        offset = (N_FFT - WIN) // 2
        windowed[offset:offset + WIN] = seg[offset:offset + WIN] * window
        frames[i] = windowed
    spec = np.fft.rfft(frames, n=N_FFT, axis=1).T  # (freq, time), matches librosa's orientation
    return spec


def melspectrogram(wav):
    emphasized = lfilter([1, -PREEMPHASIS], [1], wav)
    D = _stft(emphasized)
    mel_basis = _mel_filterbank()
    S = np.dot(mel_basis, np.abs(D))
    min_level = np.exp(MIN_LEVEL_DB / 20 * np.log(10))
    S_db = 20 * np.log10(np.maximum(min_level, S)) - REF_LEVEL_DB
    normalized = np.clip((2 * MAX_ABS_VALUE) * ((S_db - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE,
                          -MAX_ABS_VALUE, MAX_ABS_VALUE)
    return normalized


_face_session = None
_lipsync_session = None


def get_face_session():
    global _face_session
    if _face_session is None:
        _face_session = onnxruntime.InferenceSession(FACE_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _face_session


def get_lipsync_session():
    global _lipsync_session
    if _lipsync_session is None:
        _lipsync_session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _lipsync_session


def _nms(boxes, scores, iou_threshold=0.3):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-9)
        order = order[1:][iou < iou_threshold]
    return keep


def detect_face_box(image):
    """image: PIL.Image (RGB). Returns (x0,y0,x1,y1) in original image pixel coords."""
    w, h = image.size
    resized = image.resize((320, 240), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32)  # RGB, (240,320,3)
    arr = (arr - 127.0) / 128.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis]  # (1,3,240,320)

    session = get_face_session()
    input_name = session.get_inputs()[0].name
    scores, boxes = session.run(None, {input_name: arr})
    scores = scores[0][:, 1]  # face class confidence
    boxes = boxes[0]  # normalized (x0,y0,x1,y1) in [0,1]

    mask = scores > 0.7
    if not mask.any():
        mask = scores > 0.5
    if not mask.any():
        raise ValueError("no face detected in that photo - try a clearer, front-facing photo")
    boxes, scores = boxes[mask], scores[mask]
    keep = _nms(boxes, scores)
    best = keep[0]
    x0, y0, x1, y1 = boxes[best]
    x0, x1 = x0 * w, x1 * w
    y0, y1 = y0 * h, y1 * h
    pad_h = 0.2 * (y1 - y0)
    y0, y1 = max(0, y0 - pad_h), min(h, y1 + pad_h)
    x0, x1 = max(0, x0), min(w, x1)
    return int(x0), int(y0), int(x1), int(y1)


def generate(image_path, audio_path, output_path, progress_cb=None):
    """Lip-sync a single still photo to the given audio. progress_cb(done, total) is called per frame."""
    image = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = detect_face_box(image)
    face = image.crop((x0, y0, x1, y1))
    face_resized = np.asarray(face.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)).astype(np.uint8)  # RGB

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype("float32")

    mel = melspectrogram(wav)  # (n_mels, time)
    mel_chunks = []
    mel_idx_multiplier = 80.0 / FPS
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + MEL_STEP_SIZE])
        i += 1
    if not mel_chunks:
        raise ValueError("audio too short to lip-sync")

    session = get_lipsync_session()

    masked = face_resized.copy()
    masked[FACE_SIZE // 2:, :] = 0
    img_batch_base = np.concatenate([masked, face_resized], axis=2).astype(np.float32) / 255.0  # (96,96,6)

    image_arr = np.asarray(image)  # RGB (H,W,3)

    with tempfile.TemporaryDirectory() as tmp:
        total = len(mel_chunks)
        for idx, mel_chunk in enumerate(mel_chunks):
            img_batch = img_batch_base[np.newaxis].transpose(0, 3, 1, 2)  # (1,6,96,96)
            mel_batch = mel_chunk[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,80,16)
            ort_inputs = {"vid": img_batch, "mel": mel_batch}
            pred = session.run(None, ort_inputs)[0]  # (1,3,96,96)
            pred_img = (pred[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

            out_frame = image_arr.copy()
            mouth_img = Image.fromarray(pred_img).resize((x1 - x0, y1 - y0), Image.BILINEAR)
            out_frame[y0:y1, x0:x1] = np.asarray(mouth_img)
            Image.fromarray(out_frame).save(os.path.join(tmp, f"f{idx:05d}.png"))
            if progress_cb:
                progress_cb(idx + 1, total)

        subprocess.run([
            FFMPEG, "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
            "-i", audio_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-shortest", output_path,
        ], capture_output=True, check=True, timeout=180)
