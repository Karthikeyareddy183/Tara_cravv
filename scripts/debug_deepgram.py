"""
Diagnostic: test Deepgram API parameters on the actual 'Tara' segment.
Extracts 30-40s of raw audio (where Tara command expected) and tests variants.
"""
import os, io, wave, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import requests
from tara_pipeline.utils.audio import load_audio
from tara_pipeline.stages.noise_suppression import DeepFilterNetSuppressor
from tara_pipeline.config import SAMPLE_RATE

API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set DEEPGRAM_API_KEY env var")

AUDIO_FILE = "assets/tara_assignment_recording_clipped.flac"
API_URL = "https://api.deepgram.com/v1/listen"

def to_wav(audio: np.ndarray, sr: int) -> bytes:
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()

def query(wav_bytes: bytes, params: dict, label: str):
    resp = requests.post(
        API_URL,
        params=params,
        headers={"Authorization": f"Token {API_KEY}", "Content-Type": "audio/wav"},
        data=wav_bytes, timeout=15,
    )
    data = resp.json()
    transcript = (
        data.get("results", {}).get("channels", [{}])[0]
        .get("alternatives", [{}])[0].get("transcript", "").strip()
    )
    conf = (
        data.get("results", {}).get("channels", [{}])[0]
        .get("alternatives", [{}])[0].get("confidence", 0.0)
    )
    print(f"[{label}] transcript='{transcript}' conf={conf:.3f}")

print("Loading audio...")
audio, sr = load_audio(AUDIO_FILE, target_sr=SAMPLE_RATE)
print(f"Loaded {len(audio)/sr:.1f}s audio")

print("Running DeepFilterNet denoising...")
suppressor = DeepFilterNetSuppressor()
denoised, _ = suppressor.suppress(audio, sr)
print("Denoising done")

# Simulate pipeline probe windows with VAD_SPEECH_PAD_MS=1500
# Original VAD segment start: ~35.71s. With 1500ms pad: starts at 34.21s.
# WAKE_WORD_BUFFER_S=1.0s → detect_at_utterance_start passes 34.21-35.21s
# DEEPGRAM_WAKE_PROBE_S=1.5s → detect() probes 34.21-35.71s (after override)
for label, start_s, end_s in [
    ("34.21+1.5s (current)", 34.21, 35.71),
    ("34.21+2.0s", 34.21, 36.21),
    ("34.21+3.0s (proposed)", 34.21, 37.21),
    ("34.0+3.0s (reference)", 34.0, 37.0),
]:
    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    clip = denoised[start_i:end_i]
    wav = to_wav(clip, sr)
    print(f"\n--- {label} ({end_s-start_s:.2f}s) ---")
    query(wav, {"model": "nova-3", "language": "en"}, "nova-3 no-keyterm")
    query(wav, {"model": "nova-3", "language": "en", "keyterm": "Tara"}, "nova-3 keyterm=Tara")
