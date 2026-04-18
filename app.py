"""
Tara Dashboard — FastAPI backend.
POST /process  → upload audio, run pipeline, return denoised audio + transcript
GET  /          → serve frontend HTML
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tara_pipeline.config import SAMPLE_RATE
from tara_pipeline.utils.audio import load_audio
from tara_pipeline.stages.noise_suppression import create_suppressor
from tara_pipeline.pipeline import TaraPipeline

app = FastAPI(title="Tara Voice Pipeline Dashboard")

# Lazy-load heavy models once at startup
_suppressor = None
_pipeline = None


def get_suppressor():
    global _suppressor
    if _suppressor is None:
        _suppressor = create_suppressor("deepfilternet")
    return _suppressor


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TaraPipeline(iteration=4, wake_word_backend="deepgram")
    return _pipeline


# Serve static frontend
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    """
    Accept audio upload, run DeepFilterNet + full Tara pipeline.
    Returns JSON with transcript and denoised audio as base64 WAV.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    allowed = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use: {allowed}")

    # Save upload to temp file
    raw_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        # Load audio
        audio, sr = load_audio(tmp_path, target_sr=SAMPLE_RATE)
        duration_s = len(audio) / sr

        # Stage 1: Noise suppression
        suppressor = get_suppressor()
        audio_clean, ns_ms = suppressor.suppress(audio, sr)

        # Encode denoised audio as WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_clean, sr, format="WAV", subtype="PCM_16")
        denoised_wav = buf.getvalue()

        # Stage 2–4: Full pipeline (VAD + wake word + STT)
        pipeline = get_pipeline()
        result = pipeline.run(tmp_path)

        commands = [
            {
                "start_s": round(cmd.segment_start_s, 2),
                "end_s": round(cmd.segment_end_s, 2),
                "transcript": cmd.transcript,
                "wake_word_score": round(cmd.wake_word_score, 3),
                "wake_word_backend": cmd.wake_word_backend,
                "total_ms": round(cmd.total_ms),
                "over_budget": cmd.over_budget,
                "timings": {k: round(v) for k, v in cmd.timings.items()},
            }
            for cmd in result.commands
        ]

        import base64
        return JSONResponse({
            "duration_s": round(duration_s, 2),
            "vad_segments": result.vad_segment_count,
            "wake_word_triggers": result.wake_word_trigger_count,
            "wake_word_rejected": result.wake_word_reject_count,
            "commands": commands,
            "noise_suppression_ms": round(ns_ms),
            "denoised_audio_b64": base64.b64encode(denoised_wav).decode(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
