"""
Audio I/O utilities — load, resample, chunk, convert.
All audio is normalised to float32 mono 16kHz for pipeline compatibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import librosa
from loguru import logger

from tara_pipeline.config import SAMPLE_RATE, CHUNK_DURATION_S


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load any audio file (FLAC, WAV, MP3) → float32 mono at target_sr.

    Returns
    -------
    audio : np.ndarray  shape (N,) float32 in [-1, 1]
    sample_rate : int
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.debug(f"Loading audio: {path}")
    t0 = time.perf_counter()

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        logger.debug(f"Resampling {sr}Hz → {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    duration_s = len(audio) / target_sr
    logger.info(
        f"Loaded {path.name}: {duration_s:.2f}s audio | "
        f"sr={target_sr} | load_time={elapsed_ms:.1f}ms"
    )
    return audio.astype(np.float32), target_sr


def chunk_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    chunk_duration_s: float = CHUNK_DURATION_S,
) -> Generator[np.ndarray, None, None]:
    """
    Yield fixed-size audio chunks for streaming VAD.

    Parameters
    ----------
    audio           : float32 mono waveform
    sr              : sample rate
    chunk_duration_s: duration of each chunk in seconds
    """
    chunk_samples = int(sr * chunk_duration_s)
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        # Pad last chunk to full size so VAD models don't complain
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM — required by some VAD/wake word models."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 [-1, 1]."""
    return audio.astype(np.float32) / 32768.0


def save_audio(audio: np.ndarray, path: str | Path, sr: int = SAMPLE_RATE) -> None:
    """Save float32 waveform to WAV/FLAC."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    logger.debug(f"Saved audio: {path}")


def split_on_silence_segments(
    audio: np.ndarray,
    speech_timestamps: list[dict],
    sr: int = SAMPLE_RATE,
    pad_ms: int = 30,
) -> list[np.ndarray]:
    """
    Extract speech segments from audio using VAD timestamps.

    Parameters
    ----------
    audio             : full waveform
    speech_timestamps : list of {'start': int, 'end': int} in samples
    sr                : sample rate
    pad_ms            : ms to pad each side

    Returns
    -------
    List of audio segments (float32 arrays)
    """
    pad_samples = int(sr * pad_ms / 1000)
    segments = []
    for ts in speech_timestamps:
        start = max(0, ts["start"] - pad_samples)
        end = min(len(audio), ts["end"] + pad_samples)
        segments.append(audio[start:end])
    return segments


def get_audio_duration(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Return duration in seconds."""
    return len(audio) / sr
