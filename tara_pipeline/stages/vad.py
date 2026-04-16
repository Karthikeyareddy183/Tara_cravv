"""
VAD stage — Silero VAD.

Pi 5 justification:
  - 1MB model, CPU-only ONNX inference
  - <50ms on Pi 4 (faster on Pi 5 with faster CPU)
  - Trivial memory footprint, no GPU required
  - Purpose: gate audio chunks so STT only runs on voice-active segments,
    preserving the 1s STT budget for actual speech

Known failure modes (documented):
  - Very loud kitchen noise (whistle peak) can spike VAD probability → false trigger
    but DeepFilterNet upstream should attenuate this
  - Short speech bursts (<250ms) may fall below MIN_SPEECH_MS threshold → missed
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from loguru import logger

from tara_pipeline.config import (
    SAMPLE_RATE,
    CHUNK_DURATION_S,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    BUDGET_VAD_MS,
)
from tara_pipeline.utils.metrics import LatencyProfiler, stage_timer


class SpeechSegment(NamedTuple):
    """A detected speech segment with sample indices and timing."""
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_s: float


class SileroVAD:
    """
    Silero VAD wrapper.

    Detects speech segments in audio and returns their sample indices.
    Operates on full audio (batch mode) for pipeline use.
    """

    _model_cache: dict = {}  # class-level cache to avoid re-loading

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.profiler = profiler
        self._model, self._utils = self._load_model()

    def _load_model(self):
        cache_key = "silero_vad"
        if cache_key in SileroVAD._model_cache:
            logger.debug("SileroVAD: using cached model")
            return SileroVAD._model_cache[cache_key]

        logger.info("SileroVAD: loading from torch hub...")
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,  # ONNX backend — faster on CPU, Pi 5 compatible
            )
            SileroVAD._model_cache[cache_key] = (model, utils)
            logger.info("SileroVAD: loaded (ONNX backend)")
            return model, utils
        except Exception as e:
            logger.warning(f"SileroVAD ONNX load failed ({e}), trying PyTorch backend")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            SileroVAD._model_cache[cache_key] = (model, utils)
            logger.info("SileroVAD: loaded (PyTorch backend)")
            return model, utils

    def detect_segments(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> tuple[list[SpeechSegment], float]:
        """
        Run VAD on full audio and return speech segments.

        Returns
        -------
        segments   : list[SpeechSegment]
        elapsed_ms : float
        """
        get_speech_timestamps = self._utils[0]
        audio_tensor = torch.from_numpy(audio)

        with stage_timer("vad", self.profiler, BUDGET_VAD_MS) as t:
            raw_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                sampling_rate=sr,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=False,  # return samples
            )

        segments = []
        for ts in raw_timestamps:
            start_s = ts["start"] / sr
            end_s = ts["end"] / sr
            segments.append(SpeechSegment(
                start_sample=ts["start"],
                end_sample=ts["end"],
                start_s=start_s,
                end_s=end_s,
                duration_s=end_s - start_s,
            ))

        logger.info(
            f"VAD: {len(segments)} speech segments | "
            f"elapsed={t['elapsed_ms']:.1f}ms"
        )
        return segments, t["elapsed_ms"]

    def extract_segments(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> tuple[list[tuple[np.ndarray, SpeechSegment]], float]:
        """
        Detect + extract audio arrays for each speech segment.

        Returns list of (audio_array, SpeechSegment) tuples.
        """
        segments, elapsed_ms = self.detect_segments(audio, sr)
        extracted = []
        for seg in segments:
            chunk = audio[seg.start_sample : seg.end_sample]
            extracted.append((chunk, seg))
        return extracted, elapsed_ms

    def __call__(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> tuple[list[tuple[np.ndarray, SpeechSegment]], float]:
        return self.extract_segments(audio, sr)
