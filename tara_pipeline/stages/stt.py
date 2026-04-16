"""
Speech-to-Text stage.

Two implementations:
  1. WhisperSTT        — openai-whisper (Iterations 1 & 2 baseline)
  2. FasterWhisperSTT  — faster-whisper tiny.en (Iterations 3 & 4 final)

Latency budget: ≤ 1000ms hard cap.

Pi 5 note:
  STT is the ONE stage NOT required to run on Pi 5 (per assignment constraints).
  It can run on a server or edge device. faster-whisper tiny.en runs in
  150–300ms on CPU, well within budget even on modest hardware.

Known failure modes (documented):
  - Latency varies with utterance length: longer commands = more tokens = slower
  - Hallucination on noise-only segments (mitigated by VAD upstream)
  - tiny.en lower accuracy than base but faster — trade-off documented
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from loguru import logger

from tara_pipeline.config import (
    SAMPLE_RATE,
    WHISPER_BASE_MODEL,
    WHISPER_BASE_LANGUAGE,
    FASTER_WHISPER_MODEL,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_COMPUTE_TYPE,
    FASTER_WHISPER_BEAM_SIZE,
    BUDGET_STT_MS,
)
from tara_pipeline.utils.metrics import LatencyProfiler, stage_timer


@dataclass
class TranscriptionResult:
    text: str
    elapsed_ms: float
    model: str
    language: str | None = None
    segments: list[dict] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class BaseSTT(ABC):
    """Abstract STT interface."""

    def __init__(self, profiler: LatencyProfiler | None = None) -> None:
        self.profiler = profiler

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TranscriptionResult:
        """Transcribe float32 mono audio. Returns TranscriptionResult."""
        ...

    def __call__(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TranscriptionResult:
        return self.transcribe(audio, sr)


class WhisperSTT(BaseSTT):
    """
    openai-whisper baseline STT (Iterations 1 & 2).

    Slower than faster-whisper (~3-5x) but used to establish baseline.
    Expected to struggle on noisy audio without preprocessing.
    """

    def __init__(
        self,
        model_name: str = WHISPER_BASE_MODEL,
        language: str = WHISPER_BASE_LANGUAGE,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        self.model_name = model_name
        self.language = language
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import whisper
            logger.info(f"WhisperSTT: loading model '{self.model_name}'...")
            t0 = time.perf_counter()
            self._model = whisper.load_model(self.model_name)
            load_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"WhisperSTT: '{self.model_name}' loaded in {load_ms:.0f}ms")
        except ImportError as e:
            raise ImportError(
                "Install openai-whisper: pip install openai-whisper"
            ) from e

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TranscriptionResult:
        import whisper
        import librosa

        # Whisper expects 16kHz float32
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        with stage_timer("stt", self.profiler, BUDGET_STT_MS) as t:
            result = self._model.transcribe(
                audio,
                language=self.language,
                fp16=False,  # CPU only
                verbose=False,
            )

        text = result["text"].strip()
        logger.info(f"WhisperSTT ({self.model_name}): '{text}' | {t['elapsed_ms']:.0f}ms")

        if t["elapsed_ms"] > BUDGET_STT_MS:
            logger.warning(
                f"STT OVER BUDGET: {t['elapsed_ms']:.0f}ms > {BUDGET_STT_MS}ms"
            )

        return TranscriptionResult(
            text=text,
            elapsed_ms=t["elapsed_ms"],
            model=f"whisper-{self.model_name}",
            language=result.get("language"),
            segments=[
                {"text": s["text"], "start": s["start"], "end": s["end"]}
                for s in result.get("segments", [])
            ],
        )


class FasterWhisperSTT(BaseSTT):
    """
    faster-whisper STT — final pipeline (Iterations 3 & 4).

    Uses CTranslate2 int8 quantisation for ~4x speed vs openai-whisper.
    tiny.en: 150–300ms on CPU, within 1s budget.

    Why tiny.en over base:
      - English-only: smaller model, faster inference, no language detection overhead
      - Accuracy sufficient for kitchen commands ("add salt", "set timer 5 minutes")
      - tiny.en WER on clean audio ~8-12%, acceptable for command intent detection
    """

    def __init__(
        self,
        model_name: str = FASTER_WHISPER_MODEL,
        device: str = FASTER_WHISPER_DEVICE,
        compute_type: str = FASTER_WHISPER_COMPUTE_TYPE,
        beam_size: int = FASTER_WHISPER_BEAM_SIZE,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"FasterWhisperSTT: loading '{self.model_name}' | "
                f"device={self.device} | compute={self.compute_type}"
            )
            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            load_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"FasterWhisperSTT: loaded in {load_ms:.0f}ms")
        except ImportError as e:
            raise ImportError(
                "Install faster-whisper: pip install faster-whisper"
            ) from e

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TranscriptionResult:
        import librosa

        # faster-whisper expects 16kHz float32
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        with stage_timer("stt", self.profiler, BUDGET_STT_MS) as t:
            segments_iter, info = self._model.transcribe(
                audio,
                beam_size=self.beam_size,
                language="en",
                vad_filter=False,  # we run upstream VAD
                condition_on_previous_text=False,  # no context bleed between segments
            )
            segments = list(segments_iter)  # consume generator inside timer

        text = " ".join(s.text.strip() for s in segments).strip()
        logger.info(
            f"FasterWhisperSTT ({self.model_name}): '{text}' | "
            f"{t['elapsed_ms']:.0f}ms"
        )

        if t["elapsed_ms"] > BUDGET_STT_MS:
            logger.warning(
                f"STT OVER BUDGET: {t['elapsed_ms']:.0f}ms > {BUDGET_STT_MS}ms"
            )

        return TranscriptionResult(
            text=text,
            elapsed_ms=t["elapsed_ms"],
            model=f"faster-whisper-{self.model_name}",
            language=info.language,
            segments=[
                {"text": s.text, "start": s.start, "end": s.end}
                for s in segments
            ],
        )


class DeepgramSTT(BaseSTT):
    """
    Deepgram Nova-2 cloud STT.

    Why Deepgram over faster-whisper tiny.en:
      - Assignment explicitly excludes STT from Pi 5 constraint → cloud API valid
      - Nova-2 accuracy: ~95%+ vs tiny.en ~85% on noisy speech
      - API latency: ~300-500ms vs 644ms CPU inference
      - Handles accented speech and kitchen noise residuals better

    Cost: Free tier 200 hrs/month (console.deepgram.com — no credit card).

    Set DEEPGRAM_API_KEY env var. Never hardcode in source.
    """

    API_URL = "https://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en",
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        import os
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Deepgram requires DEEPGRAM_API_KEY env var. "
                "Free key at: https://console.deepgram.com"
            )
        self.model = model
        self.language = language
        logger.info(f"DeepgramSTT: model={model} | language={language} | API ready")

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TranscriptionResult:
        import io
        import wave
        import requests

        # Convert float32 → 16-bit PCM WAV bytes in memory
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = buf.getvalue()

        with stage_timer("stt", self.profiler, BUDGET_STT_MS) as t:
            resp = requests.post(
                self.API_URL,
                params={"model": self.model, "language": self.language, "smart_format": "true"},
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                data=wav_bytes,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        transcript = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
            .strip()
        )
        confidence = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("confidence", 0.0)
        )

        logger.info(
            f"DeepgramSTT ({self.model}): '{transcript}' | "
            f"conf={confidence:.2f} | {t['elapsed_ms']:.0f}ms"
        )
        if t["elapsed_ms"] > BUDGET_STT_MS:
            logger.warning(f"STT OVER BUDGET: {t['elapsed_ms']:.0f}ms > {BUDGET_STT_MS}ms")

        return TranscriptionResult(
            text=transcript,
            elapsed_ms=t["elapsed_ms"],
            model=f"deepgram-{self.model}",
            language=self.language,
        )


def create_stt(
    backend: str = "faster_whisper",
    profiler: LatencyProfiler | None = None,
    **kwargs,
) -> BaseSTT:
    """
    Factory for STT backends.

    Parameters
    ----------
    backend : "faster_whisper" | "whisper" | "deepgram"
    """
    if backend == "faster_whisper":
        return FasterWhisperSTT(profiler=profiler, **kwargs)
    elif backend == "whisper":
        return WhisperSTT(profiler=profiler, **kwargs)
    elif backend == "deepgram":
        return DeepgramSTT(profiler=profiler, **kwargs)
    else:
        raise ValueError(f"Unknown STT backend: {backend!r}")
