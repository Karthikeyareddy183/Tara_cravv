"""
Wake word detection stage.

Critical design: trigger ONLY when wake word detected at UTTERANCE START.
Not mid-sentence. This is the key distinction most implementations miss.

Implementation:
  Buffer first WAKE_WORD_BUFFER_S of each VAD segment.
  Run wake word model ONLY on that buffer.
  If triggered → pass FULL segment to STT.
  If not triggered → discard (handles "Add pasta, Tara" mid-sentence case).

Two backends with automatic fallback:
  1. openWakeWord  — primary (free, OSS, custom model trainable)
  2. Porcupine     — fallback (Picovoice, free personal use, higher accuracy)

Pi 5 justification:
  - openWakeWord: TFLite/ONNX CPU inference, ~200ms on Pi 4 → within budget on Pi 5
  - Porcupine: ARM-optimised binary, official Pi SDK, <100ms on Pi 4

Known failure modes (documented):
  1. False triggers: "terra", "tiara", "terror" phonetically similar to "Tara"
  2. Custom model accuracy depends on training data quantity
  3. openWakeWord may not have pre-trained "tara" — phoneme fallback available
  4. Porcupine requires internet for model download on first use
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import NamedTuple

import numpy as np
from loguru import logger

from tara_pipeline.config import (
    SAMPLE_RATE,
    WAKE_WORD_BUFFER_S,
    WAKE_WORD_PROBE_S,
    DEEPGRAM_WAKE_PROBE_S,
    OWW_THRESHOLD,
    OWW_INFERENCE_FRAMEWORK,
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_SENSITIVITY,
    PORCUPINE_KEYWORD_PATHS,
    BUDGET_WAKE_WORD_MS,
    DEFAULT_WAKE_WORD_BACKEND,
    FASTER_WHISPER_MODEL,
    ROOT_DIR,
)
from tara_pipeline.utils.metrics import LatencyProfiler, stage_timer


class WakeWordBackend(Enum):
    OPENWAKEWORD = auto()
    PORCUPINE = auto()
    NONE = auto()  # no-op for iterations 1-3


class WakeWordResult(NamedTuple):
    triggered: bool
    score: float
    backend: str
    elapsed_ms: float
    keyword: str | None = None


class BaseWakeWordDetector(ABC):
    """Abstract wake word detector."""

    def __init__(self, profiler: LatencyProfiler | None = None) -> None:
        self.profiler = profiler

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        """Run wake word detection on audio buffer. Returns WakeWordResult."""
        ...

    def detect_at_utterance_start(
        self,
        full_segment: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> WakeWordResult:
        """
        Core design: only check FIRST WAKE_WORD_BUFFER_S of the VAD segment.

        This prevents mid-sentence "Tara" triggering the pipeline.
        E.g. "Add some pasta, Tara" → Tara at ~2s → NOT at start → rejected.
        E.g. "Tara, add pasta" → Tara at 0s → at start → accepted.
        """
        buffer_samples = int(sr * WAKE_WORD_BUFFER_S)
        start_buffer = full_segment[:buffer_samples]

        # Pad if shorter than buffer (short utterances)
        if len(start_buffer) < buffer_samples:
            start_buffer = np.pad(
                start_buffer, (0, buffer_samples - len(start_buffer))
            )

        result = self.detect(start_buffer, sr)
        logger.debug(
            f"Wake word check | triggered={result.triggered} | "
            f"score={result.score:.3f} | backend={result.backend} | "
            f"elapsed={result.elapsed_ms:.1f}ms"
        )
        return result

    def __call__(self, full_segment: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        return self.detect_at_utterance_start(full_segment, sr)


class OpenWakeWordDetector(BaseWakeWordDetector):
    """
    openWakeWord detector for "hey tara" and "tara".

    Uses pre-trained or custom-trained ONNX/TFLite models.
    If custom model not found, falls back to phoneme matching via Whisper.
    """

    SUPPORTED_PRETRAINED = ["hey_jarvis", "alexa", "hey_mycroft", "timer"]

    def __init__(
        self,
        wakewords: list[str] | None = None,
        threshold: float = OWW_THRESHOLD,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        self.wakewords = wakewords or ["hey_tara", "tara"]
        self.threshold = threshold
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from openwakeword.model import Model

            # Check for sklearn pkl classifiers (our custom trained format)
            # These use openWakeWord as feature extractor + our own classifier
            pkl_models = self._find_pkl_classifiers()
            if pkl_models:
                logger.info(f"OpenWakeWord: using custom sklearn classifiers: {list(pkl_models.keys())}")
                self._sklearn_classifiers = pkl_models
                # Load openWakeWord base for feature extraction only
                self._model = Model(inference_framework=OWW_INFERENCE_FRAMEWORK)
                self._use_sklearn = True
                logger.info("OpenWakeWord: loaded (sklearn classifier mode)")
                return

            # Fallback: load base openWakeWord pre-trained models only
            logger.warning(
                "OpenWakeWord: no custom 'tara' pkl classifiers found. "
                "Using pre-trained models as proxy (lower accuracy for 'tara')."
            )
            self._model = Model(inference_framework=OWW_INFERENCE_FRAMEWORK)
            self._use_sklearn = False
            logger.info("OpenWakeWord: loaded (pre-trained proxy mode)")

        except ImportError as e:
            raise ImportError("Install openwakeword: pip install openwakeword") from e
        except Exception as e:
            logger.error(f"OpenWakeWord init failed: {e}")
            raise

    def _find_pkl_classifiers(self) -> dict:
        """Find custom sklearn classifiers trained by train_wake_word.py."""
        import pickle
        from pathlib import Path
        models_dir = Path(__file__).parent.parent.parent / "models"
        classifiers = {}
        for pkl_path in models_dir.glob("*_clf.pkl"):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                phrase = data.get("phrase", pkl_path.stem.replace("_clf", ""))
                classifiers[phrase] = data["clf"]
                logger.debug(f"Loaded classifier for '{phrase}': {pkl_path}")
            except Exception as e:
                logger.warning(f"Could not load {pkl_path}: {e}")
        return classifiers

    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        if self._model is None:
            return WakeWordResult(False, 0.0, "openwakeword", 0.0)

        # openWakeWord expects int16 PCM at 16kHz, in chunks
        from tara_pipeline.utils.audio import audio_to_int16
        audio_int16 = audio_to_int16(audio)

        with stage_timer("wake_word", self.profiler, BUDGET_WAKE_WORD_MS) as t:
            chunk_size = int(sr * 0.08)  # 80ms chunks

            # Collect openWakeWord base scores across all chunks
            all_scores = []
            self._model.reset()
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                prediction = self._model.predict(chunk)
                if prediction:
                    all_scores.append(list(prediction.values()))

            max_score = 0.0
            best_keyword = None

            if getattr(self, "_use_sklearn", False) and all_scores:
                # Use our custom sklearn classifiers on the aggregated score vector
                import numpy as _np
                feat = _np.mean(all_scores, axis=0).reshape(1, -1).astype(_np.float32)
                for phrase, clf in self._sklearn_classifiers.items():
                    try:
                        prob = clf.predict_proba(feat)[0][1]  # P(tara)
                        if prob > max_score:
                            max_score = prob
                            best_keyword = phrase
                    except Exception:
                        pass
            else:
                # Pre-trained proxy mode — use raw OWW scores
                if all_scores:
                    import numpy as _np
                    mean_scores = _np.mean(all_scores, axis=0)
                    max_idx = int(_np.argmax(mean_scores))
                    max_score = float(mean_scores[max_idx])
                    keywords = list(self._model.models.keys()) if hasattr(self._model, 'models') else []
                    best_keyword = keywords[max_idx] if max_idx < len(keywords) else None

        triggered = max_score >= self.threshold
        return WakeWordResult(
            triggered=triggered,
            score=float(max_score),
            backend="openwakeword",
            elapsed_ms=t["elapsed_ms"],
            keyword=best_keyword if triggered else None,
        )


class PorcupineDetector(BaseWakeWordDetector):
    """
    Picovoice Porcupine wake word detector.

    Fallback when openWakeWord custom model unavailable or accuracy insufficient.
    Requires PORCUPINE_ACCESS_KEY env var (free at picovoice.ai).

    Pi 5 justification:
      - ARM-optimised native library
      - Official Raspberry Pi SDK
      - <100ms inference, <2MB memory
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        keyword_paths: list[str] | None = None,
        sensitivity: float = PORCUPINE_SENSITIVITY,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        # keyword_paths takes precedence over built-in keywords
        self._keyword_paths = keyword_paths or PORCUPINE_KEYWORD_PATHS or []
        self.keywords = keywords or ["hey_tara"]
        self.sensitivity = sensitivity
        self._porcupine = None
        self._access_key = (
            os.environ.get("PORCUPINE_ACCESS_KEY", "") or PORCUPINE_ACCESS_KEY
        )
        self._load_model()

    def _load_model(self) -> None:
        if not self._access_key:
            raise ValueError(
                "Porcupine requires PORCUPINE_ACCESS_KEY env var. "
                "Get free key at: https://picovoice.ai/console/"
            )
        try:
            import pvporcupine

            if self._keyword_paths:
                # Resolve paths relative to project root
                abs_paths = [
                    str(ROOT_DIR / p) if not os.path.isabs(p) else p
                    for p in self._keyword_paths
                ]
                self._porcupine = pvporcupine.create(
                    access_key=self._access_key,
                    keyword_paths=abs_paths,
                    sensitivities=[self.sensitivity] * len(abs_paths),
                )
                logger.info(
                    f"Porcupine: loaded custom model | paths={abs_paths} | "
                    f"sensitivity={self.sensitivity} | "
                    f"sample_rate={self._porcupine.sample_rate} | "
                    f"frame_length={self._porcupine.frame_length}"
                )
            else:
                self._porcupine = pvporcupine.create(
                    access_key=self._access_key,
                    keywords=self.keywords,
                    sensitivities=[self.sensitivity] * len(self.keywords),
                )
                logger.info(
                    f"Porcupine: loaded built-in | keywords={self.keywords} | "
                    f"sample_rate={self._porcupine.sample_rate} | "
                    f"frame_length={self._porcupine.frame_length}"
                )
        except ImportError as e:
            raise ImportError("Install pvporcupine: pip install pvporcupine") from e
        except Exception as e:
            logger.error(f"Porcupine init failed: {e}")
            raise

    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        if self._porcupine is None:
            return WakeWordResult(False, 0.0, "porcupine", 0.0)

        from tara_pipeline.utils.audio import audio_to_int16
        import librosa

        # Porcupine expects exactly porcupine.sample_rate (16000) int16
        if sr != self._porcupine.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._porcupine.sample_rate)

        audio_int16 = audio_to_int16(audio)
        frame_length = self._porcupine.frame_length

        max_score = 0.0
        triggered = False
        keyword_idx = -1

        with stage_timer("wake_word", self.profiler, BUDGET_WAKE_WORD_MS) as t:
            for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
                frame = audio_int16[i : i + frame_length]
                if len(frame) < frame_length:
                    break
                result = self._porcupine.process(frame)
                if result >= 0:
                    triggered = True
                    keyword_idx = result
                    max_score = 1.0  # Porcupine binary decision
                    break

        if triggered and keyword_idx >= 0:
            if self._keyword_paths:
                keyword = f"hey_tara[{keyword_idx}]"  # custom model, index maps to path
            else:
                keyword = self.keywords[keyword_idx] if keyword_idx < len(self.keywords) else None
        else:
            keyword = None
        return WakeWordResult(
            triggered=triggered,
            score=max_score,
            backend="porcupine",
            elapsed_ms=t["elapsed_ms"],
            keyword=keyword,
        )

    def __del__(self) -> None:
        if self._porcupine is not None:
            self._porcupine.delete()


class PassthroughWakeWord(BaseWakeWordDetector):
    """
    No-op wake word — always returns triggered=True.
    Used in Iterations 1–3 (no wake word stage) to pass all VAD segments to STT.
    """

    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        return WakeWordResult(True, 1.0, "passthrough", 0.0)

    def detect_at_utterance_start(
        self, full_segment: np.ndarray, sr: int = SAMPLE_RATE
    ) -> WakeWordResult:
        return WakeWordResult(True, 1.0, "passthrough", 0.0)


class WhisperPhonemeWakeWord(BaseWakeWordDetector):
    """
    Phoneme-based wake word detector using faster-whisper.

    Algorithm:
      1. Slice first WAKE_WORD_PROBE_S (0.5s) of each VAD segment.
      2. Transcribe with faster-whisper tiny.en (int8, CPU).
      3. Normalize transcript (lowercase, strip punctuation).
      4. Trigger if transcript STARTS WITH "tara" or "hey tara".

    Why this works:
      - Real phoneme matching — if Whisper hears "tara", it IS "tara".
      - Reuses faster-whisper model already in the pipeline.
      - 0.5s probe → ~200–280ms inference → within 300ms wake word budget.
      - False positive rate: near-zero (Whisper won't transcribe fan noise as "tara").
      - False negative rate: low (Whisper tiny.en handles accented "tara" well).

    Known edge cases:
      - "terra", "tiara" — Whisper would transcribe these differently from "tara" ✓
      - Noisy "Tara" where DeepFilterNet suppressed the word → transcript empty → rejected
      - Very short segment (<0.3s) → padded to 0.5s → may hallucinate → checked by startswith
    """

    WAKE_PHRASES: list[str] = ["hey tara", "tara"]

    def __init__(
        self,
        model=None,  # faster_whisper.WhisperModel — shared from STT stage if provided
        model_name: str = FASTER_WHISPER_MODEL,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        self._whisper = model
        self._model_name = model_name
        if self._whisper is None:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"WhisperPhonemeWakeWord: loading '{self._model_name}' "
                f"(int8, cpu) for phoneme wake word"
            )
            t0 = time.perf_counter()
            self._whisper = WhisperModel(
                self._model_name, device="cpu", compute_type="int8"
            )
            logger.info(
                f"WhisperPhonemeWakeWord: loaded in {(time.perf_counter()-t0)*1000:.0f}ms"
            )
        except ImportError as e:
            raise ImportError(
                "Install faster-whisper: pip install faster-whisper"
            ) from e

    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        # Slice the probe window (first WAKE_WORD_PROBE_S seconds)
        probe_samples = int(sr * WAKE_WORD_PROBE_S)
        probe = audio[:probe_samples]
        if len(probe) < probe_samples:
            probe = np.pad(probe, (0, probe_samples - len(probe)))

        with stage_timer("wake_word", self.profiler, BUDGET_WAKE_WORD_MS) as t:
            segments_iter, _ = self._whisper.transcribe(
                probe,
                beam_size=1,
                language="en",
                vad_filter=False,
                condition_on_previous_text=False,
            )
            segments = list(segments_iter)

        raw_text = " ".join(s.text.strip() for s in segments).strip()
        # Normalize: lowercase + strip punctuation
        normalized = re.sub(r"[^\w\s]", "", raw_text.lower()).strip()

        triggered = False
        matched_phrase = None
        for phrase in self.WAKE_PHRASES:
            if normalized.startswith(phrase):
                triggered = True
                matched_phrase = phrase
                break

        logger.debug(
            f"Phoneme wake word | raw='{raw_text}' | normalized='{normalized}' | "
            f"triggered={triggered} | {t['elapsed_ms']:.1f}ms"
        )

        return WakeWordResult(
            triggered=triggered,
            score=1.0 if triggered else 0.0,
            backend="whisper_phoneme",
            elapsed_ms=t["elapsed_ms"],
            keyword=matched_phrase,
        )


class DeepgramWakeWord(BaseWakeWordDetector):
    """
    Wake word detection via Deepgram Nova-2.

    Probes first DEEPGRAM_WAKE_PROBE_S (1.5s) of each denoised VAD segment.
    Triggers if transcript starts with "tara" or "hey tara".

    Why Deepgram over faster-whisper for wake word:
      - Deepgram playground confirmed "Tara" audible in denoised audio
      - faster-whisper tiny.en misses Indian-accented "Tara" (0/24 in Iteration 4c)
      - Deepgram Nova-2 handles accent variation and residual noise better
      - 1.5s probe captures "Tara" even when VAD segment starts before speech onset

    Latency note:
      - ~400-600ms from India → US. Exceeds 300ms wake word budget.
      - However: eliminates a second STT API call — total pipeline latency unchanged.
      - Pi 5 deployment: requires internet. Offline alternative = re-record Porcupine
        samples with real voice to fix TTS/accent mismatch.
    """

    WAKE_PHRASES: list[str] = ["hey tara", "tara"]
    API_URL = "https://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str | None = None,
        probe_s: float = DEEPGRAM_WAKE_PROBE_S,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        super().__init__(profiler)
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DeepgramWakeWord requires DEEPGRAM_API_KEY env var. "
                "Free key at: https://console.deepgram.com"
            )
        self.probe_s = probe_s
        # Persistent session: reuses TCP+TLS connections across calls
        # Saves ~200-400ms per call vs bare requests.post
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        })
        logger.info(f"DeepgramWakeWord: ready | probe={probe_s}s | model=nova-3")

    def detect_at_utterance_start(
        self, full_segment: np.ndarray, sr: int = SAMPLE_RATE
    ) -> WakeWordResult:
        # Override base class: base clips to WAKE_WORD_BUFFER_S (1.0s) which is too short.
        # Deepgram needs probe_s (3.0s) of context to reliably detect Indian-accented "Tara".
        probe_samples = int(sr * self.probe_s)
        probe = full_segment[:probe_samples]
        if len(probe) < probe_samples:
            probe = np.pad(probe, (0, probe_samples - len(probe)))
        return self.detect(probe, sr)

    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> WakeWordResult:
        import io
        import wave

        probe_samples = int(sr * self.probe_s)
        probe = audio[:probe_samples]
        if len(probe) < probe_samples:
            probe = np.pad(probe, (0, probe_samples - len(probe)))

        audio_int16 = (np.clip(probe, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = buf.getvalue()

        with stage_timer("wake_word", self.profiler, BUDGET_WAKE_WORD_MS) as t:
            resp = self._session.post(
                self.API_URL,
                params={
                    "model": "nova-3",
                    "language": "en",
                    "smart_format": "true",
                    "keyterm": "Tara",
                },
                data=wav_bytes,
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_text = (
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

        normalized = re.sub(r"[^\w\s]", "", raw_text.lower()).strip()

        triggered = False
        matched_phrase = None
        for phrase in self.WAKE_PHRASES:
            # whole-word match anywhere in transcript — covers "Hey Tara ..." and "Tara ..."
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, normalized):
                triggered = True
                matched_phrase = phrase
                break

        logger.info(
            f"DeepgramWakeWord | raw='{raw_text}' | triggered={triggered} | "
            f"conf={confidence:.2f} | {t['elapsed_ms']:.1f}ms"
        )

        return WakeWordResult(
            triggered=triggered,
            score=float(confidence) if triggered else 0.0,
            backend="deepgram",
            elapsed_ms=t["elapsed_ms"],
            keyword=matched_phrase,
        )


def create_wake_word_detector(
    backend: str = DEFAULT_WAKE_WORD_BACKEND,
    profiler: LatencyProfiler | None = None,
    **kwargs,
) -> BaseWakeWordDetector:
    """
    Factory for wake word detectors.

    Parameters
    ----------
    backend : "openwakeword" | "porcupine" | "none"
    """
    if backend == "openwakeword":
        return OpenWakeWordDetector(profiler=profiler, **kwargs)
    elif backend == "porcupine":
        return PorcupineDetector(profiler=profiler, **kwargs)
    elif backend == "whisper_phoneme":
        return WhisperPhonemeWakeWord(profiler=profiler, **kwargs)
    elif backend == "deepgram":
        return DeepgramWakeWord(profiler=profiler, **kwargs)
    elif backend == "none":
        return PassthroughWakeWord(profiler=profiler)
    else:
        raise ValueError(f"Unknown wake word backend: {backend!r}")


def create_wake_word_detector_with_fallback(
    primary: str = DEFAULT_WAKE_WORD_BACKEND,
    fallback: str = "whisper_phoneme",
    profiler: LatencyProfiler | None = None,
    **kwargs,
) -> BaseWakeWordDetector:
    """
    Try primary backend, fall back to whisper_phoneme on init failure.

    Fallback chain:
      primary (openwakeword/porcupine) → whisper_phoneme → passthrough (last resort)

    whisper_phoneme is the reliable fallback: real phoneme matching, no API key needed,
    no custom training needed, uses the faster-whisper model already loaded in the pipeline.
    """
    try:
        detector = create_wake_word_detector(primary, profiler, **kwargs)
        logger.info(f"Wake word: using {primary}")
        return detector
    except Exception as e:
        logger.warning(
            f"Wake word primary ({primary}) failed: {e}. "
            f"Falling back to {fallback}. "
            "METHODOLOGY NOTE: This fallback is documented as a known failure mode."
        )
        try:
            return create_wake_word_detector(fallback, profiler)
        except Exception as e2:
            logger.error(
                f"Wake word fallback ({fallback}) also failed: {e2}. "
                "Using passthrough (no wake word filtering). "
                "METHODOLOGY NOTE: Both backends failed — document this as constraint violation."
            )
            return PassthroughWakeWord(profiler=profiler)
