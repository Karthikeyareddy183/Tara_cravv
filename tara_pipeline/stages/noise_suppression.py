"""
Noise suppression stage.

Two implementations:
  1. NoisereduceSupressor  — spectral gating (Iteration 2 baseline, expected partial failure)
  2. DeepFilterNetSuppressor — neural noise suppression (Iteration 3+ final)

Both return (suppressed_audio: np.ndarray, elapsed_ms: float).

Pi 5 justification:
  - DeepFilterNet exports to ONNX → runs via onnxruntime on Pi 5 CPU/NPU
  - AI HAT+ provides NPU acceleration for ONNX workloads
  - Peak RAM ~50MB (ONNX model), within Pi 5's 8GB
  - noisereduce is pure Python/numpy — trivially Pi 5 compatible
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from tara_pipeline.config import (
    SAMPLE_RATE,
    DEEPFILTERNET_ATTN_LIMIT,
    DEEPFILTERNET_POST_FILTER,
    NOISEREDUCE_PROP_DECREASE,
    NOISEREDUCE_STATIONARY,
    BUDGET_NOISE_SUPPRESSION_MS,
)
from tara_pipeline.utils.metrics import LatencyProfiler, stage_timer


class BaseNoiseSuppressor(ABC):
    """Abstract base for all noise suppression implementations."""

    def __init__(self, profiler: LatencyProfiler | None = None) -> None:
        self.profiler = profiler

    @abstractmethod
    def suppress(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
        """
        Apply noise suppression.

        Returns
        -------
        suppressed_audio : np.ndarray float32
        elapsed_ms       : float
        """
        ...

    def __call__(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
        return self.suppress(audio, sr)


class NoisereduceSuppressor(BaseNoiseSuppressor):
    """
    Spectral gating noise reduction (Iteration 2 baseline).

    Known limitations (documented for methodology):
      - Handles stationary noise well (chimney fan hum)
      - Fails on non-stationary impulsive noise (pressure cooker whistle, chopping)
      - noisereduce estimates noise profile from first N frames — any impulsive noise
        that varies over time will partially survive
    """

    def __init__(self, profiler: LatencyProfiler | None = None) -> None:
        super().__init__(profiler)
        try:
            import noisereduce as nr
            self._nr = nr
            logger.info("NoisereduceSuppressor: loaded noisereduce")
        except ImportError as e:
            raise ImportError("Install noisereduce: pip install noisereduce") from e

    def suppress(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
        with stage_timer("noise_suppression", self.profiler, BUDGET_NOISE_SUPPRESSION_MS) as t:
            suppressed = self._nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=NOISEREDUCE_PROP_DECREASE,
                stationary=NOISEREDUCE_STATIONARY,
            )
        return suppressed.astype(np.float32), t["elapsed_ms"]


# Module-level cache — one model load per process lifetime
_DFN_MODEL = None
_DFN_STATE = None
_DFN_ENHANCE = None
_DFN_INIT_DF = None


class DeepFilterNetSuppressor(BaseNoiseSuppressor):
    """
    DeepFilterNet neural noise suppression (Iteration 3+ final pipeline).

    Why chosen over noisereduce:
      - Handles non-stationary noise: pressure cooker whistle, sizzling oil, chopping
      - noisereduce only models stationary noise (fan hum) via spectral gating
      - DeepFilterNet uses a deep filter bank trained on diverse noise types
      - ONNX export available → Pi 5 + AI HAT+ NPU acceleration

    Known failure modes:
      - Very short (<50ms) impulse bursts (whistle onset) may partially survive
      - Adds ~150ms latency (within 200ms budget)
    """

    def __init__(self, profiler: LatencyProfiler | None = None) -> None:
        super().__init__(profiler)
        self._load_model()

    @staticmethod
    def _patch_torchaudio_compat() -> None:
        """
        Compatibility shim for deepfilternet 0.5.6 on torchaudio >= 2.2.

        torchaudio.backend.common.AudioMetaData was fully removed in torchaudio 2.2+.
        deepfilternet 0.5.6 imports from that path.
        Solution: define AudioMetaData as a namedtuple ourselves — deepfilternet
        only uses it as a type container for sample_rate / num_frames / num_channels.
        """
        import sys
        import types
        from collections import namedtuple
        import torchaudio

        if "torchaudio.backend.common" not in sys.modules:
            logger.debug("Applying torchaudio compat patch for deepfilternet (torchaudio >= 2.2)")

            # Reconstruct AudioMetaData as a namedtuple matching the old torchaudio spec
            AudioMetaData = namedtuple(
                "AudioMetaData",
                ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            )

            backend_mod = types.ModuleType("torchaudio.backend")
            common_mod = types.ModuleType("torchaudio.backend.common")
            common_mod.AudioMetaData = AudioMetaData

            sys.modules["torchaudio.backend"] = backend_mod
            sys.modules["torchaudio.backend.common"] = common_mod
            backend_mod.common = common_mod

            if not hasattr(torchaudio, "backend"):
                torchaudio.backend = backend_mod
            else:
                torchaudio.backend.common = common_mod

    def _load_model(self) -> None:
        global _DFN_MODEL, _DFN_STATE, _DFN_ENHANCE, _DFN_INIT_DF
        if _DFN_MODEL is not None:
            self._model = _DFN_MODEL
            self._df_state = _DFN_STATE
            self._enhance = _DFN_ENHANCE
            logger.info("DeepFilterNetSuppressor: reusing cached model")
            return
        try:
            self._patch_torchaudio_compat()
            from df import enhance, init_df
            _DFN_ENHANCE = enhance
            _DFN_INIT_DF = init_df
            _DFN_MODEL, _DFN_STATE, _ = init_df()
            self._model = _DFN_MODEL
            self._df_state = _DFN_STATE
            self._enhance = _DFN_ENHANCE
            logger.info("DeepFilterNetSuppressor: model loaded and cached")
        except ImportError as e:
            raise ImportError(
                "Install deepfilternet: pip install deepfilternet"
            ) from e
        except Exception as e:
            logger.error(f"DeepFilterNet init failed: {e}")
            raise

    def suppress(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
        import torch

        # DeepFilterNet expects (1, N) tensor at model sample rate
        model_sr = self._df_state.sr()
        if sr != model_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, N)

        with stage_timer("noise_suppression", self.profiler, BUDGET_NOISE_SUPPRESSION_MS) as t:
            enhanced = self._enhance(self._model, self._df_state, audio_tensor)

        enhanced_np = enhanced.squeeze(0).numpy().astype(np.float32)

        # Resample back if model SR differs
        if sr != model_sr:
            import librosa
            enhanced_np = librosa.resample(enhanced_np, orig_sr=model_sr, target_sr=sr)

        return enhanced_np, t["elapsed_ms"]


def create_suppressor(
    mode: str = "deepfilternet",
    profiler: LatencyProfiler | None = None,
) -> BaseNoiseSuppressor:
    """
    Factory — create noise suppressor by mode.

    Parameters
    ----------
    mode : "deepfilternet" | "noisereduce" | "none"
    """
    if mode == "deepfilternet":
        return DeepFilterNetSuppressor(profiler)
    elif mode == "noisereduce":
        return NoisereduceSuppressor(profiler)
    elif mode == "none":
        return _PassthroughSuppressor(profiler)
    else:
        raise ValueError(f"Unknown suppressor mode: {mode!r}")


class _PassthroughSuppressor(BaseNoiseSuppressor):
    """No-op suppressor for Iteration 1 (raw baseline)."""

    def suppress(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
        return audio, 0.0
