"""Tests for Silero VAD stage."""

from __future__ import annotations

import numpy as np
import pytest

from tara_pipeline.config import SAMPLE_RATE
from tara_pipeline.stages.vad import SileroVAD, SpeechSegment


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    """Shared VAD instance — model loaded once per test session."""
    return SileroVAD()


@pytest.fixture
def silence_audio() -> np.ndarray:
    """3 seconds of silence."""
    return np.zeros(SAMPLE_RATE * 3, dtype=np.float32)


@pytest.fixture
def tone_audio() -> np.ndarray:
    """1 second 440Hz sine tone (simulates voice-like signal)."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def mixed_audio(silence_audio, tone_audio) -> np.ndarray:
    """1s silence + 1s tone + 1s silence."""
    return np.concatenate([silence_audio[:SAMPLE_RATE], tone_audio, silence_audio[:SAMPLE_RATE]])


class TestSileroVADInit:
    def test_loads_successfully(self, vad: SileroVAD) -> None:
        assert vad._model is not None
        assert vad._utils is not None

    def test_threshold_set(self, vad: SileroVAD) -> None:
        assert 0 < vad.threshold <= 1.0


class TestSileroVADDetection:
    def test_silence_returns_no_segments(self, vad: SileroVAD, silence_audio: np.ndarray) -> None:
        segments, elapsed_ms = vad.detect_segments(silence_audio, SAMPLE_RATE)
        assert len(segments) == 0
        assert elapsed_ms > 0

    def test_returns_elapsed_ms(self, vad: SileroVAD, silence_audio: np.ndarray) -> None:
        _, elapsed_ms = vad.detect_segments(silence_audio, SAMPLE_RATE)
        assert isinstance(elapsed_ms, float)
        assert elapsed_ms > 0

    def test_segments_are_speech_segment_type(
        self, vad: SileroVAD, mixed_audio: np.ndarray
    ) -> None:
        segments, _ = vad.detect_segments(mixed_audio, SAMPLE_RATE)
        for seg in segments:
            assert isinstance(seg, SpeechSegment)

    def test_segment_fields_valid(
        self, vad: SileroVAD, mixed_audio: np.ndarray
    ) -> None:
        segments, _ = vad.detect_segments(mixed_audio, SAMPLE_RATE)
        for seg in segments:
            assert seg.start_sample >= 0
            assert seg.end_sample > seg.start_sample
            assert seg.start_s >= 0.0
            assert seg.end_s > seg.start_s
            assert seg.duration_s > 0.0

    def test_elapsed_within_budget(self, vad: SileroVAD, mixed_audio: np.ndarray) -> None:
        from tara_pipeline.config import BUDGET_VAD_MS
        _, elapsed_ms = vad.detect_segments(mixed_audio, SAMPLE_RATE)
        # Allow 2x budget in test environment (no Pi 5 optimisation)
        assert elapsed_ms < BUDGET_VAD_MS * 2, (
            f"VAD too slow: {elapsed_ms:.0f}ms (budget={BUDGET_VAD_MS}ms)"
        )


class TestSileroVADExtract:
    def test_extract_returns_audio_arrays(
        self, vad: SileroVAD, mixed_audio: np.ndarray
    ) -> None:
        extracted, _ = vad.extract_segments(mixed_audio, SAMPLE_RATE)
        for audio_chunk, seg in extracted:
            assert isinstance(audio_chunk, np.ndarray)
            assert audio_chunk.dtype == np.float32
            assert len(audio_chunk) > 0

    def test_extract_audio_length_matches_segment(
        self, vad: SileroVAD, mixed_audio: np.ndarray
    ) -> None:
        extracted, _ = vad.extract_segments(mixed_audio, SAMPLE_RATE)
        for audio_chunk, seg in extracted:
            expected_len = seg.end_sample - seg.start_sample
            # Allow small padding difference
            assert abs(len(audio_chunk) - expected_len) <= SAMPLE_RATE * 0.1
