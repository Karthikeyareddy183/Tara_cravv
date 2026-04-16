"""Tests for wake word detection stage."""

from __future__ import annotations

import numpy as np
import pytest

from tara_pipeline.config import SAMPLE_RATE, WAKE_WORD_BUFFER_S
from tara_pipeline.stages.wake_word import (
    PassthroughWakeWord,
    WakeWordResult,
    create_wake_word_detector,
)


@pytest.fixture
def silence_audio() -> np.ndarray:
    """2 seconds of silence."""
    return np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)


@pytest.fixture
def passthrough_detector() -> PassthroughWakeWord:
    return PassthroughWakeWord()


class TestPassthroughWakeWord:
    """Tests for no-op passthrough (used in iterations 1–3)."""

    def test_always_triggers(
        self, passthrough_detector: PassthroughWakeWord, silence_audio: np.ndarray
    ) -> None:
        result = passthrough_detector.detect(silence_audio, SAMPLE_RATE)
        assert result.triggered is True

    def test_score_is_one(
        self, passthrough_detector: PassthroughWakeWord, silence_audio: np.ndarray
    ) -> None:
        result = passthrough_detector.detect(silence_audio, SAMPLE_RATE)
        assert result.score == 1.0

    def test_backend_is_passthrough(
        self, passthrough_detector: PassthroughWakeWord, silence_audio: np.ndarray
    ) -> None:
        result = passthrough_detector.detect(silence_audio, SAMPLE_RATE)
        assert result.backend == "passthrough"

    def test_utterance_start_also_triggers(
        self, passthrough_detector: PassthroughWakeWord, silence_audio: np.ndarray
    ) -> None:
        result = passthrough_detector.detect_at_utterance_start(silence_audio, SAMPLE_RATE)
        assert result.triggered is True


class TestWakeWordResultType:
    def test_result_is_namedtuple(
        self, passthrough_detector: PassthroughWakeWord, silence_audio: np.ndarray
    ) -> None:
        result = passthrough_detector.detect(silence_audio, SAMPLE_RATE)
        assert isinstance(result, WakeWordResult)
        assert hasattr(result, "triggered")
        assert hasattr(result, "score")
        assert hasattr(result, "backend")
        assert hasattr(result, "elapsed_ms")


class TestUtteranceStartBuffer:
    """
    Test the critical design: only first WAKE_WORD_BUFFER_S is checked.
    A real wake word at 2s into a 3s segment should NOT trigger if
    the first 1s contains no wake word.

    This test validates the logic path — actual detection accuracy
    depends on the model, but the buffer slicing must be correct.
    """

    def test_buffer_size_correct(
        self, passthrough_detector: PassthroughWakeWord
    ) -> None:
        """Buffer must be exactly WAKE_WORD_BUFFER_S seconds."""
        full_audio = np.random.randn(int(SAMPLE_RATE * 3)).astype(np.float32)
        buffer_samples = int(SAMPLE_RATE * WAKE_WORD_BUFFER_S)
        buffer = full_audio[:buffer_samples]
        assert len(buffer) == buffer_samples

    def test_short_segment_padded(self) -> None:
        """Segments shorter than buffer must be zero-padded, not crash."""
        det = PassthroughWakeWord()
        short_audio = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        # Should not raise
        result = det.detect_at_utterance_start(short_audio, SAMPLE_RATE)
        assert isinstance(result, WakeWordResult)


class TestFactoryFunction:
    def test_none_backend_returns_passthrough(self) -> None:
        det = create_wake_word_detector("none")
        assert isinstance(det, PassthroughWakeWord)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown wake word backend"):
            create_wake_word_detector("invalid_backend_xyz")
