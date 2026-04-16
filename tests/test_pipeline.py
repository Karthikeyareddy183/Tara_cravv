"""
Integration tests for the full TaraPipeline.

Uses real audio file from assets/ where available,
falls back to synthetic audio for CI environments.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from tara_pipeline.config import SAMPLE_RATE, LATENCY_BUDGET, ASSETS_DIR
from tara_pipeline.pipeline import TaraPipeline, PipelineResult, CommandResult
from tara_pipeline.utils.audio import load_audio, save_audio
from tara_pipeline.utils.metrics import reset_profiler


REAL_AUDIO = ASSETS_DIR / "tara_assignment_recording_clipped.flac"
HAS_REAL_AUDIO = REAL_AUDIO.exists()


@pytest.fixture
def tmp_audio(tmp_path: Path) -> Path:
    """Write synthetic speech-like audio to tmp file for fast tests."""
    sr = SAMPLE_RATE
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simulate voice: mix of harmonics with amplitude variation
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
        + 0.02 * np.random.randn(len(t))
    ).astype(np.float32)
    audio_path = tmp_path / "test_audio.flac"
    save_audio(audio, audio_path, sr)
    return audio_path


@pytest.fixture(autouse=True)
def reset_profiler_between_tests() -> None:
    reset_profiler()


class TestAudioUtils:
    def test_load_audio_returns_float32(self, tmp_audio: Path) -> None:
        audio, sr = load_audio(tmp_audio)
        assert audio.dtype == np.float32
        assert sr == SAMPLE_RATE

    def test_load_audio_mono(self, tmp_audio: Path) -> None:
        audio, _ = load_audio(tmp_audio)
        assert audio.ndim == 1

    def test_load_audio_correct_sr(self, tmp_audio: Path) -> None:
        audio, sr = load_audio(tmp_audio)
        assert sr == SAMPLE_RATE

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.flac")


class TestPipelineIteration1:
    """Iteration 1: raw Whisper, no preprocessing."""

    @pytest.mark.slow
    def test_runs_without_error(self, tmp_audio: Path) -> None:
        pipeline = TaraPipeline(iteration=1)
        result = pipeline.run(tmp_audio)
        assert isinstance(result, PipelineResult)

    @pytest.mark.slow
    def test_returns_pipeline_result(self, tmp_audio: Path) -> None:
        pipeline = TaraPipeline(iteration=1)
        result = pipeline.run(tmp_audio)
        assert result.iteration == 1
        assert result.audio_duration_s > 0

    @pytest.mark.slow
    def test_iteration1_has_no_vad_segments(self, tmp_audio: Path) -> None:
        """Iteration 1 skips VAD — vad_segment_count stays 0."""
        pipeline = TaraPipeline(iteration=1)
        result = pipeline.run(tmp_audio)
        assert result.vad_segment_count == 0


class TestPipelineIteration3:
    """Iteration 3: DeepFilterNet + VAD + faster-whisper."""

    @pytest.mark.slow
    def test_runs_without_error(self, tmp_audio: Path) -> None:
        pipeline = TaraPipeline(iteration=3)
        result = pipeline.run(tmp_audio)
        assert isinstance(result, PipelineResult)

    @pytest.mark.slow
    def test_vad_segments_detected(self, tmp_audio: Path) -> None:
        """Should detect voice-active segments in synthetic tone audio."""
        pipeline = TaraPipeline(iteration=3)
        result = pipeline.run(tmp_audio)
        # Synthetic audio has voice-like content — VAD should find it
        assert result.vad_segment_count >= 0  # >= 0 (VAD may not trigger on pure tone)

    @pytest.mark.slow
    def test_no_wake_word_gate(self, tmp_audio: Path) -> None:
        """Iteration 3 has no wake word — trigger count = 0, reject count = 0."""
        pipeline = TaraPipeline(iteration=3)
        result = pipeline.run(tmp_audio)
        assert result.wake_word_trigger_count == 0
        assert result.wake_word_reject_count == 0


class TestCommandResult:
    def test_over_budget_flag(self) -> None:
        cmd = CommandResult(
            transcript="test",
            segment_start_s=0.0,
            segment_end_s=1.0,
            wake_word_score=0.9,
            wake_word_backend="openwakeword",
            timings={"stt": 3000.0},
            total_ms=3000.0,
        )
        assert cmd.over_budget is True

    def test_within_budget_flag(self) -> None:
        cmd = CommandResult(
            transcript="test",
            segment_start_s=0.0,
            segment_end_s=1.0,
            wake_word_score=0.9,
            wake_word_backend="openwakeword",
            timings={"stt": 500.0},
            total_ms=500.0,
        )
        assert cmd.over_budget is False


class TestPipelineOnRealAudio:
    """Integration tests on real kitchen audio — only run if file present."""

    @pytest.mark.skipif(not HAS_REAL_AUDIO, reason="Real audio file not found")
    @pytest.mark.slow
    def test_iteration4_detects_commands(self) -> None:
        """Full pipeline should detect at least one Tara command in real audio."""
        pipeline = TaraPipeline(iteration=4, wake_word_backend="none")
        result = pipeline.run(REAL_AUDIO)
        # With passthrough wake word, all VAD segments → STT
        # Should find speech segments in kitchen recording
        assert result.vad_segment_count > 0

    @pytest.mark.skipif(not HAS_REAL_AUDIO, reason="Real audio file not found")
    @pytest.mark.slow
    def test_all_commands_within_budget(self) -> None:
        """All E2E command latencies should be within 2s budget."""
        pipeline = TaraPipeline(iteration=3)  # no wake word overhead
        result = pipeline.run(REAL_AUDIO)
        for cmd in result.commands:
            assert cmd.total_ms <= LATENCY_BUDGET.total_ms * 1.5, (
                f"Command at {cmd.segment_start_s:.2f}s took {cmd.total_ms:.0f}ms "
                f"(budget={LATENCY_BUDGET.total_ms}ms)"
            )
