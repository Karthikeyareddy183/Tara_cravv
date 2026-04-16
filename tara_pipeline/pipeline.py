"""
Tara Pipeline orchestrator — coordinates all stages end-to-end.

Stages (in order):
  1. Noise Suppression  (DeepFilterNet or noisereduce)
  2. VAD                (Silero)
  3. Wake Word          (openWakeWord → Porcupine fallback)
  4. STT                (faster-whisper tiny.en)

Each stage returns its output + elapsed_ms.
Every stage failure is logged with stage name + latency at failure point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from tara_pipeline.config import (
    SAMPLE_RATE,
    LATENCY_BUDGET,
    WAKE_WORD_BUFFER_S,
    WAKE_WORD_PROBE_S,
)
from tara_pipeline.stages.noise_suppression import BaseNoiseSuppressor, create_suppressor
from tara_pipeline.stages.vad import SileroVAD, SpeechSegment
from tara_pipeline.stages.wake_word import BaseWakeWordDetector, create_wake_word_detector_with_fallback, WakeWordResult
from tara_pipeline.stages.stt import BaseSTT, TranscriptionResult, create_stt
from tara_pipeline.utils.audio import load_audio, split_on_silence_segments
from tara_pipeline.utils.metrics import LatencyProfiler, get_profiler, reset_profiler


@dataclass
class CommandResult:
    """One detected and transcribed Tara command."""
    transcript: str
    segment_start_s: float
    segment_end_s: float
    wake_word_score: float
    wake_word_backend: str
    timings: dict[str, float]  # stage → elapsed_ms
    total_ms: float

    @property
    def over_budget(self) -> bool:
        return self.total_ms > LATENCY_BUDGET.total_ms


@dataclass
class PipelineResult:
    """Full pipeline run result."""
    audio_path: str
    audio_duration_s: float
    commands: list[CommandResult] = field(default_factory=list)
    vad_segment_count: int = 0
    wake_word_trigger_count: int = 0
    wake_word_reject_count: int = 0
    total_run_ms: float = 0.0
    iteration: int = 4

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"TARA PIPELINE RESULT (Iteration {self.iteration})",
            f"{'='*60}",
            f"Audio: {self.audio_path} ({self.audio_duration_s:.1f}s)",
            f"VAD segments detected: {self.vad_segment_count}",
            f"Wake word triggers: {self.wake_word_trigger_count}",
            f"Wake word rejected: {self.wake_word_reject_count}",
            f"Commands transcribed: {len(self.commands)}",
            f"Total pipeline run: {self.total_run_ms:.0f}ms",
            f"{'='*60}",
        ]
        for i, cmd in enumerate(self.commands, 1):
            budget_flag = "OVER BUDGET" if cmd.over_budget else "OK"
            lines.append(
                f"[{i}] {cmd.segment_start_s:.2f}s–{cmd.segment_end_s:.2f}s | "
                f"{cmd.total_ms:.0f}ms ({budget_flag})"
            )
            lines.append(f"     TRANSCRIPT: {cmd.transcript!r}")
            lines.append(f"     TIMINGS: {cmd.timings}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class TaraPipeline:
    """
    End-to-end Tara voice command pipeline.

    Configurable per iteration:
      iteration=1 → raw whisper, no preprocessing
      iteration=2 → noisereduce + whisper base
      iteration=3 → deepfilternet + silero VAD + faster-whisper
      iteration=4 → deepfilternet + silero VAD + openWakeWord + faster-whisper
    """

    def __init__(
        self,
        iteration: int = 4,
        noise_mode: str = "deepfilternet",
        stt_backend: str = "faster_whisper",
        wake_word_backend: str = "openwakeword",
        profiler: LatencyProfiler | None = None,
    ) -> None:
        self._stt_backend_name = stt_backend
        self.iteration = iteration
        self.profiler = profiler or get_profiler()

        logger.info(f"Initialising TaraPipeline (Iteration {iteration})")
        logger.info(f"  noise_mode={noise_mode} | stt={stt_backend} | wake_word={wake_word_backend}")

        # Stage 1: Noise suppression
        actual_noise_mode = noise_mode if iteration >= 2 else "none"
        if iteration == 2:
            actual_noise_mode = "noisereduce"
        elif iteration >= 3:
            actual_noise_mode = "deepfilternet"

        self._suppressor: BaseNoiseSuppressor = create_suppressor(
            actual_noise_mode, self.profiler
        )

        # Stage 2: VAD (iterations 3+)
        self._vad: SileroVAD | None = None
        if iteration >= 3:
            self._vad = SileroVAD(profiler=self.profiler)

        # Stage 4: STT — created BEFORE wake word so model can be shared
        if iteration <= 2:
            actual_stt = "whisper"
            stt_kwargs: dict = {"model_name": "base"}
        elif stt_backend == "deepgram":
            actual_stt = "deepgram"
            stt_kwargs = {}
        else:
            actual_stt = "faster_whisper"
            stt_kwargs = {"model_name": "tiny.en"}
        self._stt: BaseSTT = create_stt(actual_stt, profiler=self.profiler, **stt_kwargs)

        # Stage 3: Wake word (iteration 4 only)
        # whisper_phoneme shares the STT model to avoid loading twice
        self._wake_word: BaseWakeWordDetector | None = None
        self._wake_word_backend = wake_word_backend
        if iteration >= 4:
            ww_kwargs: dict = {}
            if wake_word_backend == "whisper_phoneme" and hasattr(self._stt, "_model"):
                # Share the faster-whisper model — one instance, half the memory/warmup
                ww_kwargs["model"] = self._stt._model
                logger.info("WhisperPhonemeWakeWord: sharing STT model (no extra load)")
            self._wake_word = create_wake_word_detector_with_fallback(
                primary=wake_word_backend,
                fallback="whisper_phoneme",
                profiler=self.profiler,
                **ww_kwargs,
            )

        logger.info("TaraPipeline: all stages initialised")

    def run(self, audio_path: str | Path) -> PipelineResult:
        """
        Run full pipeline on an audio file.

        Returns PipelineResult with all detected commands and timings.
        """
        audio_path = Path(audio_path)
        t_run_start = time.perf_counter()

        # Load audio
        try:
            audio, sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Pipeline failed at audio load: {e}")
            raise

        audio_duration_s = len(audio) / sr
        result = PipelineResult(
            audio_path=str(audio_path),
            audio_duration_s=audio_duration_s,
            iteration=self.iteration,
        )

        # ── Iteration 1: Raw Whisper on full clip ────────────────────────────
        if self.iteration == 1:
            logger.info("Iteration 1: raw Whisper on full clip (no preprocessing)")
            try:
                transcript = self._stt.transcribe(audio, sr)
                result.commands.append(CommandResult(
                    transcript=transcript.text,
                    segment_start_s=0.0,
                    segment_end_s=audio_duration_s,
                    wake_word_score=1.0,
                    wake_word_backend="none",
                    timings={"stt": transcript.elapsed_ms},
                    total_ms=transcript.elapsed_ms,
                ))
            except Exception as e:
                logger.error(f"[stt] failed at {time.perf_counter() - t_run_start:.0f}ms: {e}")
                raise
            result.total_run_ms = (time.perf_counter() - t_run_start) * 1000
            logger.info(result.summary())
            return result

        # ── Stage 1: Noise suppression ───────────────────────────────────────
        try:
            audio_clean, ns_ms = self._suppressor.suppress(audio, sr)
        except Exception as e:
            logger.error(f"[noise_suppression] failed at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}")
            raise

        # ── Iteration 2: noisereduce → Whisper on full clip ──────────────────
        if self.iteration == 2:
            logger.info("Iteration 2: noisereduce → Whisper base on full clip")
            try:
                transcript = self._stt.transcribe(audio_clean, sr)
                result.commands.append(CommandResult(
                    transcript=transcript.text,
                    segment_start_s=0.0,
                    segment_end_s=audio_duration_s,
                    wake_word_score=1.0,
                    wake_word_backend="none",
                    timings={"noise_suppression": ns_ms, "stt": transcript.elapsed_ms},
                    total_ms=ns_ms + transcript.elapsed_ms,
                ))
            except Exception as e:
                logger.error(f"[stt] failed at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}")
                raise
            result.total_run_ms = (time.perf_counter() - t_run_start) * 1000
            logger.info(result.summary())
            return result

        # ── Stage 2: VAD (iterations 3+) ────────────────────────────────────
        try:
            seg_infos, vad_ms = self._vad.detect_segments(audio_clean, sr)
            segments_with_audio = [
                (audio_clean[s.start_sample:s.end_sample], s) for s in seg_infos
            ]
        except Exception as e:
            logger.error(f"[vad] failed at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}")
            raise

        result.vad_segment_count = len(seg_infos)
        logger.info(f"VAD: {len(seg_infos)} segments detected")

        if not segments_with_audio:
            logger.warning("VAD: no speech segments found in audio")
            result.total_run_ms = (time.perf_counter() - t_run_start) * 1000
            return result

        # ── Iteration 3: DeepFilterNet + VAD + faster-whisper (no wake word) ─
        if self.iteration == 3:
            logger.info("Iteration 3: DeepFilterNet + VAD + faster-whisper (no wake word gate)")
            for seg_audio, seg_info in segments_with_audio:
                try:
                    transcript = self._stt.transcribe(seg_audio, sr)
                    if transcript.text.strip():
                        result.commands.append(CommandResult(
                            transcript=transcript.text,
                            segment_start_s=seg_info.start_s,
                            segment_end_s=seg_info.end_s,
                            wake_word_score=1.0,
                            wake_word_backend="none",
                            timings={
                                "noise_suppression": ns_ms,
                                "vad": vad_ms,
                                "stt": transcript.elapsed_ms,
                            },
                            total_ms=ns_ms + vad_ms + transcript.elapsed_ms,
                        ))
                except Exception as e:
                    logger.error(
                        f"[stt] segment {seg_info.start_s:.2f}s failed "
                        f"at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}"
                    )
            result.total_run_ms = (time.perf_counter() - t_run_start) * 1000
            logger.info(result.summary())
            return result

        # ── Iteration 4: Full pipeline with wake word ────────────────────────
        logger.info("Iteration 4: Full pipeline — DeepFilterNet + VAD + WakeWord + STT")
        for seg_audio, seg_info in segments_with_audio:
            # Stage 3: Wake word — CHECK ONLY UTTERANCE START
            try:
                ww_result: WakeWordResult = self._wake_word.detect_at_utterance_start(
                    seg_audio, sr
                )
            except Exception as e:
                logger.error(
                    f"[wake_word] segment {seg_info.start_s:.2f}s failed "
                    f"at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}"
                )
                result.wake_word_reject_count += 1
                continue

            if not ww_result.triggered:
                result.wake_word_reject_count += 1
                logger.debug(
                    f"Wake word not detected in segment {seg_info.start_s:.2f}s–{seg_info.end_s:.2f}s "
                    f"(score={ww_result.score:.3f}) — discarding"
                )
                continue

            result.wake_word_trigger_count += 1
            logger.info(
                f"Wake word TRIGGERED at {seg_info.start_s:.2f}s | "
                f"score={ww_result.score:.3f} | backend={ww_result.backend}"
            )

            # Step: Extract command audio — clip wake word from segment
            # Only transcribe the command AFTER "Hey Tara"/"Tara", not the wake word itself
            # whisper_phoneme probe was WAKE_WORD_PROBE_S (0.5s)
            # OWW/porcupine checked WAKE_WORD_BUFFER_S (1.0s)
            if self._wake_word_backend == "whisper_phoneme":
                clip_samples = int(sr * WAKE_WORD_PROBE_S)
            else:
                clip_samples = int(sr * WAKE_WORD_BUFFER_S)
            # Only clip if enough audio remains after (>0.3s) — avoids empty STT on short segs
            min_command_samples = int(sr * 0.3)
            if len(seg_audio) > clip_samples + min_command_samples:
                command_audio = seg_audio[clip_samples:]
                logger.debug(
                    f"Segment {seg_info.start_s:.2f}s: clipped {clip_samples/sr:.1f}s "
                    f"wake word → {len(command_audio)/sr:.2f}s command audio"
                )
            else:
                # Segment too short to clip cleanly — STT full segment
                command_audio = seg_audio
                logger.debug(
                    f"Segment {seg_info.start_s:.2f}s: short segment ({len(seg_audio)/sr:.2f}s) "
                    f"— STT full audio without wake word clip"
                )

            # Stage 4: STT — transcribe command only (wake word already clipped)
            try:
                transcript = self._stt.transcribe(command_audio, sr)
            except Exception as e:
                logger.error(
                    f"[stt] segment {seg_info.start_s:.2f}s failed "
                    f"at {(time.perf_counter()-t_run_start)*1000:.0f}ms: {e}"
                )
                continue

            total_ms = ns_ms + vad_ms + ww_result.elapsed_ms + transcript.elapsed_ms

            result.commands.append(CommandResult(
                transcript=transcript.text,
                segment_start_s=seg_info.start_s,
                segment_end_s=seg_info.end_s,
                wake_word_score=ww_result.score,
                wake_word_backend=ww_result.backend,
                timings={
                    "noise_suppression": ns_ms,
                    "vad": vad_ms,
                    "wake_word": ww_result.elapsed_ms,
                    "stt": transcript.elapsed_ms,
                },
                total_ms=total_ms,
            ))

            if total_ms > LATENCY_BUDGET.total_ms:
                logger.warning(
                    f"TOTAL LATENCY OVER BUDGET: {total_ms:.0f}ms > "
                    f"{LATENCY_BUDGET.total_ms}ms"
                )

        result.total_run_ms = (time.perf_counter() - t_run_start) * 1000
        logger.info(result.summary())
        return result
