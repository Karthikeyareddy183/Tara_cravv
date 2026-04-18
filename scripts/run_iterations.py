"""
Run all 4 iterations sequentially against the real FLAC file.
Captures output for methodology documentation.

Usage:
    python scripts/run_iterations.py
    python scripts/run_iterations.py --iterations 1 2 3 4
    python scripts/run_iterations.py --audio path/to/file.flac

Logs saved to:
    logs/iteration_1.log ... logs/iteration_4.log
    docs/iteration_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tara_pipeline.pipeline import TaraPipeline, PipelineResult
from tara_pipeline.utils.metrics import get_profiler, reset_profiler
from tara_pipeline.config import ASSETS_DIR, DEFAULT_WAKE_WORD_BACKEND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all pipeline iterations")
    parser.add_argument(
        "--audio",
        type=Path,
        default=ASSETS_DIR / "tara_assignment_recording_clipped.flac",
        help="Path to audio file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="Which iterations to run",
    )
    parser.add_argument(
        "--wake-word-backend",
        choices=["openwakeword", "porcupine", "whisper_phoneme", "deepgram", "none"],
        default=DEFAULT_WAKE_WORD_BACKEND,
    )
    return parser.parse_args()


ITERATION_DESCRIPTIONS = {
    1: "Baseline: raw Whisper base on full clip (no preprocessing, expected to fail)",
    2: "Baseline+: noisereduce spectral gating → Whisper base (expected to fail on impulsive noise)",
    3: "Improved: DeepFilterNet + Silero VAD + faster-whisper tiny.en (no wake word gate)",
    4: "Final: DeepFilterNet + Silero VAD + openWakeWord/Porcupine + faster-whisper tiny.en",
}


def run_iteration(
    iteration: int,
    audio_path: Path,
    wake_word_backend: str = DEFAULT_WAKE_WORD_BACKEND,
) -> dict:
    """Run a single iteration and return results dict for methodology doc."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: {ITERATION_DESCRIPTIONS[iteration]}")
    print(f"{'='*70}")

    reset_profiler()

    # Configure log file
    log_file = Path("logs") / f"iteration_{iteration}.log"
    log_file.parent.mkdir(exist_ok=True)
    log_id = logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation=None,
        mode="w",  # overwrite per run
    )

    t0 = time.perf_counter()
    error_msg = None

    try:
        pipeline = TaraPipeline(
            iteration=iteration,
            wake_word_backend=wake_word_backend if iteration == 4 else "none",
        )
        result: PipelineResult = pipeline.run(audio_path)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Iteration {iteration} FAILED: {e}")
        result = None
    finally:
        elapsed_total = (time.perf_counter() - t0) * 1000
        logger.remove(log_id)

    profiler = get_profiler()
    stage_timings = profiler.all_timings()

    # Print latency table
    print(profiler.report())

    if result:
        print(result.summary())

    # Build result dict
    iteration_data = {
        "iteration": iteration,
        "description": ITERATION_DESCRIPTIONS[iteration],
        "audio": str(audio_path),
        "error": error_msg,
        "total_elapsed_ms": elapsed_total,
        "commands": [],
        "vad_segments": result.vad_segment_count if result else 0,
        "wake_word_triggers": result.wake_word_trigger_count if result else 0,
        "wake_word_rejects": result.wake_word_reject_count if result else 0,
        "stage_timings": {
            stage: {
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "n": len(times),
            }
            for stage, times in stage_timings.items()
        },
    }

    if result:
        for cmd in result.commands:
            iteration_data["commands"].append({
                "transcript": cmd.transcript,
                "start_s": cmd.segment_start_s,
                "end_s": cmd.segment_end_s,
                "total_ms": cmd.total_ms,
                "over_budget": cmd.over_budget,
                "wake_word_score": cmd.wake_word_score,
                "wake_word_backend": cmd.wake_word_backend,
                "timings": cmd.timings,
            })

    print(f"\n[Iteration {iteration}] Done in {elapsed_total:.0f}ms total")
    print(f"Log saved to: {log_file}")
    return iteration_data


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")

    if not args.audio.exists():
        logger.error(f"Audio not found: {args.audio}")
        sys.exit(1)

    all_results = []

    for iteration in args.iterations:
        data = run_iteration(
            iteration=iteration,
            audio_path=args.audio,
            wake_word_backend=args.wake_word_backend,
        )
        all_results.append(data)

    # Save all results for methodology doc
    output_path = Path("docs") / "iteration_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL ITERATIONS COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Print comparison summary
    print("\nIteration Comparison:")
    print(f"{'Iter':<6} {'Commands':<10} {'Avg STT ms':<12} {'Status'}")
    print("-" * 40)
    for data in all_results:
        n_cmds = len(data["commands"])
        stt_timings = data["stage_timings"].get("stt", {})
        avg_stt = f"{stt_timings.get('avg_ms', 0):.0f}" if stt_timings else "N/A"
        error = f"ERROR: {data['error'][:30]}" if data.get("error") else "OK"
        print(f"{data['iteration']:<6} {n_cmds:<10} {avg_stt:<12} {error}")


if __name__ == "__main__":
    main()
