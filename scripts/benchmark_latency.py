"""
Benchmark latency across all pipeline stages and all detected triggers.

Usage:
    python scripts/benchmark_latency.py assets/tara_assignment_recording_clipped.flac

Output: per-stage latency table (Avg, P95, Budget) across all "Hey Tara"/"Tara" instances.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tabulate import tabulate
from tara_pipeline.pipeline import TaraPipeline
from tara_pipeline.utils.metrics import reset_profiler, get_profiler
from tara_pipeline.config import LATENCY_BUDGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tara pipeline latency benchmark")
    parser.add_argument("audio", type=Path, help="Audio file to benchmark")
    parser.add_argument(
        "--iteration",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of pipeline runs to average over",
    )
    parser.add_argument(
        "--wake-word-backend",
        choices=["openwakeword", "porcupine", "none"],
        default="openwakeword",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save results as JSON",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # suppress verbose during benchmark

    if not args.audio.exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"\nBenchmarking Iteration {args.iteration} | {args.runs} runs | {args.audio.name}")
    print("=" * 60)

    # Accumulate timings across runs
    all_stage_timings: dict[str, list[float]] = defaultdict(list)
    all_command_totals: list[float] = []
    all_results = []

    # Build pipeline once (model loading not counted in benchmark)
    pipeline = TaraPipeline(
        iteration=args.iteration,
        wake_word_backend=args.wake_word_backend,
    )

    for run_idx in range(args.runs):
        reset_profiler()
        result = pipeline.run(args.audio)
        profiler = get_profiler()

        for stage, timings in profiler.all_timings().items():
            all_stage_timings[stage].extend(timings)

        for cmd in result.commands:
            all_command_totals.append(cmd.total_ms)

        all_results.append(result)
        print(f"  Run {run_idx+1}/{args.runs}: {len(result.commands)} commands detected")

    # Build latency table
    stage_order = ["noise_suppression", "vad", "wake_word", "stt"]
    budget_map = {
        "noise_suppression": LATENCY_BUDGET.noise_suppression_ms,
        "vad": LATENCY_BUDGET.vad_ms,
        "wake_word": LATENCY_BUDGET.wake_word_ms,
        "stt": LATENCY_BUDGET.stt_ms,
    }

    rows = []
    total_avg = 0.0

    for stage in stage_order:
        if stage not in all_stage_timings:
            continue
        times = all_stage_timings[stage]
        avg = np.mean(times)
        p95 = np.percentile(times, 95) if len(times) > 1 else times[0]
        p50 = np.percentile(times, 50) if len(times) > 1 else times[0]
        budget = budget_map.get(stage, "-")
        status = "OVER" if avg > budget else "OK"
        rows.append([
            stage.replace("_", " ").title(),
            f"{avg:.0f}",
            f"{p50:.0f}",
            f"{p95:.0f}",
            str(budget),
            status,
        ])
        total_avg += avg

    # Total row
    total_status = "OVER" if total_avg > LATENCY_BUDGET.total_ms else "OK"
    rows.append([
        "TOTAL (sum of avg)",
        f"{total_avg:.0f}",
        "-",
        "-",
        str(LATENCY_BUDGET.total_ms),
        total_status,
    ])

    print("\n")
    print(tabulate(
        rows,
        headers=["Stage", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Budget (ms)", "Status"],
        tablefmt="github",
    ))

    # Per-command breakdown
    if all_command_totals:
        print(f"\nPer-command E2E latency across all triggers ({len(all_command_totals)} total):")
        print(f"  Min:  {min(all_command_totals):.0f}ms")
        print(f"  Avg:  {np.mean(all_command_totals):.0f}ms")
        print(f"  P95:  {np.percentile(all_command_totals, 95):.0f}ms")
        print(f"  Max:  {max(all_command_totals):.0f}ms")
        over_budget = sum(1 for t in all_command_totals if t > LATENCY_BUDGET.total_ms)
        print(f"  Over budget: {over_budget}/{len(all_command_totals)}")

    # Transcript summary from last run
    print(f"\nTranscripts from last run ({len(all_results[-1].commands)} commands):")
    for i, cmd in enumerate(all_results[-1].commands, 1):
        print(f"  [{i}] {cmd.segment_start_s:.2f}s: {cmd.transcript!r}")

    # Save JSON if requested
    if args.output_json:
        output = {
            "iteration": args.iteration,
            "runs": args.runs,
            "audio": str(args.audio),
            "stage_timings": {
                stage: {
                    "avg_ms": float(np.mean(times)),
                    "p50_ms": float(np.percentile(times, 50)) if len(times) > 1 else times[0],
                    "p95_ms": float(np.percentile(times, 95)) if len(times) > 1 else times[0],
                    "budget_ms": budget_map.get(stage, None),
                    "n_samples": len(times),
                }
                for stage, times in all_stage_timings.items()
            },
            "e2e_latency": {
                "min_ms": float(min(all_command_totals)) if all_command_totals else None,
                "avg_ms": float(np.mean(all_command_totals)) if all_command_totals else None,
                "p95_ms": float(np.percentile(all_command_totals, 95)) if len(all_command_totals) > 1 else None,
                "max_ms": float(max(all_command_totals)) if all_command_totals else None,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
