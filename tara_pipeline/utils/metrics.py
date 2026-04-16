"""
Latency profiler — per-stage timer and report generator.
Uses time.perf_counter() exclusively (not time.time()).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
from loguru import logger
from tabulate import tabulate

from tara_pipeline.config import LATENCY_BUDGET


@dataclass
class StageResult:
    """Output + timing from a single pipeline stage."""
    stage_name: str
    elapsed_ms: float
    budget_ms: int
    over_budget: bool = field(init=False)

    def __post_init__(self) -> None:
        self.over_budget = self.elapsed_ms > self.budget_ms


class LatencyProfiler:
    """
    Accumulates per-stage timings across multiple pipeline runs.
    Call .report() to print the full latency table.
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._timings[stage].append(elapsed_ms)
        if elapsed_ms > self._budget_for(stage):
            logger.warning(
                f"OVER BUDGET | {stage}: {elapsed_ms:.1f}ms "
                f"(budget={self._budget_for(stage)}ms)"
            )
        else:
            logger.debug(f"{stage}: {elapsed_ms:.1f}ms")

    def _budget_for(self, stage: str) -> int:
        budgets = {
            "noise_suppression": LATENCY_BUDGET.noise_suppression_ms,
            "vad": LATENCY_BUDGET.vad_ms,
            "wake_word": LATENCY_BUDGET.wake_word_ms,
            "stt": LATENCY_BUDGET.stt_ms,
        }
        return budgets.get(stage, 9999)

    def report(self) -> str:
        """Return formatted latency table string and print it."""
        rows = []
        total_avgs = []

        stage_order = ["noise_suppression", "vad", "wake_word", "stt"]
        for stage in stage_order:
            if stage not in self._timings:
                continue
            times = self._timings[stage]
            avg = np.mean(times)
            p95 = np.percentile(times, 95) if len(times) > 1 else times[0]
            budget = self._budget_for(stage)
            flag = "OVER" if avg > budget else "OK"
            rows.append([
                stage.replace("_", " ").title(),
                f"{avg:.0f}",
                f"{p95:.0f}",
                str(budget),
                flag,
            ])
            total_avgs.append(avg)

        # Add other stages not in standard order
        for stage, times in self._timings.items():
            if stage not in stage_order:
                avg = np.mean(times)
                p95 = np.percentile(times, 95) if len(times) > 1 else times[0]
                rows.append([
                    stage.replace("_", " ").title(),
                    f"{avg:.0f}",
                    f"{p95:.0f}",
                    "-",
                    "-",
                ])
                total_avgs.append(avg)

        total_avg = sum(total_avgs)
        total_flag = "OVER" if total_avg > LATENCY_BUDGET.total_ms else "OK"
        rows.append([
            "TOTAL",
            f"{total_avg:.0f}",
            "-",
            str(LATENCY_BUDGET.total_ms),
            total_flag,
        ])

        table = tabulate(
            rows,
            headers=["Stage", "Avg (ms)", "P95 (ms)", "Budget (ms)", "Status"],
            tablefmt="github",
        )
        logger.info(f"\nLatency Report:\n{table}")
        return table

    def all_timings(self) -> dict[str, list[float]]:
        return dict(self._timings)


@contextmanager
def stage_timer(
    stage_name: str,
    profiler: LatencyProfiler | None = None,
    budget_ms: int | None = None,
) -> Generator[dict, None, None]:
    """
    Context manager that times a pipeline stage.

    Usage
    -----
    with stage_timer("stt", profiler) as t:
        result = model.transcribe(audio)
    print(t["elapsed_ms"])
    """
    timing: dict = {}
    t_start = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        timing["elapsed_ms"] = elapsed_ms
        timing["stage"] = stage_name
        if profiler is not None:
            profiler.record(stage_name, elapsed_ms)
        if budget_ms is not None and elapsed_ms > budget_ms:
            logger.warning(
                f"[{stage_name}] {elapsed_ms:.1f}ms exceeds budget {budget_ms}ms"
            )


# Global profiler instance — pipeline imports this
_global_profiler = LatencyProfiler()


def get_profiler() -> LatencyProfiler:
    return _global_profiler


def reset_profiler() -> None:
    global _global_profiler
    _global_profiler = LatencyProfiler()
