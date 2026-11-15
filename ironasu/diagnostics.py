"""Stage-level bottleneck reports for layer-streaming inference measurements.

Durations should come from the *same* measured interval. Concurrent disk/compute
may overlap, so stage shares are descriptive, not additive fractions of wall time.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import time
from collections import defaultdict
from typing import Iterator

STAGES = ("disk_io", "transfer", "compute", "prefill", "queue", "other")
ADVICE = {
    "disk_io": "Measure NVMe read bandwidth/cache effects; compare AirLLM's existing prefetching and supported weight compression.",
    "transfer": "Profile host-device copies; verify PCIe/USB4/OCuLink link and pinned-memory constraints before changing hardware.",
    "compute": "Try smaller/quantized models or conditional computation, then recheck code-task quality; streamed weights alone do not save dense FLOPs.",
    "prefill": "Reduce prompt context using explicitly opted-in code retrieval; compare coding-task accuracy before enabling by default.",
    "queue": "Limit concurrent requests, tune queue thresholds, or add capacity; do not increase batch size without VRAM measurement.",
    "other": "Add finer stage instrumentation before selecting an optimization.",
}


@dataclass(frozen=True)
class StageEvent:
    stage: str
    seconds: float


class PhaseTrace:
    def __init__(self) -> None:
        self.events: list[StageEvent] = []

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        started = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(StageEvent(stage, time.perf_counter() - started))

    def report(self, wall_seconds: float, min_share: float = 0.40) -> dict:
        return diagnose(self.events, wall_seconds, min_share)


def diagnose(events: list[StageEvent], wall_seconds: float, min_share: float = 0.40) -> dict:
    if wall_seconds <= 0 or not 0 < min_share <= 1:
        raise ValueError("wall_seconds must be positive and min_share in (0,1]")
    totals = defaultdict(float)
    for event in events:
        if event.stage not in STAGES or event.seconds < 0:
            raise ValueError("invalid trace stage or duration")
        totals[event.stage] += event.seconds
    ranked = sorted(totals.items(), key=lambda p: (-p[1], p[0]))
    major_stage = ranked[0][0] if ranked and ranked[0][1] / wall_seconds >= min_share else None
    return {
        "wall_seconds": round(wall_seconds, 6),
        "observed_seconds_by_stage": {stage: round(value, 6) for stage, value in ranked},
        "bottleneck": major_stage or "unresolved",
        "largest_stage_wall_ratio": round(ranked[0][1] / wall_seconds, 4) if ranked else 0.0,
        "advice": ADVICE[major_stage] if major_stage else "No stage reaches the threshold. Instrument more phases; do not guess.",
        "warning": "Overlapping stages are not additive. Ratios are not percentages of FLOPs, SSD bytes, or saved compute.",
    }


def from_json(source: dict) -> dict:
    """Use external AirLLM/OS trace data; never pretend missing phases were measured."""
    if not isinstance(source, dict) or not isinstance(source.get("events"), list):
        raise ValueError("trace needs events array")
    return diagnose(
        [StageEvent(stage=e["stage"], seconds=float(e["seconds"])) for e in source["events"]],
        float(source["wall_seconds"]),
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose a measured ironasu phase trace")
    parser.add_argument("trace", help="JSON containing wall_seconds and [{stage,seconds}] events")
    args = parser.parse_args()
    with open(args.trace, "r", encoding="utf-8") as fh:
        report = from_json(json.load(fh))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
