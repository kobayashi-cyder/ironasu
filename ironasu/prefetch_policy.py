"""Bounded heuristic gate for one-stage speculative CPU weight prefetch.

Measured overlap is a hint, not a guarantee of end-to-end speedup.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadAheadObservation:
    used: bool
    read_seconds: float
    wait_seconds: float


class AdaptivePrefetchGate:
    """Temporarily disable unhelpful prefetch and periodically probe again."""

    def __init__(self, *, window: int = 16, cooldown: int = 8,
                 minimum_overlap: float = .10, maximum_waste: float = .25):
        if type(window) is not int or not 2 <= window <= 1024:
            raise ValueError("window must be 2..1024")
        if type(cooldown) is not int or not 1 <= cooldown <= 1024:
            raise ValueError("cooldown must be 1..1024")
        if not 0 <= minimum_overlap < 1 or not 0 <= maximum_waste <= 1:
            raise ValueError("invalid thresholds")
        self.window = window
        self.cooldown = cooldown
        self.minimum_overlap = minimum_overlap
        self.maximum_waste = maximum_waste
        self._observations: deque[ReadAheadObservation] = deque(maxlen=window)
        self._paused = 0
        self.considered = 0
        self.submitted = 0
        self.skipped = 0
        self.used = 0
        self.wasted = 0
        self.pauses = 0
        self.read_seconds = 0.0
        self.overlap_seconds = 0.0
        self.wait_seconds = 0.0

    def should_submit(self) -> bool:
        self.considered += 1
        if self._paused > 0:
            self._paused -= 1
            self.skipped += 1
            return False
        self.submitted += 1
        return True

    def observe(self, *, used: bool, read_seconds: float,
                wait_seconds: float = 0.0) -> None:
        if not (read_seconds >= 0 and wait_seconds >= 0):
            raise ValueError("timings must be nonnegative")
        self._observations.append(ReadAheadObservation(used, read_seconds, wait_seconds))
        self.read_seconds += read_seconds
        self.wait_seconds += wait_seconds
        if used:
            self.used += 1
            self.overlap_seconds += max(0.0, read_seconds - wait_seconds)
        else:
            self.wasted += 1
        if len(self._observations) < self.window:
            return
        observations = list(self._observations)
        wasted = sum(not item.used for item in observations)
        total_read = sum(item.read_seconds for item in observations)
        overlap = sum(max(0.0, item.read_seconds - item.wait_seconds)
                      for item in observations if item.used)
        waste_ratio = wasted / self.window
        overlap_ratio = overlap / total_read if total_read > 0 else 0.0
        if waste_ratio > self.maximum_waste or overlap_ratio < self.minimum_overlap:
            self._paused = self.cooldown
            self.pauses += 1
        self._observations.clear()

    def snapshot(self) -> dict:
        return {
            "mode": "adaptive", "considered": self.considered,
            "submitted": self.submitted, "skipped": self.skipped,
            "used": self.used, "wasted": self.wasted,
            "pauses": self.pauses, "paused_remaining": self._paused,
            "read_seconds": round(self.read_seconds, 6),
            "overlap_seconds": round(self.overlap_seconds, 6),
            "wait_seconds": round(self.wait_seconds, 6),
            "warning": "Heuristic overlap is NOT a proven latency gain.",
        }
