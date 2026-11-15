"""Opt-in AirLLM v3+ method probes. Uses best-effort private method names.

This is diagnostic instrumentation, not an AirLLM patch or an optimization.
Version changes may require adjusting the wrapped method names.
"""
from __future__ import annotations

from collections import defaultdict
from functools import wraps
import threading
import time
from typing import Any


class AirLLMStageProbe:
    """Time AirLLM's disk/shard decoder and weight materialization methods.

    Timings can overlap when prefetch threads work while kernels execute. The
    GPU may run asynchronously; `transfer_setup` is Python wall time and not
    necessarily measured H2D transfer time. Call detach() after the run.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._original: dict[str, Any] = {}
        self._lock = threading.Lock()

    def attach(self) -> "AirLLMStageProbe":
        # _load_streamed_layer is the v3+ path incl. MoE subset loads.
        name = "_load_streamed_layer" if callable(getattr(self.model, "_load_streamed_layer", None)) else "load_layer_to_cpu"
        for attr, stage in ((name, "disk_io_plus_decode"), ("move_layer_to_device", "transfer_plus_setup")):
            original = getattr(self.model, attr, None)
            if not callable(original):
                continue
            if attr in self._original:
                continue
            self._original[attr] = original

            @wraps(original)
            def wrapped(*args: Any, __original=original, __stage=stage, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return __original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    with self._lock:
                        self.times[__stage] += elapsed
                        self.counts[__stage] += 1

            setattr(self.model, attr, wrapped)
        return self

    def detach(self) -> None:
        for name, method in self._original.items():
            setattr(self.model, name, method)
        self._original.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "durations_seconds": {k: round(v, 6) for k, v in self.times.items()},
                "counts": dict(self.counts),
                "warning": (
                    "Disk timings include deserialization/CPU work; transfer/setup includes GPU async dispatch "
                    "and may omit GPU completion. Prefetch work overlaps compute: do not sum stages to estimate wall time."
                ),
            }

    def __enter__(self) -> "AirLLMStageProbe":
        return self.attach()

    def __exit__(self, *_: Any) -> None:
        self.detach()
