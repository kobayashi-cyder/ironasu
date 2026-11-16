"""Conservative, request-scoped execution budgets for the ironasu gateway.

Budgets are deliberately restricted to policies the gateway can enforce.
They do not pretend to measure/limit GPU memory, FLOPs or total generation time.
No model output is fabricated, shortened, or accepted without verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MAX_QUEUE_BUDGET_MS = 120_000
_SUPPORTED = frozenset({"max_queue_ms", "preserve_auto_quality", "allow_cache"})


@dataclass(frozen=True)
class RequestBudget:
    """Immutable plan request; absent budgets preserve historical routing.

    An explicit budget defaults to conservative auto quality preservation:
    it disables *only* the existing pressure-triggered downgrade of short
    ambiguous prompts, not explicit fast mode or recognized simple tasks.
    """
    explicit: bool = False
    max_queue_ms: int | None = None
    preserve_auto_quality: bool = False
    allow_cache: bool = True

    @classmethod
    def from_request(cls, payload: dict[str, Any]) -> "RequestBudget":
        if "x_ironasu_budget" not in payload:
            return cls()
        raw = payload["x_ironasu_budget"]
        if not isinstance(raw, dict):
            raise ValueError("x_ironasu_budget must be an object")
        unknown = set(raw) - _SUPPORTED
        if unknown:
            raise ValueError("unsupported x_ironasu_budget field(s): " + ", ".join(sorted(map(str, unknown))))
        queue = raw.get("max_queue_ms")
        if queue is not None and (type(queue) is not int or not 0 <= queue <= MAX_QUEUE_BUDGET_MS):
            raise ValueError("max_queue_ms must be an integer in 0..120000")
        preserve = raw.get("preserve_auto_quality", True)
        allow_cache = raw.get("allow_cache", True)
        if type(preserve) is not bool or type(allow_cache) is not bool:
            raise ValueError("preserve_auto_quality and allow_cache must be booleans")
        return cls(True, queue, preserve, allow_cache)

    def queue_timeout_s(self, configured_timeout_s: float) -> float:
        """Never extend the server's configured queue timeout."""
        if self.max_queue_ms is None:
            return configured_timeout_s
        return min(configured_timeout_s, self.max_queue_ms / 1000.0)

    def pressure_for_routing(self, observed_pressure: bool) -> bool:
        """Retain explicit model routing; suppress only pressure-based auto fallback."""
        return observed_pressure and not self.preserve_auto_quality

    def diagnostic(self, *, selected_route: str) -> dict[str, Any]:
        """An explanation with no prompt, model credentials or generated text."""
        return {
            "policy": "explicit" if self.explicit else "legacy",
            "selected_route": selected_route,
            "preserve_auto_quality": self.preserve_auto_quality,
            "max_queue_ms": self.max_queue_ms,
            "cache_permitted": self.allow_cache,
            "limits": ["queue waiting only; backend decode time and GPU memory are not capped"],
        }
