"""Backend-independent scheduling components for ironasu (experimental).

Do NOT emit unverified draft tokens. A compatible model executor must perform
target-model speculative verification, sampling, and real device transfers.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Deque, Generic, Hashable, Iterable, Optional, TypeVar

Token = int
State = TypeVar("State")
Weights = TypeVar("Weights")


class NgramDraftIndex:
    """Bounded, per-session hash-table index of *unverified* draft continuations.

    Tokens may contain sensitive data. Never share this index across tenants.
    """

    def __init__(self, order: int = 4, max_entries: int = 16384, max_draft: int = 8):
        if order < 1 or max_entries < 1 or max_draft < 1:
            raise ValueError("order, max_entries, and max_draft must be positive")
        self.order = order
        self.max_entries = max_entries
        self.max_draft = max_draft
        self._entries: OrderedDict[tuple[Token, ...], tuple[Token, ...]] = OrderedDict()

    def remember(self, tokens: Iterable[Token]) -> None:
        """Index contiguous follow-ons from explicitly approved token history."""
        sequence = tuple(tokens)
        for offset in range(len(sequence) - self.order):
            key = sequence[offset : offset + self.order]
            continuation = sequence[offset + self.order : offset + self.order + self.max_draft]
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = continuation
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def propose(self, context: Iterable[Token], count: int) -> tuple[Token, ...]:
        """A proposal is not an accepted or verified generation."""
        if count <= 0:
            return ()
        suffix = deque(context, maxlen=self.order)
        if len(suffix) != self.order:
            return ()
        key = tuple(suffix)
        proposal = self._entries.get(key)
        if proposal is None:
            return ()
        self._entries.move_to_end(key)
        return proposal[: min(count, self.max_draft)]

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class SpeculationChoice:
    draft_tokens: int
    reason: str
    estimated_speedup: Optional[float] = None


@dataclass
class _EMA:
    mean: float
    observations: int = 1

    def add(self, value: float, weight: float) -> None:
        self.mean = self.mean * (1.0 - weight) + value * weight
        self.observations += 1


class AdaptiveSpeculationScheduler:
    """Choose speculative draft length from measured decode-only seconds/token.

    Measurements exclude prompt prefill and queuing. The backend must provide
    the accepted draft count and the number of VERIFIED emitted tokens. The
    scheduler deliberately disables speculation until a real baseline exists.
    """

    def __init__(
        self,
        draft_lengths: tuple[int, ...] = (2, 4, 6, 8),
        min_samples: int = 3,
        explore_every: int = 8,
        min_speedup: float = 1.10,
        ema_weight: float = 0.25,
    ) -> None:
        if (not draft_lengths or any(k < 1 for k in draft_lengths)
                or len(set(draft_lengths)) != len(draft_lengths)):
            raise ValueError("draft_lengths must be distinct positive integers")
        if min_samples < 1 or explore_every < 1:
            raise ValueError("min_samples and explore_every must be positive")
        if min_speedup <= 1 or not isfinite(min_speedup):
            raise ValueError("min_speedup must be finite and > 1")
        if not 0 < ema_weight <= 1:
            raise ValueError("ema_weight must be in (0, 1]")
        self.draft_lengths = tuple(sorted(draft_lengths))
        self.min_samples = min_samples
        self.explore_every = explore_every
        self.min_speedup = min_speedup
        self.ema_weight = ema_weight
        self._baseline: Optional[_EMA] = None
        self._spec: dict[int, _EMA] = {}
        self._accept: dict[int, _EMA] = {}
        self._choices = 0

    @staticmethod
    def _validate_cycle(seconds: float, verified_tokens: int) -> float:
        if not isfinite(seconds) or seconds <= 0 or verified_tokens < 1:
            raise ValueError("seconds must be finite and > 0; verified_tokens >= 1")
        return seconds / verified_tokens

    def record_baseline(self, decode_seconds: float, verified_tokens: int) -> None:
        value = self._validate_cycle(decode_seconds, verified_tokens)
        if self._baseline is None:
            self._baseline = _EMA(value)
        else:
            self._baseline.add(value, self.ema_weight)

    def record_speculative(
        self,
        draft_tokens: int,
        accepted_tokens: int,
        decode_seconds: float,
        verified_tokens: int,
    ) -> None:
        if draft_tokens not in self.draft_lengths:
            raise ValueError("unsupported draft_tokens")
        if not 0 <= accepted_tokens <= draft_tokens:
            raise ValueError("accepted_tokens out of range")
        if verified_tokens > accepted_tokens + 1:
            raise ValueError("verified tokens cannot exceed accepted draft + target bonus")
        value = self._validate_cycle(decode_seconds, verified_tokens)
        slot = self._spec.get(draft_tokens)
        if slot is None:
            self._spec[draft_tokens] = _EMA(value)
        else:
            slot.add(value, self.ema_weight)
        acceptance = accepted_tokens / draft_tokens
        slot = self._accept.get(draft_tokens)
        if slot is None:
            self._accept[draft_tokens] = _EMA(acceptance)
        else:
            slot.add(acceptance, self.ema_weight)

    def choose(
        self, *, backend_supports_speculation: bool,
        queue_busy: bool = False, memory_headroom: bool = True
    ) -> SpeculationChoice:
        """Only recommend speculation with verification support and headroom."""
        if not backend_supports_speculation:
            return SpeculationChoice(0, "backend verification unavailable")
        if queue_busy or not memory_headroom:
            return SpeculationChoice(0, "resource pressure: prioritize target")
        if self._baseline is None or self._baseline.observations < self.min_samples:
            return SpeculationChoice(0, "need measured non-speculative baseline")

        self._choices += 1
        under_sampled = [k for k in self.draft_lengths
                         if self._spec.get(k, _EMA(0, 0)).observations < self.min_samples]
        if under_sampled and (self._choices == 1 or self._choices % self.explore_every == 0):
            k = min(under_sampled, key=lambda n: self._spec.get(n, _EMA(0, 0)).observations)
            return SpeculationChoice(k, "bounded exploration")

        eligible = [(k, entry.mean) for k, entry in self._spec.items()
                    if entry.observations >= self.min_samples]
        if not eligible:
            return SpeculationChoice(0, "not enough speculative measurements")
        k, best_s_per_token = min(eligible, key=lambda pair: pair[1])
        speedup = self._baseline.mean / best_s_per_token
        if speedup < self.min_speedup:
            return SpeculationChoice(0, "measured speedup below threshold", speedup)
        return SpeculationChoice(k, "measured improvement", speedup)

    def snapshot(self) -> dict:
        return {
            "baseline_seconds_per_token": self._baseline.mean if self._baseline else None,
            "spec_seconds_per_token": {k: v.mean for k, v in self._spec.items()},
            "acceptance_fraction_by_draft_length": {k: v.mean for k, v in self._accept.items()},
            "samples": {k: v.observations for k, v in self._spec.items()},
        }


@dataclass(frozen=True)
class LayerSpec:
    layer_id: Hashable
    estimated_bytes: int

    def __post_init__(self) -> None:
        if self.estimated_bytes <= 0:
            raise ValueError("layer estimated_bytes must be > 0")


class OrderedLayerPrefetcher(Generic[Weights, State]):
    """Overlap loading L+1 with executing L, in strict model-layer order.

    Limits outstanding work and *estimated weight bytes*, NOT actual VRAM.
    load/apply must be supplied by a real backend. apply must not retain the
    loaded weights after returning for resource accounting to remain valid.
    """

    def __init__(self, max_inflight: int = 2, max_staged_bytes: int = 2 << 30):
        if max_inflight < 1 or max_staged_bytes < 1:
            raise ValueError("max_inflight and max_staged_bytes must be positive")
        self.max_inflight = max_inflight
        self.max_staged_bytes = max_staged_bytes

    def run(
        self,
        layers: Iterable[LayerSpec],
        load: Callable[[Hashable], Weights],
        apply: Callable[[Hashable, Weights, State], State],
        initial_state: State,
    ) -> State:
        remaining = iter(layers)
        next_spec = next(remaining, None)
        pending: Deque[tuple[LayerSpec, Future[Weights]]] = deque()
        reserved = 0
        state = initial_state
        with ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix="ironasu-prefetch") as pool:
            while next_spec is not None or pending:
                while next_spec is not None and len(pending) < self.max_inflight:
                    if next_spec.estimated_bytes > self.max_staged_bytes:
                        raise ValueError("one layer exceeds max_staged_bytes")
                    if reserved + next_spec.estimated_bytes > self.max_staged_bytes:
                        break
                    pending.append((next_spec, pool.submit(load, next_spec.layer_id)))
                    reserved += next_spec.estimated_bytes
                    next_spec = next(remaining, None)
                if not pending:
                    raise RuntimeError("layer prefetch could not make progress")
                spec, future = pending.popleft()
                weights = future.result()
                try:
                    state = apply(spec.layer_id, weights, state)
                finally:
                    del weights
                    reserved -= spec.estimated_bytes
        return state
