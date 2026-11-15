"""Experimental speculative initialization of immutable CPU metadata.

Not a KV cache, tensor store, draft-token decoder, or external side-effect
rollback. Candidate builders must have NO external side effects.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock, RLock
from types import MappingProxyType
from typing import Callable, Mapping

_DELETED = object()
_MISSING = object()
_COMPACT_DEPTH = 32


class BusySpeculationError(RuntimeError):
    """Existing speculation is still active or draining."""


@dataclass(frozen=True)
class _Node:
    parent: _Node | None
    delta: Mapping[str, object]
    depth: int


@dataclass(frozen=True)
class Snapshot:
    generation: int
    revision: int
    _node: _Node

    def get(self, key: str, default: bytes | None = None) -> bytes | None:
        node: _Node | None = self._node
        while node is not None:
            value = node.delta.get(key, _MISSING)
            if value is not _MISSING:
                return default if value is _DELETED else value  # type: ignore[return-value]
            node = node.parent
        return default

    def materialize(self) -> dict[str, bytes]:
        """Full copy: O(all live and historic entries), NOT O(1)."""
        chain: list[Mapping[str, object]] = []
        node: _Node | None = self._node
        while node is not None:
            chain.append(node.delta)
            node = node.parent
        result: dict[str, bytes] = {}
        for delta in reversed(chain):
            for key, value in delta.items():
                if value is _DELETED:
                    result.pop(key, None)
                else:
                    result[key] = value  # type: ignore[assignment]
        return result


class Branch:
    """A single-writer copy-on-write branch, sealed after one publication attempt."""

    def __init__(self, base: Snapshot, byte_budget: int, key_budget: int):
        self.base = base
        self._delta: dict[str, object] = {}
        self._byte_budget = byte_budget
        self._key_budget = key_budget
        self._bytes = 0
        self._sealed = False

    def _check(self, key: str) -> None:
        if self._sealed:
            raise RuntimeError("branch is sealed")
        if not isinstance(key, str) or not key or len(key) > 512:
            raise ValueError("key must be a nonempty string of <=512 characters")
        if key not in self._delta and len(self._delta) >= self._key_budget:
            raise MemoryError("candidate exceeds the key-count budget")

    def write(self, key: str, value: bytes) -> None:
        self._check(key)
        if not isinstance(value, bytes):
            raise TypeError("state values must be immutable bytes")
        previous = self._delta.get(key)
        new_bytes = self._bytes - (len(previous) if isinstance(previous, bytes) else 0) + len(value)
        if new_bytes > self._byte_budget:
            raise MemoryError("candidate exceeds the payload-byte budget")
        self._delta[key] = value
        self._bytes = new_bytes

    def delete(self, key: str) -> None:
        self._check(key)
        previous = self._delta.get(key)
        if isinstance(previous, bytes):
            self._bytes -= len(previous)
        self._delta[key] = _DELETED

    def get(self, key: str, default: bytes | None = None) -> bytes | None:
        value = self._delta.get(key, _MISSING)
        if value is _DELETED:
            return default
        return self.base.get(key, default) if value is _MISSING else value  # type: ignore[return-value]

    def seal(self) -> Snapshot:
        if self._sealed:
            raise RuntimeError("branch is sealed")
        self._sealed = True
        node = _Node(self.base._node, MappingProxyType(dict(self._delta)),
                     self.base._node.depth + 1)
        return Snapshot(self.base.generation, self.base.revision, node)


class StateStore:
    """Validated compare-and-swap state publication.

    Old snapshots stay valid after reset. Reset does not zero physical RAM or
    delete files. Compaction occasionally costs O(number of live keys).
    """

    def __init__(self, *, max_delta_bytes: int = 16 * 1024 * 1024,
                 max_delta_keys: int = 4096):
        if type(max_delta_bytes) is not int or max_delta_bytes < 0:
            raise ValueError("max_delta_bytes must be >=0")
        if type(max_delta_keys) is not int or max_delta_keys < 1:
            raise ValueError("max_delta_keys must be >=1")
        self._byte_budget = max_delta_bytes
        self._key_budget = max_delta_keys
        self._lock = RLock()
        self._generation = 0
        self._revision = 0
        self._root = _Node(None, MappingProxyType({}), 0)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._generation, self._revision, self._root)

    def fork(self, base: Snapshot | None = None) -> Branch:
        return Branch(self.snapshot() if base is None else base,
                      self._byte_budget, self._key_budget)

    def reset(self) -> Snapshot:
        """O(1) logical invalidation via a new root and generation."""
        with self._lock:
            self._generation += 1
            self._revision += 1
            self._root = _Node(None, MappingProxyType({}), 0)
            return Snapshot(self._generation, self._revision, self._root)

    def try_promote(self, branch: Branch, *,
                    verify: Callable[[Snapshot], bool] | None = None) -> bool:
        """Validate before publication, then reject conflicting revisions.

        verify must be pure; it runs outside the store lock. A failed validation,
        reset, or competing commit cannot publish a candidate.
        """
        proposed = branch.seal()
        if verify is not None and not verify(proposed):
            return False
        with self._lock:
            base = branch.base
            if (base.generation != self._generation or
                    base.revision != self._revision or base._node is not self._root):
                return False
            self._root = proposed._node
            if self._root.depth >= _COMPACT_DEPTH:
                self._root = _Node(None, MappingProxyType(proposed.materialize()), 0)
            self._revision += 1
            return True


@dataclass(frozen=True)
class SpeculationResult:
    selected: str | None
    committed: bool
    generation: int
    revision: int
    rejected: tuple[str, ...]
    errors: Mapping[str, str]


class SpeculativeInitializer:
    """Race bounded pure builders and accept the first VERIFIED result.

    A losing running task may finish after run() returns. Until it drains,
    another run is refused rather than building an unbounded work queue.
    Call close() or use a context manager to drain all worker threads.
    """

    def __init__(self, store: StateStore, *, max_candidates: int = 2):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
            raise ValueError("max_candidates must be 1..8")
        self.store = store
        self.max_candidates = max_candidates
        self._executor = ThreadPoolExecutor(
            max_workers=max_candidates, thread_name_prefix="ironasu-spec")
        self._lock = Lock()
        self._running = False
        self._closed = False
        self._pending: set[Future[Branch]] = set()

    def __enter__(self) -> SpeculativeInitializer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._running:
                raise BusySpeculationError("cannot close during run")
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _build(base: Snapshot, byte_budget: int, key_budget: int,
               builder: Callable[[Branch], None]) -> Branch:
        branch = Branch(base, byte_budget, key_budget)
        builder(branch)
        return branch

    def run(self, strategies: Mapping[str, Callable[[Branch], None]], *,
            verify: Callable[[str, Snapshot], bool]) -> SpeculationResult:
        if not strategies or len(strategies) > self.max_candidates:
            raise ValueError("supply between 1 and max_candidates strategies")
        if not callable(verify):
            raise TypeError("verify callback must be callable")
        if any(not isinstance(k, str) or not k or not callable(v)
               for k, v in strategies.items()):
            raise ValueError("strategies need nonempty names and callable builders")
        with self._lock:
            if self._closed:
                raise RuntimeError("initializer is closed")
            if self._running or any(not f.done() for f in self._pending):
                raise BusySpeculationError("previous candidates still running")
            self._pending.clear()
            self._running = True
        errors: dict[str, str] = {}
        rejected: list[str] = []
        selected: str | None = None
        try:
            base = self.store.snapshot()
            work = {
                self._executor.submit(self._build, base, self.store._byte_budget,
                                      self.store._key_budget, builder): name
                for name, builder in strategies.items()
            }
            with self._lock:
                self._pending = set(work)
            for future in as_completed(work):
                name = work[future]
                try:
                    candidate = future.result()
                    ok = self.store.try_promote(
                        candidate, verify=lambda snap: bool(verify(name, snap)))
                    if ok:
                        selected = name
                        break
                    rejected.append(name)
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
            if selected is not None:
                for f in work:
                    if not f.done():
                        f.cancel()
        finally:
            with self._lock:
                self._running = False
        snap = self.store.snapshot()
        return SpeculationResult(selected, selected is not None, snap.generation,
                                 snap.revision, tuple(rejected),
                                 MappingProxyType(dict(errors)))


def main() -> None:
    """Synthetic operation timing only; NOT an LLM speed benchmark."""
    import argparse
    import json
    from time import perf_counter
    p = argparse.ArgumentParser(description="Compare rebuild and O(1) logical reset")
    p.add_argument("--keys", type=int, default=10000)
    p.add_argument("--rounds", type=int, default=100)
    a = p.parse_args()
    if not 1 <= a.keys <= 100000 or not 1 <= a.rounds <= 10000:
        p.error("keys and rounds out of supported range")
    source = {f"k{i}": b"value" for i in range(a.keys)}
    values = source.copy()
    start = perf_counter()
    for _ in range(a.rounds):
        values.clear()
        values.update(source)
    rebuilt = perf_counter() - start
    store = StateStore()
    start = perf_counter()
    for _ in range(a.rounds):
        store.reset()
    reset = perf_counter() - start
    print(json.dumps({
        "keys": a.keys, "rounds": a.rounds,
        "rebuild_seconds": rebuilt, "logical_reset_seconds": reset,
        "warning": "Different operations: rebuild repopulates all keys, reset leaves "
                   "an empty state. No inference speedup is implied."
    }, indent=2))


if __name__ == "__main__":
    main()
