"""IroNasu Compass: evidence-gated self tuning for out-of-core inference.

Measure candidates on the *same prompt*, reject changed decoded tokens, and only
recommend a configuration if its measured gain clears a conservative threshold.
Never mutate checkpoints or silently assume that a faster run is more accurate.

This module intentionally does not require Torch to import: its decision logic
is testable with ordinary fake observations and no model/GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class Candidate:
    name: str
    prefetch: bool = False
    cache_mb: int = 0
    pin_stages: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.cache_mb, bool) or not isinstance(self.cache_mb, int) or self.cache_mb < 0:
            raise ValueError("cache_mb must be nonnegative integer")
        if self.pin_stages and not self.cache_mb:
            raise ValueError("pin_stages requires positive cache_mb")
        if len(set(self.pin_stages)) != len(self.pin_stages):
            raise ValueError("duplicate stage")

    def arguments(self) -> dict:
        return {"prefetch": self.prefetch, "cpu_cache_mb": self.cache_mb,
                "pin_stages": list(self.pin_stages) or None}


@dataclass(frozen=True)
class Observation:
    output: str
    seconds: float
    snapshot: dict[str, Any]


def candidates(cache_mb: int = 0, hot_stages: tuple[str, ...] = ()) -> list[Candidate]:
    """Generate a small, bounded exploration space; never allocate a cache implicitly."""
    base = [Candidate("baseline"), Candidate("prefetch", prefetch=True)]
    if cache_mb:
        base.extend((Candidate("fixed_cache", cache_mb=cache_mb),
                     Candidate("cache_and_prefetch", prefetch=True, cache_mb=cache_mb)))
        if hot_stages:
            base.append(Candidate("hot_cache", cache_mb=cache_mb, pin_stages=hot_stages))
    return base


def choose_hot_stages(trace: dict[str, Any], stage_bytes: dict[str, int],
                      budget_bytes: int) -> tuple[str, ...]:
    """Rank candidate CPU-resident layers by observed read time per byte.

    A heuristic only: a real warm-cache trial must confirm the choice. Do not
    conflate layer read time with FLOP savings or treat page-cache hits as disk I/O.
    """
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")
    measured = trace.get("per_stage", {})
    choices = []
    for name, data in measured.items():
        size = stage_bytes.get(name, 0)
        if size > 0 and size <= budget_bytes and data.get("loads", 0) > 0:
            measured_s = max(0.0, float(data.get("disk_seconds", 0)))
            choices.append((measured_s / size, measured_s, name, size))
    choices.sort(key=lambda e: (-e[0], -e[1], e[2]))
    selected = []
    left = budget_bytes
    for _, _, name, size in choices:
        if size <= left:
            selected.append(name)
            left -= size
    return tuple(selected)


def _median_secs(observations: list[Observation]) -> float:
    return statistics.median(o.seconds for o in observations)


def select_safe_candidate(records: dict[str, list[Observation]],
                          configs: list[Candidate], *, min_gain: float = .10,
                          baseline_after: list[Observation] | None = None) -> dict[str, Any]:
    """Fail-closed gate: baseline stays selected without repeated, matching outputs."""
    if not 0 <= min_gain < 1:
        raise ValueError("min_gain must be in [0,1)")
    if not configs or configs[0].name != "baseline" or not records.get("baseline"):
        raise ValueError("baseline observations required")
    initial = records["baseline"]
    for observations in [*records.values(), baseline_after or []]:
        for o in observations:
            if not isinstance(o.output, str) or o.seconds <= 0:
                raise ValueError("all observations need text output and positive elapsed time")
    answer = initial[0].output
    # Drift check: baseline run after all candidates exposes timing changes due
    # to OS page-cache warmup or background load. Use the later baseline as the
    # comparison reference when available; don't silently overstate certainty.
    baseline_checks = initial + (baseline_after or [])
    baseline_consistent = all(o.output == answer for o in baseline_checks)
    base_time = _median_secs(baseline_after) if baseline_after else _median_secs(initial)
    results = []
    selected = configs[0]
    selected_gain = 0.0
    for config in configs:
        obs = records.get(config.name, [])
        if not obs:
            results.append({"candidate": config.name, "status": "not_measured"})
            continue
        value = _median_secs(obs)
        gain = (base_time - value) / base_time
        status = "eligible"
        if not baseline_consistent:
            status = "rejected_unstable_baseline"
        elif len(obs) < len(initial):
            status = "rejected_incomplete_runs"
        elif any(o.output != answer for o in obs):
            status = "rejected_output_mismatch"
        elif config.name != "baseline" and gain < min_gain:
            status = "rejected_insufficient_gain"
        if config.name == "baseline":
            status = "reference" if baseline_consistent else "rejected_unstable_baseline"
        if status == "eligible" and gain > selected_gain:
            selected, selected_gain = config, gain
        results.append({"candidate": config.name, "status": status,
                        "median_seconds": round(value, 6),
                        "relative_gain": round(gain, 5),
                        "arguments": config.arguments()})
    return {
        "selected": selected.name,
        "selected_arguments": selected.arguments(),
        "measured_gain": round(selected_gain, 5),
        "reference_seconds": round(base_time, 6),
        "baseline_stable": baseline_consistent,
        "trials": results,
        "warning": ("Output equality checks decoded generated text on ONE prompt, not semantic "
                    "correctness or logit parity. Gain is sampled wall time, not FLOPs. "
                    "OS page cache, model variation and thermal effects can bias results."),
    }


def _run_trials(make_runner: Callable[[Candidate], Callable[[], Observation]],
                config: Candidate, repeats: int) -> list[Observation]:
    """One engine per candidate; repeated generations exercise persistent caches."""
    runner = make_runner(config)
    try:
        return [runner() for _ in range(repeats)]
    finally:
        close = getattr(runner, "close", None)
        if close is not None:
            close()


def tune(make_runner: Callable[[Candidate], Callable[[], Observation]], *,
         cache_mb: int = 0, repeats: int = 2, min_gain: float = .10,
         stage_bytes: dict[str, int] | None = None) -> dict[str, Any]:
    """Explore measured strategies; record failures and roll back to baseline.

    The reference is repeated at the end. If re-running it fails or changes the
    output, report no recommendation rather than guessing a speedup.
    """
    if repeats < 1 or repeats > 10:
        raise ValueError("repeats must be in 1..10")
    plans = candidates(cache_mb)
    records: dict[str, list[Observation]] = {}
    errors = {}
    baseline_after = None
    try:
        records["baseline"] = _run_trials(make_runner, plans[0], repeats)
    except Exception as exc:
        raise RuntimeError(f"Baseline failed: {type(exc).__name__}: {exc}") from exc
    if cache_mb and stage_bytes:
        hot = choose_hot_stages(records["baseline"][-1].snapshot, stage_bytes, cache_mb * 1024**2)
        if hot:
            plans.append(Candidate("hot_cache", cache_mb=cache_mb, pin_stages=hot))
    for config in plans[1:]:
        try:
            records[config.name] = _run_trials(make_runner, config, repeats)
        except Exception as exc:
            errors[config.name] = f"{type(exc).__name__}: {exc}"
    try:
        baseline_after = _run_trials(make_runner, plans[0], repeats)
    except Exception as exc:
        errors["baseline_after"] = f"{type(exc).__name__}: {exc}"
        # Treat inability to verify reference stability as fail-closed.
        return {"selected": "baseline", "selected_arguments": plans[0].arguments(),
                "measured_gain": 0.0, "baseline_stable": False,
                "errors": errors, "warning": "Reference re-test failed; optimization not recommended"}
    report = select_safe_candidate(records, plans, min_gain=min_gain, baseline_after=baseline_after)
    report["errors"] = errors
    report["repeats_per_trial"] = repeats
    report["exploration_count"] = len(plans)
    report["hot_stages"] = list(next((c.pin_stages for c in plans if c.name == "hot_cache"), ()))
    return report


def tune_local_model(directory: str | Path, prompt: str, *, device: str = "cpu",
                     tokens: int = 4, cache_mb: int = 0, repeats: int = 2,
                     min_gain: float = .10, profile: bool = True) -> dict[str, Any]:
    """Run empirical trials on a trusted local Transformers checkpoint.

    CPU cache is opt-in. GPU timings synchronize by default to avoid measuring
    only asynchronous CUDA launch times. Engine is closed after each trial.
    """
    from .stream import PagedCausalLM
    from .checkpoint import inspect_checkpoint

    class Runner:
        def __init__(self, config: Candidate):
            self.engine = PagedCausalLM.from_local_dir(directory, device=device,
                sync_timing=profile, **config.arguments())
        def __call__(self) -> Observation:
            started = perf_counter()
            generated = self.engine.generate(prompt, max_new_tokens=tokens)
            elapsed = perf_counter() - started
            return Observation(generated, elapsed, self.engine.pager.snapshot())
        def close(self):
            self.engine.close()

    sizes = inspect_checkpoint(directory)["stages_bytes"] if cache_mb else None
    return tune(Runner, cache_mb=cache_mb, repeats=repeats,
                min_gain=min_gain, stage_bytes=sizes)


def main():
    parser = argparse.ArgumentParser(description="IroNasu Compass — measured safe tuning; local models only")
    parser.add_argument("model_dir", help="trusted local model directory")
    parser.add_argument("--prompt", default="Write a Python function that sums integers.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-gain", type=float, default=.10)
    parser.add_argument("--output", help="write structured evidence to JSON (do not include private prompts)")
    args = parser.parse_args()
    if args.tokens < 1 or args.tokens > 1024:
        parser.error("--tokens must be between 1 and 1024")
    report = tune_local_model(args.model_dir, args.prompt, device=args.device, tokens=args.tokens,
                             cache_mb=args.cache_mb, repeats=args.repeats, min_gain=args.min_gain)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
