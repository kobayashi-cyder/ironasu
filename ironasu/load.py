"""Real concurrent-load A/B benchmark and fail-closed per-concurrency gate.

Concurrent output throughput is measured over the complete batch wall time,
not the sum of overlapping per-request latencies. No GPU utilization, TTFT,
TPOT, cost or quality beyond exact text equality is inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import isfinite
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

from .benchmark import Completion, make_client, summarize
from .regression import GatePolicy, _count, _number, _positive


def _cases(prompts: Iterable[str], *, repeats: int, warmup: int,
           concurrencies: tuple[int, ...]) -> tuple[str, ...]:
    cases = tuple(prompts)
    if not cases or len(cases) > 1000 or any(not isinstance(p, str) or not p for p in cases):
        raise ValueError("prompts must contain between 1 and 1000 nonempty strings")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be in 1..20")
    if type(warmup) is not int or not 0 <= warmup <= 10:
        raise ValueError("warmup must be in 0..10")
    if (not concurrencies or any(type(c) is not int or not 1 <= c <= 128 for c in concurrencies)
            or len(set(concurrencies)) != len(concurrencies)):
        raise ValueError("concurrencies must contain distinct integers in 1..128")
    return cases


def _batch(client: Callable[[str], Completion], cases: tuple[str, ...],
           workers: int) -> tuple[list[Completion], float]:
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ironasu-load") as pool:
        # map preserves prompt indices even when completion order changes.
        rows = list(pool.map(client, cases))
    elapsed = perf_counter() - start
    if elapsed <= 0 or not isfinite(elapsed):
        raise RuntimeError("invalid batch wall time")
    if any(not isinstance(row, Completion) for row in rows):
        raise TypeError("client must return Completion instances")
    return rows, elapsed


def _summary(rows: list[Completion], wall_seconds: float) -> dict[str, Any]:
    stats = summarize(rows)
    tokens = stats["measured_output_tokens"]
    return {
        "samples": stats["samples"],
        "p50_seconds": stats["p50_seconds"],
        "p95_seconds": stats["p95_seconds"],
        "mean_seconds": stats["mean_seconds"],
        "measured_output_tokens": tokens,
        "batch_wall_seconds": wall_seconds,
        "requests_per_wall_second": len(rows) / wall_seconds,
        "concurrent_output_tokens_per_wall_second": tokens / wall_seconds if tokens is not None else None,
    }


def compare_load(
    prompts: Iterable[str],
    *,
    baseline: Callable[[str], Completion],
    candidate: Callable[[str], Completion],
    concurrencies: tuple[int, ...] = (1, 8, 32),
    repeats: int = 2,
    warmup: int = 1,
) -> dict[str, Any]:
    """Run whole-workload batches, alternating baseline/candidate trial order.

    Both phases are sequential in each trial to avoid benchmark servers
    competing with one another. Each phase itself has bounded client parallelism.
    """
    cases = _cases(prompts, repeats=repeats, warmup=warmup, concurrencies=concurrencies)
    for _ in range(warmup):
        baseline(cases[0])
        candidate(cases[0])
    profiles = []
    for workers in concurrencies:
        all_rows: dict[str, list[Completion]] = {"baseline": [], "candidate": []}
        trials: dict[str, list[list[Completion]]] = {"baseline": [], "candidate": []}
        wall: dict[str, float] = {"baseline": 0.0, "candidate": 0.0}
        for trial in range(repeats):
            order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
            for side in order:
                rows, elapsed = _batch(baseline if side == "baseline" else candidate, cases, workers)
                trials[side].append(rows)
                all_rows[side].extend(rows)
                wall[side] += elapsed
        mismatches: list[dict[str, int]] = []
        for trial in range(repeats):
            for i, (b, c) in enumerate(zip(trials["baseline"][trial], trials["candidate"][trial])):
                if b.content != c.content:
                    mismatches.append({"prompt_index": i, "repeat": trial})
        b_stats = _summary(all_rows["baseline"], wall["baseline"])
        c_stats = _summary(all_rows["candidate"], wall["candidate"])
        br = b_stats["concurrent_output_tokens_per_wall_second"]
        cr = c_stats["concurrent_output_tokens_per_wall_second"]
        profiles.append({
            "concurrency": workers,
            "baseline": b_stats,
            "candidate": c_stats,
            "exact_output_match_rate": 1 - len(mismatches) / (len(cases) * repeats),
            "mismatches": mismatches,
            "p95_latency_ratio": c_stats["p95_seconds"] / b_stats["p95_seconds"],
            "concurrent_throughput_ratio": cr / br if br is not None and br > 0 and cr is not None else None,
        })
    return {
        "measurement_kind": "concurrent_http_batches",
        "prompts": len(cases), "repeats": repeats, "warmup": warmup,
        "profiles": profiles,
        "limitations": [
            "Client threads overlap requests, but this is not server GPU utilization or actual batching evidence.",
            "TTFT/TPOT and quality beyond exact output text match are not recorded.",
            "Warm caches, server/network configuration and interference may bias small samples.",
            "Benchmarking both endpoints sequentially requires repeated/reversed trials to limit order effects.",
        ],
    }


def evaluate_load(report: dict[str, Any], policy: GatePolicy | None = None) -> dict[str, Any]:
    """Fail if ANY concurrency level exceeds p95 or throughput constraints."""
    if not isinstance(report, dict) or report.get("measurement_kind") != "concurrent_http_batches":
        raise ValueError("expected ironasu concurrent HTTP batch benchmark report")
    policy = policy if policy is not None else GatePolicy()
    if policy.latency_only:
        raise ValueError("latency_only is not supported for the concurrent load gate")
    prompts = _count(report.get("prompts"), "prompts", positive=True)
    repeats = _count(report.get("repeats"), "repeats", positive=True)
    expected = prompts * repeats
    profiles = report.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("report profiles must be a nonempty array")
    seen = set()
    evaluated: list[dict[str, Any]] = []
    for row in profiles:
        if not isinstance(row, dict):
            raise ValueError("profile must be an object")
        workers = _count(row.get("concurrency"), "concurrency", positive=True)
        if workers > 128 or workers in seen:
            raise ValueError("invalid or duplicate concurrency")
        seen.add(workers)
        b, c = row.get("baseline"), row.get("candidate")
        if not isinstance(b, dict) or not isinstance(c, dict):
            raise ValueError("missing baseline/candidate profile")
        bs = _count(b.get("samples"), "baseline.samples", positive=True)
        cs = _count(c.get("samples"), "candidate.samples", positive=True)
        bt = _count(b.get("measured_output_tokens"), "baseline.output_tokens", positive=True)
        ct = _count(c.get("measured_output_tokens"), "candidate.output_tokens", positive=True)
        b_latency = _positive(b.get("p95_seconds"), "baseline.p95_seconds")
        c_latency = _positive(c.get("p95_seconds"), "candidate.p95_seconds")
        b_wall = _positive(b.get("batch_wall_seconds"), "baseline.batch_wall_seconds")
        c_wall = _positive(c.get("batch_wall_seconds"), "candidate.batch_wall_seconds")
        b_rate = _positive(b.get("concurrent_output_tokens_per_wall_second"), "baseline.concurrent_rate")
        c_rate = _positive(c.get("concurrent_output_tokens_per_wall_second"), "candidate.concurrent_rate")
        # Ensure the report cannot pass using a fabricated/inconsistent throughput
        # rather than the measured token totals and elapsed batch wall time.
        if not abs(b_rate - bt / b_wall) <= max(1e-8, 1e-6 * b_rate):
            raise ValueError("baseline throughput inconsistent with batch timing")
        if not abs(c_rate - ct / c_wall) <= max(1e-8, 1e-6 * c_rate):
            raise ValueError("candidate throughput inconsistent with batch timing")
        parity = _number(row.get("exact_output_match_rate"), "exact_output_match_rate")
        if parity > 1 or not isinstance(row.get("mismatches"), list):
            raise ValueError("invalid parity or mismatches")
        p95_ratio = c_latency / b_latency
        rate_ratio = c_rate / b_rate
        checks = {
            "sufficient_samples": bs == cs == expected and expected >= policy.min_samples,
            "output_parity": parity == 1.0 and len(row["mismatches"]) == 0,
            "equal_output_tokens": bt == ct,
            "p95_latency": p95_ratio <= 1 + policy.max_p95_regression,
            "concurrent_throughput": rate_ratio >= 1 + policy.min_throughput_gain,
        }
        evaluated.append({
            "concurrency": workers, "passed": all(checks.values()),
            "checks": checks,
            "p95_latency_ratio": p95_ratio,
            "concurrent_throughput_ratio": rate_ratio,
            "baseline_requests_per_second": bs / b_wall,
            "candidate_requests_per_second": cs / c_wall,
        })
    return {
        "passed": all(item["passed"] for item in evaluated),
        "profiles": evaluated,
        "policy": {
            "min_samples": policy.min_samples,
            "max_p95_regression": policy.max_p95_regression,
            "min_throughput_gain": policy.min_throughput_gain,
        },
        "limitations": [
            "A pass only establishes measured constraints on supplied workloads and tested concurrency.",
            "Output parity is exact text equality, not correctness or model quality.",
            "No TTFT, TPOT, GPU utilization, VRAM, energy or cost measurement is made.",
            "Check model/hardware/network equivalence and repeat trials before production changes.",
        ],
    }


def _parse_concurrencies(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("concurrency values must be comma-separated integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure actual concurrent HTTP A/B load on two model servers")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", default="1,8,32", help="comma-separated worker counts")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        if not isinstance(prompts, list):
            raise ValueError("prompts must be a JSON array")
        kw = {"model": args.model, "key": os.getenv("IRONASU_BACKEND_KEY", ""),
              "max_tokens": args.max_tokens, "timeout": args.timeout}
        report = compare_load(
            prompts,
            baseline=make_client(args.baseline_url, **kw),
            candidate=make_client(args.candidate_url, **kw),
            concurrencies=_parse_concurrencies(args.concurrency),
            repeats=args.repeat, warmup=args.warmup,
        )
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 1 if any(p["mismatches"] for p in report["profiles"]) else 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


def gate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject regressions at ANY measured concurrency")
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-p95-regression", type=float, default=0.05)
    parser.add_argument("--min-throughput-gain", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = GatePolicy(args.min_samples, args.max_p95_regression, args.min_throughput_gain)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        verdict = evaluate_load(report, policy)
        serialized = json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if verdict["passed"] else 1
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
