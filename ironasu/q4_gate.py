"""Fail-closed Q4 adoption gate: checkpoint size, measured timing and scored quality.

Consumes an ironasu-benchmark-q4 report plus independent, paired evaluation
scores. The Q4 pager is lossy. Identical text is neither necessary nor
sufficient to establish semantic quality. No GPU/FLOP/energy claim is inferred.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from statistics import mean, median
from typing import Any

from .regression import _count, _number, _positive


def _score(value: Any, field: str) -> float:
    score = _number(value, field)
    if score > 1:
        raise ValueError(f"{field} must be within [0,1]")
    return score


@dataclass(frozen=True)
class Q4GatePolicy:
    min_cases: int = 10
    min_repeats: int = 3
    min_bytes_saved: float = 0.10
    max_median_slowdown: float = 0.05
    max_mean_quality_loss: float = 0.01
    max_case_quality_loss: float = 0.05

    def __post_init__(self) -> None:
        _count(self.min_cases, "min_cases", positive=True)
        _count(self.min_repeats, "min_repeats", positive=True)
        for name in ("min_bytes_saved", "max_median_slowdown",
                     "max_mean_quality_loss", "max_case_quality_loss"):
            if _number(getattr(self, name), name) > 1:
                raise ValueError(f"{name} must be <= 1")


def _times(side: Any, label: str) -> list[float]:
    if not isinstance(side, dict):
        raise ValueError(f"{label} must be an object")
    raw = side.get("generation_s")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}.generation_s must be a nonempty list")
    return [_positive(v, f"{label}.generation_s[{i}]") for i, v in enumerate(raw)]


def evaluate_q4(
    benchmark: dict[str, Any],
    quality: dict[str, Any],
    policy: Q4GatePolicy | None = None,
) -> dict[str, Any]:
    """Require matched, independent quality evidence before adopting lossy Q4.

    A passed gate is *local evidence under the supplied evaluation workload*,
    not a proof of coding accuracy, GPU optimization or generalization.
    """
    if not isinstance(benchmark, dict) or not isinstance(quality, dict):
        raise ValueError("benchmark and quality must be JSON objects")
    policy = policy if policy is not None else Q4GatePolicy()
    if not isinstance(policy, Q4GatePolicy):
        raise ValueError("policy must be Q4GatePolicy")

    payload = benchmark.get("checkpoint_payload")
    if not isinstance(payload, dict):
        raise ValueError("missing checkpoint_payload")
    base_bytes = _count(payload.get("baseline_bytes"), "baseline_bytes", positive=True)
    q4_bytes = _count(payload.get("q4_bytes"), "q4_bytes", positive=True)
    saved_fraction = 1.0 - q4_bytes / base_bytes
    reported_fraction = _number(payload.get("q4_fraction"), "q4_fraction")
    if not isclose(reported_fraction, q4_bytes / base_bytes, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("q4_fraction does not match reported checkpoint sizes")

    b = benchmark.get("baseline")
    q = benchmark.get("q4")
    baseline_times = _times(b, "baseline")
    candidate_times = _times(q, "q4")
    if len(baseline_times) != len(candidate_times):
        raise ValueError("baseline and Q4 timing sample counts must match")
    b_median, q_median = median(baseline_times), median(candidate_times)
    speedup = b_median / q_median
    reported_speedup = _positive(benchmark.get("speed_ratio_q4_vs_baseline"), "speed_ratio_q4_vs_baseline")
    if not isclose(speedup, reported_speedup, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("reported speed ratio does not agree with timing samples")
    for name, side in (("baseline", b), ("q4", q)):
        if type(side.get("stable_output")) is not bool:
            raise ValueError(f"{name}.stable_output must be boolean")

    suite = quality.get("suite_id")
    evaluator = quality.get("evaluator")
    if not isinstance(suite, str) or not suite.strip() or len(suite) > 256:
        raise ValueError("quality.suite_id must identify the evaluation dataset")
    if not isinstance(evaluator, str) or not evaluator.strip() or len(evaluator) > 256:
        raise ValueError("quality.evaluator must identify the independent scoring method")
    cases = quality.get("cases")
    if not isinstance(cases, list):
        raise ValueError("quality.cases must be a list")
    identifiers: set[str] = set()
    base_scores: list[float] = []
    q4_scores: list[float] = []
    losses: list[tuple[str, float]] = []
    for i, entry in enumerate(cases):
        if not isinstance(entry, dict):
            raise ValueError(f"quality.cases[{i}] must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
            raise ValueError(f"quality.cases[{i}].id must be a nonempty string <=128 characters")
        if identifier in identifiers:
            raise ValueError(f"duplicate quality case id: {identifier}")
        identifiers.add(identifier)
        baseline_score = _score(entry.get("baseline_score"), f"quality.cases[{i}].baseline_score")
        q4_score = _score(entry.get("q4_score"), f"quality.cases[{i}].q4_score")
        base_scores.append(baseline_score)
        q4_scores.append(q4_score)
        losses.append((identifier, baseline_score - q4_score))

    # Absence of scored quality must FAIL rather than be confused with parity.
    mean_loss = mean(v for _, v in losses) if losses else None
    worst_case = max(losses, key=lambda x: x[1]) if losses else None
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks.append({"name": name, "passed": bool(passed),
                       "observed": observed, "requirement": requirement})

    check("minimum_quality_cases", len(cases) >= policy.min_cases,
          len(cases), f">= {policy.min_cases} independent paired scores")
    check("minimum_timing_repeats", len(baseline_times) >= policy.min_repeats,
          len(baseline_times), f">= {policy.min_repeats} paired repetitions")
    check("checkpoint_payload_saved", saved_fraction >= policy.min_bytes_saved,
          saved_fraction, f">= {policy.min_bytes_saved:g} payload bytes saved")
    slowdown = q_median / b_median - 1.0
    check("median_generation_slowdown", slowdown <= policy.max_median_slowdown,
          slowdown, f"<= {policy.max_median_slowdown:g} fractional slowdown")
    check("stable_greedy_runs", b["stable_output"] and q["stable_output"],
          {"baseline": b["stable_output"], "q4": q["stable_output"]},
          "both variants repeat their own output deterministically")
    check("mean_quality_loss", mean_loss is not None and mean_loss <= policy.max_mean_quality_loss,
          mean_loss, f"<= {policy.max_mean_quality_loss:g} absolute score points on [0,1]")
    check("worst_case_quality_loss",
          worst_case is not None and worst_case[1] <= policy.max_case_quality_loss,
          {"id": worst_case[0], "loss": worst_case[1]} if worst_case else None,
          f"<= {policy.max_case_quality_loss:g} absolute score points on [0,1]")

    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "quality_suite": suite,
        "evaluator": evaluator,
        "quality_cases": len(cases),
        "baseline_mean_score": mean(base_scores) if base_scores else None,
        "q4_mean_score": mean(q4_scores) if q4_scores else None,
        "median_generation_speedup_q4_over_baseline": speedup,
        "checkpoint_payload_saved_fraction": saved_fraction,
        "limitations": [
            "Quality scores are externally supplied; this tool does not execute or audit the evaluator.",
            "Q4 is lossy: exact decoded-text parity is not a semantic quality metric.",
            "Generation timing uses sequential wall time, not GPU-only decode throughput.",
            "The existing Q4 benchmark runs baseline first: OS page-cache/order bias is possible.",
            "Weights/tokenizer/hardware/prompt identity and data leakage must be checked externally.",
            "No p95 TTFT/TPOT, GPU utilization, energy, FLOPs or deployment cost is measured.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fail-closed adoption gate for measured Q4 + scored quality")
    p.add_argument("benchmark", type=Path, help="ironasu-benchmark-q4 JSON report")
    p.add_argument("--quality", type=Path, required=True, help="independent paired quality scores JSON")
    p.add_argument("--min-cases", type=int, default=10)
    p.add_argument("--min-repeats", type=int, default=3)
    p.add_argument("--min-bytes-saved", type=float, default=0.10)
    p.add_argument("--max-median-slowdown", type=float, default=0.05)
    p.add_argument("--max-mean-quality-loss", type=float, default=0.01)
    p.add_argument("--max-case-quality-loss", type=float, default=0.05)
    p.add_argument("--output", type=Path, help="optional verdict JSON")
    args = p.parse_args(argv)
    try:
        policy = Q4GatePolicy(
            min_cases=args.min_cases, min_repeats=args.min_repeats,
            min_bytes_saved=args.min_bytes_saved, max_median_slowdown=args.max_median_slowdown,
            max_mean_quality_loss=args.max_mean_quality_loss,
            max_case_quality_loss=args.max_case_quality_loss,
        )
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
        quality = json.loads(args.quality.read_text(encoding="utf-8"))
        verdict = evaluate_q4(benchmark, quality, policy)
        report = json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        print(report, end="")
        return 0 if verdict["passed"] else 1
    except (ValueError, TypeError, OSError) as exc:
        p.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
