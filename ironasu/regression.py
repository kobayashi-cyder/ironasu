"""Fail-closed, hardware-agnostic gate for measured ironasu A/B reports.

Consumes the JSON produced by ironasu-benchmark-ab. This gate does not infer
GPU utilization, energy, decoding-only speed or semantic correctness.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be numeric")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return float(value)


def _positive(value: Any, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _count(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


@dataclass(frozen=True)
class GatePolicy:
    min_samples: int = 10
    max_p95_regression: float = 0.05
    min_throughput_gain: float = 0.0
    latency_only: bool = False

    def __post_init__(self) -> None:
        _count(self.min_samples, "min_samples", positive=True)
        if not 0 <= _number(self.max_p95_regression, "max_p95_regression") <= 1:
            raise ValueError("max_p95_regression must be <= 1")
        if not 0 <= _number(self.min_throughput_gain, "min_throughput_gain") <= 10:
            raise ValueError("min_throughput_gain must be <= 10")
        if type(self.latency_only) is not bool:
            raise ValueError("latency_only must be a bool")


def evaluate_ab(report: dict[str, Any], policy: GatePolicy | None = None) -> dict[str, Any]:
    """Gate paired sequential HTTP trials; reject missing evidence by default.

    PASS means the measured regression thresholds were met, not that a model's
    answers are correct or GPU efficiency is proven. `latency_only` requires
    an explicit choice and never claims a throughput measurement.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    policy = policy if policy is not None else GatePolicy()
    b, c = report.get("baseline"), report.get("candidate")
    if not isinstance(b, dict) or not isinstance(c, dict):
        raise ValueError("missing baseline/candidate summaries")
    prompts = _count(report.get("prompts"), "prompts", positive=True)
    repeats = _count(report.get("repeats"), "repeats", positive=True)
    samples = prompts * repeats
    b_samples = _count(b.get("samples"), "baseline.samples", positive=True)
    c_samples = _count(c.get("samples"), "candidate.samples", positive=True)
    parity = _number(report.get("exact_output_match_rate"), "exact_output_match_rate")
    if parity > 1:
        raise ValueError("exact_output_match_rate must be <= 1")
    mismatches = report.get("mismatches")
    if not isinstance(mismatches, list):
        raise ValueError("mismatches must be a list")
    b_p95 = _positive(b.get("p95_seconds"), "baseline.p95_seconds")
    c_p95 = _positive(c.get("p95_seconds"), "candidate.p95_seconds")
    b_p50 = _positive(b.get("p50_seconds"), "baseline.p50_seconds")
    c_p50 = _positive(c.get("p50_seconds"), "candidate.p50_seconds")

    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks.append({"name": name, "passed": bool(passed),
                       "observed": observed, "requirement": requirement})

    check("sample_count", b_samples == c_samples == samples and samples >= policy.min_samples,
          {"baseline": b_samples, "candidate": c_samples},
          f"equal to prompts*repeats and >= {policy.min_samples}")
    check("exact_output_parity", parity == 1.0 and not mismatches,
          parity, "100% exact decoded text parity, no mismatches")
    p95_ratio = c_p95 / b_p95
    check("p95_latency", p95_ratio <= 1 + policy.max_p95_regression,
          p95_ratio, f"candidate/baseline <= {1 + policy.max_p95_regression:g}")

    throughput_ratio: float | None = None
    if not policy.latency_only:
        bt = _count(b.get("measured_output_tokens"), "baseline.measured_output_tokens", positive=True)
        ct = _count(c.get("measured_output_tokens"), "candidate.measured_output_tokens", positive=True)
        check("equal_token_counts", bt == ct, {"baseline": bt, "candidate": ct},
              "same positive total decoded token count")
        b_rate = _positive(b.get("throughput_tokens_per_wall_second"),
                           "baseline.throughput_tokens_per_wall_second")
        c_rate = _positive(c.get("throughput_tokens_per_wall_second"),
                           "candidate.throughput_tokens_per_wall_second")
        throughput_ratio = c_rate / b_rate
        check("sequential_throughput", throughput_ratio >= 1 + policy.min_throughput_gain,
              throughput_ratio, f"candidate/baseline >= {1 + policy.min_throughput_gain:g}")

    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "p50_latency_ratio": c_p50 / b_p50,
        "p95_latency_ratio": p95_ratio,
        "sequential_throughput_ratio": throughput_ratio,
        "throughput_evaluated": not policy.latency_only,
        "samples": samples,
        "limitations": [
            "Comparability of hardware, weights, tokenizer, request settings and load must be verified externally.",
            "HTTP latency includes queue/network time; serialized token rate is not concurrent throughput.",
            "Exact text parity is not semantic correctness; quality evaluation must be conducted separately.",
            "GPU/VRAM/energy metrics and p95 TTFT/TPOT are not collected by this report.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed regression gate for ironasu-benchmark-ab JSON")
    parser.add_argument("report", type=Path, help="JSON report from ironasu-benchmark-ab")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-p95-regression", type=float, default=0.05,
                        help="maximum allowed fractional p95 slowdown (default: 0.05)")
    parser.add_argument("--min-throughput-gain", type=float, default=0.0,
                        help="minimum fractional serialized tokens/s improvement (default: 0)")
    parser.add_argument("--latency-only", action="store_true",
                        help="explicitly skip token-usage/throughput check when server omits usage")
    parser.add_argument("--output", type=Path, help="optional JSON verdict file")
    args = parser.parse_args(argv)
    try:
        policy = GatePolicy(args.min_samples, args.max_p95_regression,
                            args.min_throughput_gain, args.latency_only)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        verdict = evaluate_ab(report, policy)
        serialized = json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if verdict["passed"] else 1
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
