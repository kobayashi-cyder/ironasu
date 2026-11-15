"""Reproducible A/B benchmark for two explicitly configured inference servers.

Start the servers with the *same* model, tokenizer, prompt template and greedy
sampling policy, one speculative and one baseline. Never infer GPU/SSD gains
from a local synthetic benchmark or HTTP latency alone.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import ceil, isfinite
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Completion:
    content: str
    elapsed_seconds: float
    completion_tokens: int | None

    def __post_init__(self):
        if not isinstance(self.content, str) or not isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("invalid completion or timing")
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError("completion token count cannot be negative")


def _pct(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot summarize empty data")
    return sorted(values)[max(0, ceil(len(values) * percentile / 100) - 1)]


def summarize(completions: list[Completion]) -> dict:
    latencies = [r.elapsed_seconds for r in completions]
    tokens = [r.completion_tokens for r in completions]
    measured_tokens = sum(t for t in tokens if t is not None)
    measured_seconds = sum(r.elapsed_seconds for r in completions
                           if r.completion_tokens is not None)
    return {
        "samples": len(latencies),
        "p50_seconds": median(latencies),
        "p95_seconds": _pct(latencies, 95),
        "mean_seconds": sum(latencies) / len(latencies),
        "measured_output_tokens": measured_tokens if all(t is not None for t in tokens) else None,
        "throughput_tokens_per_wall_second": (
            measured_tokens / measured_seconds
            if all(t is not None for t in tokens) and measured_seconds > 0 else None),
    }


def compare(
    prompts: Iterable[str],
    *,
    baseline: Callable[[str], Completion],
    candidate: Callable[[str], Completion],
    repeats: int = 2,
    warmup: int = 1,
) -> dict:
    cases = tuple(prompts)
    if not cases or len(cases) > 1000 or any(not isinstance(p, str) or not p for p in cases):
        raise ValueError("prompts must contain between 1 and 1000 nonempty strings")
    if not 1 <= repeats <= 100 or not 0 <= warmup <= 10:
        raise ValueError("invalid repeat or warmup count")
    # Warmups are excluded; do not claim throughput from them.
    for _ in range(warmup):
        baseline(cases[0])
        candidate(cases[0])
    b_rows: list[Completion] = []
    c_rows: list[Completion] = []
    mismatches: list[dict] = []
    for trial in range(repeats):
        for i, prompt in enumerate(cases):
            # Alternate ordering to reduce systematic warm-cache advantage.
            if (trial + i) % 2 == 0:
                b, c = baseline(prompt), candidate(prompt)
            else:
                c, b = candidate(prompt), baseline(prompt)
            b_rows.append(b)
            c_rows.append(c)
            if b.content != c.content:
                mismatches.append({"prompt_index": i, "repeat": trial})
    baseline_stats = summarize(b_rows)
    candidate_stats = summarize(c_rows)
    return {
        "prompts": len(cases), "repeats": repeats, "warmup": warmup,
        "baseline": baseline_stats, "candidate": candidate_stats,
        "exact_output_match_rate": 1 - len(mismatches) / len(b_rows),
        "mismatches": mismatches,  # indices only; never echo user prompts/outputs
        "p50_latency_speedup": baseline_stats["p50_seconds"] / candidate_stats["p50_seconds"],
        "limitations": [
            "HTTP wall-clock measurements include queue/network overhead, not just model decode.",
            "GPU/RAM/SSD/energy telemetry is not collected by this tool.",
            "A mismatch can reflect model/quantization/tokenizer/seed/server differences.",
            "A speedup is established only for the measured inputs and test environment.",
        ],
    }


def _url(base: str) -> str:
    base = base.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("endpoints must be http(s)")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def make_client(endpoint: str, model: str, key: str = "",
                max_tokens: int = 64, timeout: int = 600) -> Callable[[str], Completion]:
    url = _url(endpoint)
    if not model or not 1 <= max_tokens <= 1024 or not 1 <= timeout <= 7200:
        raise ValueError("invalid model, max_tokens or timeout")
    def complete(prompt: str) -> Completion:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = Request(url, data=body, headers=headers, method="POST")
        started = perf_counter()
        try:
            with urlopen(req, timeout=timeout) as response:
                data = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(f"completion server unavailable: {type(exc).__name__}") from exc
        elapsed = perf_counter() - started
        try:
            content = data["choices"][0]["message"]["content"]
            tokens = (data.get("usage") or {}).get("completion_tokens")
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("invalid usage")
            if not isinstance(content, str):
                raise ValueError("text response required")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise RuntimeError("server did not return a valid text chat completion") from exc
        return Completion(content, elapsed, tokens)
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure baseline vs speculative server, same prompts")
    parser.add_argument("--prompts", required=True, type=Path, help="UTF-8 JSON array of prompt strings")
    parser.add_argument("--baseline-url", required=True, help="non-speculative server /v1")
    parser.add_argument("--candidate-url", required=True, help="speculative server /v1")
    parser.add_argument("--model", required=True, help="same model ID on both servers")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, help="optional JSON report file")
    args = parser.parse_args(argv)
    try:
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        if not isinstance(prompts, list):
            raise ValueError("prompts must be a JSON array")
        key = os.getenv("IRONASU_BACKEND_KEY", "")
        report = compare(
            prompts,
            baseline=make_client(args.baseline_url, args.model, key,
                                 args.max_tokens, args.timeout),
            candidate=make_client(args.candidate_url, args.model, key,
                                  args.max_tokens, args.timeout),
            repeats=args.repeat, warmup=args.warmup)
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return int(report["exact_output_match_rate"] != 1)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
