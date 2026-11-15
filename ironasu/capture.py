"""Sequential benchmark captures: compare models that cannot coexist in memory.

Capture responses as hashes plus timing/token metadata. Save baseline first,
reconfigure the *same physical machine* for speculative inference, then record
candidate. No user prompts or model outputs are written to capture files.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

from .benchmark import Completion, make_client, summarize


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def record_capture(
    prompts: list[str],
    invoke: Callable[[str], Completion],
    *,
    model: str,
    max_tokens: int,
    repeats: int = 3,
    warmup: int = 1,
) -> dict:
    if not prompts or len(prompts) > 1000 or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("expected 1..1000 nonempty prompts")
    if not model or not 1 <= max_tokens <= 1024 or not 1 <= repeats <= 100 or not 0 <= warmup <= 10:
        raise ValueError("invalid capture configuration")
    for _ in range(warmup):
        invoke(prompts[0])
    rows = []
    for trial in range(repeats):
        for i, prompt in enumerate(prompts):
            completion = invoke(prompt)
            rows.append({
                "index": i,
                "trial": trial,
                "answer_sha256": _digest(completion.content),
                "seconds": completion.elapsed_seconds,
                "completion_tokens": completion.completion_tokens,
            })
    return {
        "schema": 1, "model": model, "max_tokens": max_tokens,
        "prompt_hashes": [_digest(p) for p in prompts],
        "repeats": repeats, "warmup": warmup, "rows": rows,
        "redacted": True,
    }


def compare_captures(baseline: dict, candidate: dict) -> dict:
    for key in ("schema", "model", "max_tokens", "prompt_hashes", "repeats"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"capture configurations differ: {key}")
    if baseline.get("schema") != 1 or baseline.get("redacted") is not True or candidate.get("redacted") is not True:
        raise ValueError("unknown capture schema")
    expected = len(baseline.get("prompt_hashes", [])) * baseline["repeats"]
    if expected < 1 or len(baseline.get("rows", [])) != expected or len(candidate.get("rows", [])) != expected:
        raise ValueError("capture row count does not match prompt/trial count")
    a, b = [], []
    mismatch = []
    for i, (left, right) in enumerate(zip(baseline["rows"], candidate["rows"])):
        if ((left["index"], left["trial"]) != (right["index"], right["trial"])
                or (left["index"], left["trial"]) != (
                    i % len(baseline["prompt_hashes"]), i // len(baseline["prompt_hashes"]))):
            raise ValueError("capture row ordering mismatch")
        a.append(Completion(left["answer_sha256"], left["seconds"], left["completion_tokens"]))
        b.append(Completion(right["answer_sha256"], right["seconds"], right["completion_tokens"]))
        if left["answer_sha256"] != right["answer_sha256"]:
            mismatch.append({"prompt_index": left["index"], "repeat": left["trial"]})
    a_summary, b_summary = summarize(a), summarize(b)
    return {
        "prompts": len(baseline["prompt_hashes"]),
        "repeats": baseline["repeats"], "baseline": a_summary,
        "candidate": b_summary,
        "exact_output_match_rate": 1 - len(mismatch) / expected,
        "mismatches": mismatch,
        "p50_latency_speedup": a_summary["p50_seconds"] / b_summary["p50_seconds"],
        "limitations": [
            "Separate capture times may reflect thermal, OS, disk-cache or hardware state changes.",
            "Tokens/s uses wall-clock time and requires reported token counts.",
            "VRAM/RAM, SSD/PCIe traffic and energy are not automatically measured.",
            "Never claim huge-model speedup without model-specific correctness and memory telemetry.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-device sequential LLM benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="record one server configuration")
    capture.add_argument("--prompts", type=Path, required=True)
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--max-tokens", type=int, default=64)
    capture.add_argument("--repeat", type=int, default=3)
    capture.add_argument("--warmup", type=int, default=1)
    compare_parser = sub.add_parser("compare", help="compare previously recorded configurations")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
            if not isinstance(prompts, list):
                raise ValueError("prompts JSON must be an array")
            client = make_client(args.endpoint, args.model,
                                 os.getenv("IRONASU_BACKEND_KEY", ""),
                                 args.max_tokens)
            result = record_capture(prompts, client, model=args.model,
                                    max_tokens=args.max_tokens,
                                    repeats=args.repeat, warmup=args.warmup)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"captured_samples": len(result["rows"]),
                              "file": str(args.output), "redacted": True}))
            return 0
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        report = compare_captures(baseline, candidate)
        serialized = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return int(report["exact_output_match_rate"] != 1)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
