"""Reproducible local-only A/B wall-time tests for PagedCausalLM.

These are not TTFT, end-to-end FLOP counts, or physical SSD read metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

MODES = ("baseline", "fixed_prefetch", "adaptive_prefetch", "kv", "kv_adaptive")


def percentile(values: list[float], rank: float) -> float:
    if not values or not 0 <= rank <= 100:
        raise ValueError("invalid percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * rank / 100
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Never report speed improvement if text output or reference is unstable."""
    before, after = records.get("baseline", []), records.get("baseline_after", [])
    if not before or not after:
        raise ValueError("before and after baselines required")
    for runs in records.values():
        if not runs or any(not isinstance(r["output"], str) or r["seconds"] <= 0 for r in runs):
            raise ValueError("all trials need positive seconds and text output")
    correct = before[0]["output"]
    stable = all(row["output"] == correct for row in before + after)
    reference = median([row["seconds"] for row in after[1:]] or
                       [row["seconds"] for row in after])
    compared = {}
    for mode, rows in records.items():
        wall = [row["seconds"] for row in rows]
        warm = wall[1:] or wall
        matches = stable and all(row["output"] == correct for row in rows)
        warm_median = median(warm)
        compared[mode] = {
            "runs": len(rows),
            "cold_seconds": round(wall[0], 6),
            "warm_p50_seconds": round(warm_median, 6),
            "warm_p95_seconds": round(percentile(warm, 95), 6),
            "output_matches_baseline": matches,
            "warm_relative_gain": (round((reference - warm_median) / reference, 5)
                                   if matches else None),
            "final_kv_stats": rows[-1].get("kv_cache"),
            "pager_read_payload_bytes": rows[-1].get("pager_read_payload_bytes"),
            "peak_cuda_allocated_bytes": (max(row.get("peak_cuda_allocated_bytes", 0)
                                             for row in rows) or None),
        }
    return {
        "reference_stable": stable,
        "reference_warm_p50_seconds": round(reference, 6),
        "comparisons": compared,
        "warning": "Wall-time != TTFT. Read payload bytes != actual physical SSD "
                   "reads. Output equality for one prompt != general model parity. "
                   "Cloning KV can be slower than recomputation.",
    }


def run_benchmark(directory: str | Path, prompt: str, *, device: str = "cpu",
                  tokens: int = 8, repeats: int = 3, kv_cache_mb: int = 256,
                  modes: tuple[str, ...] = ("baseline", "kv")) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt required")
    if type(tokens) is not int or not 1 <= tokens <= 1024:
        raise ValueError("tokens must be 1..1024")
    if type(repeats) is not int or not 2 <= repeats <= 20:
        raise ValueError("repeats must be 2..20")
    if type(kv_cache_mb) is not int or not 0 <= kv_cache_mb <= 32768:
        raise ValueError("kv_cache_mb must be 0..32768")
    if not modes or len(modes) != len(set(modes)) or any(m not in MODES for m in modes):
        raise ValueError("invalid modes")
    from .stream import PagedCausalLM
    import torch
    results = {}
    # Later baseline detects OS page cache, power and thermal drift.
    trial_modes = ("baseline", *(m for m in modes if m != "baseline"), "baseline_after")
    for mode in trial_modes:
        prefetch = mode in ("fixed_prefetch", "adaptive_prefetch", "kv_adaptive")
        adaptive = mode in ("adaptive_prefetch", "kv_adaptive")
        cache = kv_cache_mb if mode in ("kv", "kv_adaptive") else 0
        if mode in ("kv", "kv_adaptive") and not cache:
            raise ValueError("KV modes require positive kv_cache_mb")
        engine = PagedCausalLM.from_local_dir(
            directory, device=device, prefetch=prefetch,
            prefetch_mode="adaptive" if adaptive else "fixed",
            kv_cache_mb=cache)
        observations = []
        try:
            for _ in range(repeats):
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                output = engine.generate(prompt, max_new_tokens=tokens)
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = perf_counter() - start
                row = {
                    "seconds": seconds, "output": output,
                    "kv_cache": engine.kv_cache_snapshot(),
                    "pager_read_payload_bytes":
                        engine.pager.snapshot()["cpu_cache"]["disk_read_bytes"],
                }
                if device == "cuda":
                    row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                observations.append(row)
        finally:
            engine.close()
        results[mode] = observations
    report = summarize(results)
    report["config"] = {
        "device": device, "tokens": tokens, "repeats": repeats,
        "modes": list(modes), "kv_cache_mb": kv_cache_mb,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Local paged LLM A/B wall-time test")
    parser.add_argument("model_dir", help="trusted local HF safetensors directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kv-cache-mb", type=int, default=256)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["baseline", "kv"])
    parser.add_argument("--output", help="JSON output file path")
    args = parser.parse_args()
    report = run_benchmark(
        args.model_dir, args.prompt, device=args.device,
        tokens=args.tokens, repeats=args.repeats,
        kv_cache_mb=args.kv_cache_mb, modes=tuple(args.modes))
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
