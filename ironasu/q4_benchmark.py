"""Paired baseline/Q4 timing; report evidence, never claim improvement by default.

This benchmark executes the exact same greedy request on both checkpoints.
Output parity is informational: Q4 is lossy. Application-level quality tests
are needed before deciding whether a Q4 result is acceptable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .stream import PagedCausalLM


def _measure(model_dir: str, q4_dir: str | None, prompt: str, tokens: int,
             device: str, repeats: int, prefetch: bool, cache_mb: int):
    # One engine at a time; avoids needing both model pagers in memory.
    started = time.perf_counter()
    engine = PagedCausalLM.from_local_dir(model_dir, device=device, q4_dir=q4_dir,
                                          sync_timing=(device == "cuda"), prefetch=prefetch,
                                          cpu_cache_mb=cache_mb)
    load_s = time.perf_counter() - started
    try:
        times, outputs = [], []
        for _ in range(repeats):
            before = time.perf_counter()
            result = engine.generate(prompt, max_new_tokens=tokens)
            times.append(time.perf_counter() - before)
            outputs.append(result)
        stats = engine.pager.snapshot()
        return {"model_setup_s": load_s, "generation_s": times,
                "median_generation_s": statistics.median(times),
                "generated_text": outputs[-1], "stable_output": len(set(outputs)) == 1,
                "cpu_cache": stats["cpu_cache"], "stage_count": stats["stage_count"]}
    finally:
        engine.close()


def compare_q4(model_dir: str, q4_dir: str, *, prompt: str, tokens: int = 4,
               repeats: int = 2, device: str = "cpu", prefetch: bool = False,
               cache_mb: int = 0) -> dict:
    if not prompt or not 1 <= tokens <= 128 or not 1 <= repeats <= 10 or cache_mb < 0:
        raise ValueError("Invalid benchmark parameters")
    from .q4 import Q4Index
    from .paging import SafetensorsIndex
    ref_index = SafetensorsIndex(model_dir)
    q_index = Q4Index(q4_dir)
    if ref_index.keys() != q_index.keys():
        raise ValueError("Q4 tensor names differ from baseline checkpoint")
    names = sorted(ref_index.keys())
    ref_bytes = sum(ref_index.tensor_byte_sizes(names).values())
    q_bytes = sum(q_index.tensor_byte_sizes(names).values())
    # The baseline runs first; OS page caching may favor the second variant.
    # Repeating with inverse order on the target machine is recommended.
    baseline = _measure(model_dir, None, prompt, tokens, device, repeats, prefetch, cache_mb)
    compressed = _measure(model_dir, q4_dir, prompt, tokens, device, repeats, prefetch, cache_mb)
    speed_ratio = baseline["median_generation_s"] / compressed["median_generation_s"]
    return {"warning": "Q4 is lossy. Output comparison does not establish code correctness; order/page-cache bias exists.",
            "checkpoint_payload": {"baseline_bytes": ref_bytes, "q4_bytes": q_bytes,
                                    "q4_fraction": q_bytes / ref_bytes},
            "baseline": baseline, "q4": compressed,
            "speed_ratio_q4_vs_baseline": speed_ratio,
            "identical_generated_text": (baseline["stable_output"] and compressed["stable_output"]
                                         and baseline["generated_text"] == compressed["generated_text"]),
            "recommendation": ("validate_on_programming_test_suite" if speed_ratio > 1.05
                               else "do_not_enable_for_speed")}


def main():
    p = argparse.ArgumentParser(description="Compare original safetensors vs lossily quantized Q4 pager")
    p.add_argument("model_dir")
    p.add_argument("--q4-dir", required=True)
    p.add_argument("--prompt", default="Explain binary search in Python.")
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--cpu-cache-mb", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    report = compare_q4(args.model_dir, args.q4_dir, prompt=args.prompt, tokens=args.tokens,
                        repeats=args.repeats, device=args.device, prefetch=args.prefetch,
                        cache_mb=args.cpu_cache_mb)
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(formatted, encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    main()
