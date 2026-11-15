"""Optional on-device AirLLM A/B benchmark. No model downloads on import.

Use the same model and prompt with prefetch on/off. Reports measured time, not
claimed FLOP reductions. AirLLM already implements prefetching itself.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Any
from .airllm_probe import AirLLMStageProbe


def _linux_read_bytes() -> int | None:
    try:
        with open("/proc/self/io", encoding="utf-8") as f:
            for row in f:
                if row.startswith("read_bytes:"):
                    return int(row.split(":", 1)[1].strip())
    except OSError:
        pass
    return None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_prefetch = {row["prefetch_requested"]: row for row in results}
    report: dict[str, Any] = {"trials": results}
    on, off = by_prefetch.get(True), by_prefetch.get(False)
    if on and off:
        different_effective = on.get("prefetch_effective", True) != off.get("prefetch_effective", False)
        same = on["output_token_ids"] == off["output_token_ids"]
        report["exact_token_match"] = same
        if same and different_effective and on["generation_seconds"] > 0:
            report["prefetch_speedup_ratio"] = round(off["generation_seconds"] / on["generation_seconds"], 3)
        else:
            report["prefetch_speedup_ratio"] = None
        report["note"] = (
            "Warm/cold OS page-cache effects and model load order can bias measurements. "
            "Repeat and alternate order; validate any output changes."
        )
    return report


def run_trial(*, model_id: str, prompt: str, device: str, max_new_tokens: int,
              max_seq_len: int, prefetch: bool, compression: str | None = None) -> dict[str, Any]:
    try:
        import torch
        from airllm import AutoModel
    except ImportError as exc:
        raise RuntimeError("Install compatible torch + airllm in a separate environment") from exc
    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_id,
        device=device,
        max_seq_len=max_seq_len,
        prefetching=prefetch,
        compression=compression,
        profiling_mode=True,
        delete_original=False,
    )
    loading_seconds = time.perf_counter() - start
    inputs = model.tokenizer(
        [prompt], return_tensors="pt", return_attention_mask=False,
        truncation=True, max_length=max_seq_len, padding=False,
    )["input_ids"].to(model.device)
    on_cuda = str(model.device).startswith("cuda")
    if on_cuda:
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    before_reads = _linux_read_bytes()
    started = time.perf_counter()
    with AirLLMStageProbe(model) as probe:
        with torch.inference_mode():
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens,
                                     use_cache=True, do_sample=False, return_dict_in_generate=True)
    phase_report = probe.snapshot()
    if on_cuda:
        torch.cuda.synchronize(model.device)
    seconds = time.perf_counter() - started
    after_reads = _linux_read_bytes()
    full_ids = outputs.sequences[0].tolist()
    generated = full_ids[inputs.shape[-1]:]
    record = {
        "model": model_id,
        "device": device,
        "compression": compression,
        "prefetch_requested": prefetch,
        "prefetch_effective": bool(getattr(model, "prefetching", False)),
        "model_load_seconds": round(loading_seconds, 4),
        "generation_seconds": round(seconds, 4),
        "generated_tokens": len(generated),
        "tokens_per_second": round(len(generated) / seconds, 4) if seconds else None,
        "output_token_ids": generated,
        "linux_process_read_bytes": after_reads - before_reads if after_reads is not None and before_reads is not None else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if on_cuda else None,
        "airllm_stages": phase_report,
        "warning": "Read bytes are process block I/O, not exclusively model SSD traffic; generation timer includes I/O and compute.",
    }
    del outputs, model, inputs
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark actual AirLLM on this hardware")
    parser.add_argument("--model", required=True, help="HF model ID or already-downloaded local path")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0 or mps")
    parser.add_argument("--prompt", default="Write a Python function to add two integers.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--compression", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--prefetch", choices=["off", "on", "both"], default="both")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or not 16 <= args.max_seq_len <= 131072:
        parser.error("token counts are out of supported benchmark bounds")
    choices = [False, True] if args.prefetch == "both" else [args.prefetch == "on"]
    results = [run_trial(
        model_id=args.model, prompt=args.prompt, device=args.device,
        max_new_tokens=args.max_new_tokens, max_seq_len=args.max_seq_len,
        prefetch=choice, compression=None if args.compression == "none" else args.compression,
    ) for choice in choices]
    output = json.dumps(summarize(results), ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
