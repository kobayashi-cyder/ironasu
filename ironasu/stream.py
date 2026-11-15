"""Local Hugging Face safetensors model loader for ironasu's out-of-core engine.

No remote-code execution, pickle weights, automatic download, or implicit deletion
of model files. Call from a trusted *local model directory* only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .paging import LayerPager, PagingError, SafetensorsIndex


class PagedCausalLM:
    def __init__(self, model: Any, tokenizer: Any, pager: LayerPager, *,
                 kv_cache_mb: int = 0, kv_cache_entries: int = 8):
        from .kv_cache import PrefixKVCache
        if type(kv_cache_mb) is not int or kv_cache_mb < 0:
            raise ValueError("kv_cache_mb must be a nonnegative integer")
        self.model = model
        self.tokenizer = tokenizer
        self.pager = pager
        self.kv_cache = (PrefixKVCache(max_bytes=kv_cache_mb * 1024**2,
                                       max_entries=kv_cache_entries)
                         if kv_cache_mb else None)

    @classmethod
    def from_local_dir(cls, directory: str | Path, *, device: str = "cpu",
                       sync_timing: bool = False, prefetch: bool = False,
                       cpu_cache_mb: int = 0, pin_stages: list[str] | None = None,
                       q4_dir: str | Path | None = None,
                       expert_paging: bool = False, expert_cache_mb: int = 0,
                       use_mxfp4: bool = False, kv_cache_mb: int = 0,
                       kv_cache_entries: int = 8,
                       prefetch_mode: str = "fixed") -> "PagedCausalLM":
        try:
            import torch
            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PagingError("Install ironasu[stream] (requires transformers and accelerate)") from exc
        if q4_dir is not None and use_mxfp4:
            raise PagingError("--q4-dir and --native-mxfp4 are mutually exclusive")
        path = Path(directory).expanduser().resolve(strict=True)
        if q4_dir is None:
            index = SafetensorsIndex(path)
        else:
            from .q4 import Q4Index
            index = Q4Index(q4_dir)
        if use_mxfp4:
            from .mxfp4 import MXFP4Index
            index = MXFP4Index(index)
        # Local-only: prohibit downloading weights, tokenizers and remote Python code.
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, attn_implementation="eager")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        pager = LayerPager(model, index, device=device, sync_timing=sync_timing,
                           prefetch=prefetch, cpu_cache_mb=cpu_cache_mb,
                           pin_stages=pin_stages, expert_paging=expert_paging,
                           expert_cache_mb=expert_cache_mb,
                           prefetch_mode=prefetch_mode)
        return cls(model, tokenizer, pager,
                   kv_cache_mb=kv_cache_mb, kv_cache_entries=kv_cache_entries)

    def generate(self, prompt: str, *, max_new_tokens: int = 16) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pager.device)
        with self.pager.session() as model, torch.inference_mode():
            if self.kv_cache is None:
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = output[0, inputs["input_ids"].shape[-1]:]
            else:
                from .kv_cache import greedy_decode_with_exact_prefix
                generated = greedy_decode_with_exact_prefix(
                    model, inputs, max_new_tokens=max_new_tokens,
                    kv_cache=self.kv_cache, eos_token_id=self.tokenizer.eos_token_id)[0]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def reset_kv_cache(self) -> None:
        """Invalidate cached prefixes before changing weights or tokenizer."""
        if self.kv_cache is not None:
            self.kv_cache.reset()

    def kv_cache_snapshot(self) -> dict[str, Any]:
        return (self.kv_cache.snapshot() if self.kv_cache is not None
                else {"enabled": False})

    def close(self) -> None:
        self.reset_kv_cache()
        self.pager.close()


def main() -> None:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Out-of-core local safetensors text generation (experimental)")
    parser.add_argument("model_dir", help="trusted LOCAL Hugging Face directory containing safetensors")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--q4-dir", default=None, help="optional separately packed ironasu Q4 checkpoint; lossy")
    parser.add_argument("--profile", action="store_true", help="synchronize CUDA for stage timing; slower")
    parser.add_argument("--prefetch", action="store_true", help="bounded one-stage CPU prefetch; needs extra RAM")
    parser.add_argument("--prefetch-mode", choices=("fixed", "adaptive"), default="fixed",
                        help="adaptive mode temporarily skips unhelpful speculative reads")
    parser.add_argument("--cpu-cache-mb", type=int, default=0,
                        help="pin up to this many MiB of selected checkpoint tensors in CPU RAM; 0 disables")
    parser.add_argument("--pin-stage", action="append", default=None,
                        help="cache only this stage (repeatable); e.g. lm_head, model.embed_tokens")
    parser.add_argument("--native-mxfp4", action="store_true",
                        help="unpack native MXFP4 weight_packed/weight_scale pairs (reference CPU/CUDA decoder)")
    parser.add_argument("--expert-paging", action="store_true",
                        help="sparse ModuleList expert hooks; only selected experts loaded")
    parser.add_argument("--expert-cache-mb", type=int, default=0,
                        help="bounded CPU LRU of routed experts in MiB; requires --expert-paging")
    parser.add_argument("--kv-cache-mb", type=int, default=0,
                        help="opt-in exact-token greedy KV reuse; payload MiB limit, 0 disables")
    parser.add_argument("--kv-cache-entries", type=int, default=8,
                        help="max number of cached prompts, 1..128")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat the same prompt (useful for KV warm-hit test), 1..100")
    parser.add_argument("--auto-tune", action="store_true",
                        help="opt-in: compare paging strategies on a short same-prompt trial before generating")
    parser.add_argument("--tune-tokens", type=int, default=2,
                        help="tokens per trial (default: 2; each candidate regenerates them)")
    parser.add_argument("--tune-repeats", type=int, default=2,
                        help="per-strategy repetitions, 1..10 (default 2)")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100:
        parser.error("--repeat must be in 1..100")
    if args.kv_cache_mb < 0 or not 1 <= args.kv_cache_entries <= 128:
        parser.error("--kv-cache-mb must be >=0 and --kv-cache-entries in 1..128")
    if args.prefetch_mode == "adaptive" and not args.prefetch:
        parser.error("--prefetch-mode adaptive requires --prefetch")
    selected = {"prefetch": args.prefetch, "cpu_cache_mb": args.cpu_cache_mb,
                "pin_stages": args.pin_stage}
    if args.native_mxfp4 and args.q4_dir is not None:
        parser.error("--native-mxfp4 and --q4-dir cannot be combined")
    if args.auto_tune and args.native_mxfp4:
        parser.error("--auto-tune with --native-mxfp4 is not yet supported")
    if args.auto_tune and args.expert_paging:
        parser.error("--auto-tune with --expert-paging is not yet supported")
    if args.auto_tune and args.q4_dir is not None:
        parser.error("--auto-tune with --q4-dir is not supported yet; use ironasu-benchmark-q4")
    if args.auto_tune:
        if args.pin_stage is not None or args.prefetch:
            parser.error("--auto-tune cannot be combined with manually selected --pin-stage/--prefetch")
        from .tuner import tune_local_model
        report = tune_local_model(args.model_dir, args.prompt, device=args.device,
                                  tokens=args.tune_tokens, cache_mb=args.cpu_cache_mb,
                                  repeats=args.tune_repeats)
        selected = report["selected_arguments"]
        print(json.dumps({"ironasu_compass": report}, indent=2, ensure_ascii=False))
    engine = PagedCausalLM.from_local_dir(
        args.model_dir, device=args.device, sync_timing=args.profile,
        q4_dir=args.q4_dir, expert_paging=args.expert_paging,
        expert_cache_mb=args.expert_cache_mb, use_mxfp4=args.native_mxfp4,
        kv_cache_mb=args.kv_cache_mb, kv_cache_entries=args.kv_cache_entries,
        prefetch_mode=args.prefetch_mode, **selected,
    )
    try:
        for _ in range(args.repeat):
            print(engine.generate(args.prompt, max_new_tokens=args.tokens))
        print(json.dumps({"pager": engine.pager.snapshot(),
                          "kv_cache": engine.kv_cache_snapshot()},
                         ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
