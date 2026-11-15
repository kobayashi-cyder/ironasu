"""Local Hugging Face safetensors model loader for ironasu's out-of-core engine.

No remote-code execution, pickle weights, automatic download, or implicit deletion
of model files. Call from a trusted *local model directory* only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .paging import LayerPager, PagingError, SafetensorsIndex


class PagedCausalLM:
    def __init__(self, model: Any, tokenizer: Any, pager: LayerPager):
        self.model = model
        self.tokenizer = tokenizer
        self.pager = pager

    @classmethod
    def from_local_dir(cls, directory: str | Path, *, device: str = "cpu",
                       sync_timing: bool = False, prefetch: bool = False,
                       cpu_cache_mb: int = 0, pin_stages: list[str] | None = None,
                       q4_dir: str | Path | None = None) -> "PagedCausalLM":
        try:
            import torch
            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PagingError("Install ironasu[stream] (requires transformers and accelerate)") from exc
        path = Path(directory).expanduser().resolve(strict=True)
        if q4_dir is None:
            index = SafetensorsIndex(path)
        else:
            from .q4 import Q4Index
            index = Q4Index(q4_dir)
        # Local-only: prohibit downloading weights, tokenizers and remote Python code.
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, attn_implementation="eager")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        pager = LayerPager(model, index, device=device, sync_timing=sync_timing,
                           prefetch=prefetch, cpu_cache_mb=cpu_cache_mb,
                           pin_stages=pin_stages)
        return cls(model, tokenizer, pager)

    def generate(self, prompt: str, *, max_new_tokens: int = 16) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pager.device)
        with self.pager.session() as model, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def close(self) -> None:
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
    parser.add_argument("--cpu-cache-mb", type=int, default=0,
                        help="pin up to this many MiB of selected checkpoint tensors in CPU RAM; 0 disables")
    parser.add_argument("--pin-stage", action="append", default=None,
                        help="cache only this stage (repeatable); e.g. lm_head, model.embed_tokens")
    parser.add_argument("--auto-tune", action="store_true",
                        help="opt-in: compare paging strategies on a short same-prompt trial before generating")
    parser.add_argument("--tune-tokens", type=int, default=2,
                        help="tokens per trial (default: 2; each candidate regenerates them)")
    parser.add_argument("--tune-repeats", type=int, default=2,
                        help="per-strategy repetitions, 1..10 (default 2)")
    args = parser.parse_args()
    selected = {"prefetch": args.prefetch, "cpu_cache_mb": args.cpu_cache_mb,
                "pin_stages": args.pin_stage}
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
        q4_dir=args.q4_dir, **selected,
    )
    try:
        print(engine.generate(args.prompt, max_new_tokens=args.tokens))
        print(json.dumps(engine.pager.snapshot(), ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
