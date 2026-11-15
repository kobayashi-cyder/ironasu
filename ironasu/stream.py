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
        self.last_speculative_report = None
        self._kimi_text_only = False

    @classmethod
    def from_local_dir(cls, directory: str | Path, *, device: str = "cpu",
                       sync_timing: bool = False, prefetch: bool = False,
                       cpu_cache_mb: int = 0, pin_stages: list[str] | None = None,
                       q4_dir: str | Path | None = None,
                       expert_paging: bool = False, expert_cache_mb: int = 0,
                       use_mxfp4: bool = False,
                       kimi_text_only: bool = False,
                       trust_local_code: bool = False) -> "PagedCausalLM":
        # Code from a model directory is executable Python, not trusted data.
        # Kimi must be an explicit opt-in: never enable it merely based on config.
        if kimi_text_only and not trust_local_code:
            raise PagingError("Kimi text-only requires explicit trust_local_code=True")
        if trust_local_code and not kimi_text_only:
            raise PagingError("trust_local_code is only supported with kimi_text_only")
        if kimi_text_only and q4_dir is not None:
            raise PagingError("Kimi text-only currently requires original native MXFP4 safetensors")
        try:
            import torch
            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PagingError("Install ironasu[stream] (requires transformers and accelerate)") from exc
        if q4_dir is not None and use_mxfp4:
            raise PagingError("--q4-dir and --native-mxfp4 are mutually exclusive")
        path = Path(directory).expanduser().resolve(strict=True)
        if kimi_text_only:
            from .kimi import require_local_kimi_code, inspect_kimi_config
            # Reject unknown architecture, missing files and unexpected auto_map
            # before executing model-supplied Python. This is not a code audit.
            inspect_kimi_config(path)
            require_local_kimi_code(path)
            use_mxfp4 = True  # K3's routed experts are natively MXFP4.
            expert_paging = True  # Never materialize all 896 experts at once.
        if q4_dir is None:
            index = SafetensorsIndex(path)
        else:
            from .q4 import Q4Index
            index = Q4Index(q4_dir)
        if use_mxfp4:
            from .mxfp4 import MXFP4Index
            index = MXFP4Index(index)
        if kimi_text_only:
            from .kimi import TextWeightView
            index = TextWeightView(index)
            # The official K3 multimodal wrapper contains a 401M vision tower.
            # Instantiate ONLY its text model and map the original checkpoint
            # keys into it. No full checkpoint is loaded by Transformers.
            config = AutoConfig.from_pretrained(
                path, local_files_only=True, trust_remote_code=True)
            # K3's nested text_config has _name_or_path=""; AutoModel's
            # dynamic class lookup could resolve that empty path incorrectly.
            # Import only the exact reviewed class from this local directory.
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            model_cls = get_class_from_dynamic_module(
                "modeling_kimi_linear.KimiLinearForCausalLM",
                str(path), local_files_only=True)
            with init_empty_weights(include_buffers=False):
                model = model_cls(config.text_config)
        else:
            # All other architectures: NEVER run model repository Python.
            config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=False, attn_implementation="eager")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=kimi_text_only)
        pager = LayerPager(model, index, device=device, sync_timing=sync_timing,
                           prefetch=prefetch, cpu_cache_mb=cpu_cache_mb,
                           pin_stages=pin_stages, expert_paging=expert_paging,
                           expert_cache_mb=expert_cache_mb)
        instance = cls(model, tokenizer, pager)
        instance._kimi_text_only = kimi_text_only
        return instance

    def generate(self, prompt: str, *, max_new_tokens: int = 16,
                 speculative_ngram: bool = False) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pager.device)
        if speculative_ngram:
            if self._kimi_text_only:
                raise PagingError("speculative N-gram path is not validated for Kimi text-only")
            from .speculative_decode import GreedySpeculativeDecoder, PagedTorchGreedyTarget
            tokens = tuple(int(x) for x in inputs["input_ids"][0].tolist())
            eos_value = self.tokenizer.eos_token_id
            eos_ids = ([eos_value] if isinstance(eos_value, int) else
                       (eos_value or []))
            with self.pager.session() as model, torch.inference_mode():
                target = PagedTorchGreedyTarget(model, torch, str(self.pager.device))
                decoder = GreedySpeculativeDecoder(target, synchronize=target.synchronize)
                report = decoder.generate(tokens, max_new_tokens=max_new_tokens,
                                          eos_token_ids=eos_ids)
                self.last_speculative_report = report
            return self.tokenizer.decode(report.new_tokens, skip_special_tokens=True)
        self.last_speculative_report = None
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
    parser.add_argument("--kimi-text-only", action="store_true",
                        help="experimental Kimi K3 text-only adapter (native MXFP4 + sparse experts)")
    parser.add_argument("--trust-local-code", action="store_true",
                        help="explicitly execute LOCAL Kimi model Python; review files first")
    parser.add_argument("--native-mxfp4", action="store_true",
                        help="unpack native MXFP4 weight_packed/weight_scale pairs (reference CPU/CUDA decoder)")
    parser.add_argument("--expert-paging", action="store_true",
                        help="sparse ModuleList expert hooks; only selected experts loaded")
    parser.add_argument("--expert-cache-mb", type=int, default=0,
                        help="bounded CPU LRU of routed experts in MiB; requires --expert-paging")
    parser.add_argument("--speculative-ngram", action="store_true",
                        help="opt-in exact greedy target verification; may be slower, KV crop or replay fallback")
    parser.add_argument("--auto-tune", action="store_true",
                        help="opt-in: compare paging strategies on a short same-prompt trial before generating")
    parser.add_argument("--tune-tokens", type=int, default=2,
                        help="tokens per trial (default: 2; each candidate regenerates them)")
    parser.add_argument("--tune-repeats", type=int, default=2,
                        help="per-strategy repetitions, 1..10 (default 2)")
    args = parser.parse_args()
    selected = {"prefetch": args.prefetch, "cpu_cache_mb": args.cpu_cache_mb,
                "pin_stages": args.pin_stage}
    if args.native_mxfp4 and args.q4_dir is not None:
        parser.error("--native-mxfp4 and --q4-dir cannot be combined")
    if args.kimi_text_only != args.trust_local_code:
        parser.error("--kimi-text-only requires --trust-local-code (and vice versa)")
    if args.kimi_text_only and args.auto_tune:
        parser.error("--auto-tune is not supported with --kimi-text-only")
    if args.kimi_text_only and args.q4_dir is not None:
        parser.error("Kimi text-only requires original native MXFP4 weights")
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
        kimi_text_only=args.kimi_text_only,
        trust_local_code=args.trust_local_code, **selected,
    )
    try:
        print(engine.generate(args.prompt, max_new_tokens=args.tokens,
                              speculative_ngram=args.speculative_ngram))
        if engine.last_speculative_report is not None:
            from dataclasses import asdict
            print(json.dumps({"speculative_decode": asdict(engine.last_speculative_report)},
                             ensure_ascii=False, indent=2))
        print(json.dumps(engine.pager.snapshot(), ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
