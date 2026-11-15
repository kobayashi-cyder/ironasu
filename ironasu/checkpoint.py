"""Read-only checkpoint sizing for a local, trusted safetensors model.

Does not import Transformers, allocate a model, or load weight tensors. Stage
classification is based on checkpoint key names and is only an estimate; the
runtime pager makes its own architecture and shape checks.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paging import SafetensorsIndex

_BLOCK = re.compile(r"^(model\.layers|transformer\.h|gpt_neox\.layers)\.(\d+)\.")


def _stage_for_tensor(name: str) -> str:
    match = _BLOCK.match(name)
    if match:
        return f"{match.group(1)}.{int(match.group(2))}"
    for prefix in (
        "model.embed_tokens", "model.norm", "lm_head",
        "transformer.wte", "transformer.wpe", "transformer.ln_f",
        "gpt_neox.embed_in", "gpt_neox.final_layer_norm", "embed_out",
    ):
        if name.startswith(prefix + "."):
            return prefix
    return "unrecognized"


def inspect_checkpoint(directory: str | Path, *, cpu_cache_mb: int = 0) -> dict[str, Any]:
    if isinstance(cpu_cache_mb, bool) or not isinstance(cpu_cache_mb, int) or cpu_cache_mb < 0:
        raise ValueError("cpu_cache_mb must be a nonnegative integer")
    index = SafetensorsIndex(directory)
    names = sorted(index.keys())
    sizes = index.tensor_byte_sizes(names)
    stages: dict[str, int] = defaultdict(int)
    for name in names:
        stages[_stage_for_tensor(name)] += sizes[name]
    budget = cpu_cache_mb * 1024 * 1024
    priority = {
        "lm_head": 0, "embed_out": 0,
        "model.embed_tokens": 1, "transformer.wte": 1, "gpt_neox.embed_in": 1,
    }
    # Preserves stable input order for equal-priority stages, by stage number.
    stage_names = sorted(stages, key=lambda name: (
        priority.get(name, 2),
        int(_BLOCK.match(name + ".weight").group(2)) if _BLOCK.match(name + ".weight") else -1,
        name,
    ))
    pinned: list[str] = []
    free = budget
    if budget:
        for name in stage_names:
            if name != "unrecognized" and stages[name] <= free:
                pinned.append(name)
                free -= stages[name]
    disk_bytes = sum(p.stat().st_size for p in index.shards)
    return {
        "checkpoint_dir": str(index.root),
        "shards": len(index.shards),
        "disk_file_bytes": disk_bytes,
        "tensor_payload_bytes": sum(sizes.values()),
        "tensor_count": len(sizes),
        "max_recognized_stage_bytes": max((size for name, size in stages.items()
                                          if name != "unrecognized"), default=0),
        "stages_bytes": dict(sorted(stages.items())),
        "unrecognized_tensor_bytes": stages.get("unrecognized", 0),
        "cpu_cache_plan": {
            "budget_bytes": budget,
            "estimated_pinned_stages": pinned,
            "estimated_pinned_bytes": budget - free,
        },
        "warning": (
            "Header-only estimate; not a peak RAM/VRAM guarantee. Runtime stage "
            "compatibility and model output parity must be tested separately."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only memory/disk sizing of a local safetensors model")
    parser.add_argument("model_dir", help="local safetensors checkpoint directory")
    parser.add_argument("--cpu-cache-mb", type=int, default=0)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    result = inspect_checkpoint(args.model_dir, cpu_cache_mb=args.cpu_cache_mb)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
