"""Sparse MoE expert weight support for out-of-core inference.

Only experts represented as individually called nn.ModuleList modules are supported.
Packed/vectorized MoE implementations (including Kimi K3 MXFP4) need a separate
execution adapter. A dict of expert tensors avoids re-reading hot experts, but
never changes router decisions or dense model FLOPs.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import re
import threading
from typing import Any


class ExpertLRU:
    """Byte-bounded LRU of CPU weight dictionaries keyed by expert module path.

    The cache stores *original tensors*, never device copies. The working set for
    the current expert and transient dequantization memory is additional.
    """

    def __init__(self, budget_bytes: int = 0):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("budget_bytes must be a nonnegative integer")
        self.budget_bytes = budget_bytes
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, tuple[dict[str, Any], int]] = OrderedDict()
        self.occupied_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0

    @staticmethod
    def _size(tensors: dict[str, Any]) -> int:
        from .q4 import _bytes
        return sum(_bytes(t) for t in tensors.values())

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key][0]

    def put(self, key: str, tensors: dict[str, Any]) -> None:
        size = self._size(tensors)
        with self._lock:
            if key in self._entries:
                return
            if size > self.budget_bytes or not self.budget_bytes:
                self.bypasses += 1
                return
            while self._entries and self.occupied_bytes + size > self.budget_bytes:
                _, (_, old_size) = self._entries.popitem(last=False)
                self.occupied_bytes -= old_size
                self.evictions += 1
            self._entries[key] = (tensors, size)
            self.occupied_bytes += size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.occupied_bytes = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_bytes": self.budget_bytes,
                "occupied_bytes": self.occupied_bytes,
                "cached_experts": list(self._entries.keys()),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "oversize_bypasses": self.bypasses,
            }


def discover_experts(model: Any, decoder_stages: list[str]) -> dict[str, Any]:
    """Find expert modules that must be executed via module.__call__ hooks.

    Deliberately fail closed for unknown expert implementations; a packed tensor
    or direct F.linear bypasses hooks and is not safe for this pager.
    """
    import torch
    result: dict[str, Any] = {}
    for name in decoder_stages:
        if not re.search(r"\.layers\.\d+$|\.h\.\d+$", name):
            continue
        block = model.get_submodule(name)
        for local, child in block.named_modules():
            if not local or not local.split(".")[-1] == "experts":
                continue
            if not isinstance(child, torch.nn.ModuleList):
                raise ValueError(f"Expert container is not a ModuleList: {name}.{local}")
            if not len(child):
                raise ValueError(f"Empty expert container: {name}.{local}")
            for i, expert in enumerate(child):
                if not any(True for _ in expert.parameters(recurse=True)):
                    raise ValueError(f"Expert has no parameters: {name}.{local}.{i}")
                result[f"{name}.{local}.{i}"] = expert
    return result


# Only checkpoint-name planning. No model, weights, custom code or GPU required.
_EXPERT_KEY = re.compile(r"^(?P<container>.+?\.layers\.\d+\..*?\.experts)\.(?P<expert>\d+)\.(?P<tensor>.+)$")


def inspect_moe_weight_map(weight_map: dict[str, str], *, layer: str | None = None,
                           selected: list[int] | None = None) -> dict[str, Any]:
    """Plan how many indexed tensor keys/shards a selected expert set touches.

    Works with ordinary safetensors and MXFP4 weight_packed/scale key names.
    No physical read byte estimate is claimed without inspecting headers.
    """
    if not isinstance(weight_map, dict):
        raise ValueError("weight_map must be a dict")
    experts: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str) or not shard:
            raise ValueError("Invalid weight_map entry")
        match = _EXPERT_KEY.fullmatch(key)
        if match:
            experts[match.group("container")][int(match.group("expert"))].append(key)
    containers = sorted(experts)
    result: dict[str, Any] = {"expert_containers": len(containers),
                              "expert_sets": {c: len(experts[c]) for c in containers},
                              "assumption": "Index metadata only; no tensor bytes or runnable architecture implied"}
    if selected is not None:
        if layer is None:
            raise ValueError("layer must be specified with selected experts")
        if not selected or any(type(n) is not int or n < 0 for n in selected):
            raise ValueError("selected must contain nonnegative integer expert IDs")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate selected expert ID")
        if layer not in experts:
            raise ValueError(f"Expert container not found: {layer}")
        missing = set(selected) - set(experts[layer])
        if missing:
            raise ValueError(f"Missing expert weights: {sorted(missing)}")
        all_keys = [k for names in experts[layer].values() for k in names]
        active_keys = [k for n in selected for k in experts[layer][n]]
        result["selection"] = {
            "layer": layer,
            "selected_experts": selected,
            "total_experts": len(experts[layer]),
            "total_tensor_keys": len(all_keys),
            "selected_tensor_keys": len(active_keys),
            "total_indexed_shards": len({weight_map[k] for k in all_keys}),
            "selected_indexed_shards": len({weight_map[k] for k in active_keys}),
            "selected_weight_names": active_keys,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE expert checkpoint index planner (metadata only)")
    parser.add_argument("checkpoint", help="Local model.safetensors.index.json, or its parent directory")
    parser.add_argument("--layer", help="Exact expert container, e.g. model.layers.1.mlp.experts")
    parser.add_argument("--expert", type=int, action="append", default=None,
                        help="Expert index to simulate selecting (repeatable; requires --layer)")
    parser.add_argument("--include-names", action="store_true", help="Show selected full weight keys")
    args = parser.parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors.index.json"
    with path.open(encoding="utf-8") as f:
        mapping = json.load(f)["weight_map"]
    result = inspect_moe_weight_map(mapping, layer=args.layer, selected=args.expert)
    if "selection" in result and not args.include_names:
        del result["selection"]["selected_weight_names"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
