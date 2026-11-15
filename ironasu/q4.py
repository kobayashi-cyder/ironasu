"""Safe, local, opt-in Q4 weight streaming for ironasu.

Q4 is a *lossy* symmetric per-group format. It does not implement a fused
low-bit GEMM kernel; weights are materialized at each layer forward. This
reduces checkpoint bytes and CPU-to-device transfer, not necessarily latency.

Only floating-point tensors with a supported dtype are quantized. Small and
non-floating tensors stay lossless. No pickle or remote code is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import math
import os
import time

from .paging import PagingError, SafetensorsIndex, _deps

FORMAT = "ironasu-q4-v1"
INDEX_FILE = "ironasu-q4.index.json"
DTYPES = {"float16", "bfloat16", "float32"}


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _bytes(tensor: Any) -> int:
    """Resident payload size, excluding object and allocator overhead."""
    if isinstance(tensor, Q4Tensor):
        return tensor.storage_bytes
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class Q4Tensor:
    packed: Any
    scales: Any
    shape: tuple[int, ...]
    dtype: Any
    group_size: int

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() * self.packed.element_size() + self.scales.numel() * self.scales.element_size()

    def to(self, *, device: Any):
        """Dequantize on target CPU/CUDA. Peak allocation can exceed weight size."""
        torch, _ = _deps()
        packed = self.packed.to(device=device, non_blocking=False)
        scales = self.scales.to(device=device, non_blocking=False)
        # Sign extend individual nibbles from two's complement.
        lo = (packed & 0x0F).to(torch.int16)
        hi = ((packed >> 4) & 0x0F).to(torch.int16)
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        values = torch.stack((lo, hi), dim=1).reshape(-1).float()
        values = values[:math.prod(self.shape)]
        scale_per_element = torch.repeat_interleave(scales.float(), self.group_size)[:values.numel()]
        result = (values * scale_per_element).reshape(self.shape)
        return result.to(dtype=self.dtype)


def pack_q4(tensor: Any, group_size: int = 64) -> Q4Tensor:
    torch, _ = _deps()
    if group_size not in (32, 64, 128, 256):
        raise ValueError("group_size must be 32, 64, 128, or 256")
    if _dtype_name(tensor) not in DTYPES:
        raise ValueError("only float16/bfloat16/float32 may be quantized")
    # Float intermediates are temporary; this operation is one tensor at a time.
    values = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not torch.isfinite(values).all().item():
        raise ValueError("non-finite tensor cannot be safely quantized")
    count = values.numel()
    if not count:
        raise ValueError("cannot quantize an empty tensor")
    pad = (-count) % group_size
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
    groups = values.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1).div(7.0)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    digits = torch.round(groups / scales[:, None]).clamp(-7, 7).to(torch.int16).flatten()
    packed = ((digits[0::2] & 15) | ((digits[1::2] & 15) << 4)).to(torch.uint8).contiguous()
    return Q4Tensor(packed, scales.contiguous(), tuple(tensor.shape), tensor.dtype, group_size)


class Q4Index:
    """Read per-tensor Q4 safetensors using a validated manifest."""
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)
        index_file = self.root / INDEX_FILE
        if not index_file.is_file():
            raise PagingError(f"Missing {INDEX_FILE}")
        data = json.loads(index_file.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT or not isinstance(data.get("tensors"), dict) or not data["tensors"]:
            raise PagingError("Invalid or unsupported Q4 checkpoint manifest")
        self.entries: dict[str, dict[str, Any]] = {}
        self.paths: dict[str, Path] = {}
        for name, item in data["tensors"].items():
            if not isinstance(name, str) or not name or not isinstance(item, dict):
                raise PagingError("Invalid Q4 tensor manifest entry")
            rel = item.get("file")
            if not isinstance(rel, str) or not rel.startswith("weights/") or not rel.endswith(".safetensors"):
                raise PagingError("Unsafe Q4 tensor path")
            path = (self.root / rel).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(self.root):
                raise PagingError("Q4 tensor path escapes checkpoint directory")
            kind, shape, dtype = item.get("kind"), item.get("shape"), item.get("dtype")
            if kind not in ("q4", "raw") or not isinstance(shape, list) or not all(isinstance(n, int) and n >= 0 for n in shape):
                raise PagingError("Invalid Q4 shape or format")
            if not isinstance(dtype, str) or not dtype.startswith(("float", "bfloat", "int", "uint", "bool")):
                raise PagingError("Invalid Q4 dtype")
            if kind == "q4" and (dtype not in DTYPES or item.get("group_size") not in (32, 64, 128, 256)):
                raise PagingError("Unsupported Q4 dtype or group size")
            if not isinstance(item.get("stored_bytes"), int) or item["stored_bytes"] < 0:
                raise PagingError("Invalid Q4 byte count")
            self.entries[name] = item
            self.paths[name] = path
        self.shards = sorted(set(self.paths.values()))

    def keys(self) -> set[str]:
        return set(self.entries)

    def tensor_byte_sizes(self, names: list[str]) -> dict[str, int]:
        try:
            return {name: self.entries[name]["stored_bytes"] for name in dict.fromkeys(names)}
        except KeyError as exc:
            raise PagingError(f"Q4 checkpoint tensor missing: {exc}") from exc

    def read(self, names: list[str]) -> dict[str, Any]:
        torch, safe_open = _deps()
        result = {}
        for name in dict.fromkeys(names):
            if name not in self.entries:
                raise PagingError(f"Q4 checkpoint tensor missing: {name}")
            entry = self.entries[name]
            with safe_open(str(self.paths[name]), framework="pt", device="cpu") as source:
                if entry["kind"] == "raw":
                    value = source.get_tensor("raw")
                    if tuple(value.shape) != tuple(entry["shape"]) or _dtype_name(value) != entry["dtype"]:
                        raise PagingError(f"Q4 raw tensor shape/dtype mismatch: {name}")
                    result[name] = value
                else:
                    packed = source.get_tensor("packed")
                    scales = source.get_tensor("scales")
                    size = math.prod(entry["shape"])
                    group = entry["group_size"]
                    groups = (size + group - 1) // group
                    if packed.dtype != torch.uint8 or scales.dtype != torch.float32 or packed.numel() != groups * group // 2 or scales.numel() != groups:
                        raise PagingError(f"Corrupt Q4 tensor payload: {name}")
                    dtype = getattr(torch, entry["dtype"])
                    result[name] = Q4Tensor(packed, scales, tuple(entry["shape"]), dtype, group)
            if _bytes(result[name]) != entry["stored_bytes"]:
                raise PagingError(f"Q4 byte count mismatch: {name}")
        return result


def quantize_checkpoint(source: str | Path, output: str | Path, *, group_size: int = 64,
                        min_elements: int = 128) -> dict[str, Any]:
    """Convert a local safetensors checkpoint one tensor at a time, no overwrite."""
    torch, _ = _deps()
    from safetensors.torch import save_file
    if group_size not in (32, 64, 128, 256) or isinstance(min_elements, bool) or min_elements < 1:
        raise ValueError("Invalid group size or min_elements")
    original = SafetensorsIndex(source)
    target = Path(output).expanduser().absolute()
    if target.resolve() == original.root or original.root in target.resolve().parents:
        raise ValueError("Output must be outside the original checkpoint directory")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError("Output directory must be empty (refusing overwrite)")
    (target / "weights").mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict[str, Any]] = {}
    src_bytes = 0
    dst_bytes = 0
    start = time.perf_counter()
    for i, name in enumerate(sorted(original.keys())):
        value = original.read([name])[name]
        src_bytes += _bytes(value)
        payload: dict[str, Any]
        if _dtype_name(value) in DTYPES and value.numel() >= min_elements:
            q = pack_q4(value, group_size)
            payload = {"packed": q.packed, "scales": q.scales}
            kind = "q4"
        else:
            payload = {"raw": value.detach().cpu().contiguous()}
            kind = "raw"
        rel = f"weights/{i:08d}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}.safetensors"
        path = target / rel
        save_file(payload, str(path))
        nbytes = sum(_bytes(t) for t in payload.values())
        dst_bytes += nbytes
        meta = {"file": rel, "kind": kind, "shape": list(value.shape), "dtype": _dtype_name(value),
                "stored_bytes": nbytes}
        if kind == "q4":
            meta["group_size"] = group_size
        tensors[name] = meta
        del value, payload
    data = {"format": FORMAT, "group_size": group_size, "tensors": tensors,
            "source_payload_bytes": src_bytes, "quantized_payload_bytes": dst_bytes}
    partial = target / (INDEX_FILE + ".tmp")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, target / INDEX_FILE)
    return {"tensor_count": len(tensors), "original_payload_bytes": src_bytes,
            "stored_payload_bytes": dst_bytes, "payload_ratio": dst_bytes / src_bytes,
            "duration_s": time.perf_counter() - start, "output": str(target)}


def main():
    parser = argparse.ArgumentParser(description="Opt-in experimental Q4 weight packing (lossy)")
    parser.add_argument("model_dir", help="Original trusted local HF safetensors directory")
    parser.add_argument("--output", required=True, help="New, empty output directory outside model_dir")
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128, 256))
    parser.add_argument("--min-elements", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(quantize_checkpoint(args.model_dir, args.output, group_size=args.group_size,
                                        min_elements=args.min_elements), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
