"""Reference OCP MXFP4 E2M1 / E8M0 unpacking for local safetensors.

This decodes exact representable FP4 values, not the original pre-quantized
weights. It is NOT a fused GEMM kernel and materializes the full matrix.

Kimi-K3 routed experts store `<stem>.weight_packed` (U8 nibbles) plus
`<stem>.weight_scale` (U8 E8M0 per 32 values). This adapter exposes a logical
`<stem>.weight` to the normal ironasu layer/expert pager, without rewriting the
checkpoint. Kimi's full custom architecture still requires separate support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MXFP4 support requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class MXFP4Tensor:
    packed: Any
    scales: Any
    shape: tuple[int, ...]
    dtype: Any

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() * self.packed.element_size() + self.scales.numel() * self.scales.element_size()

    def to(self, *, device: str | Any):
        """Unpack last dimension (2 values/byte), scaling blocks of 32.

        e8m0=255 yields NaN per OCP spec. This intentionally uses float32
        intermediates to preserve the exact encoded values before casting.
        """
        torch = _torch()
        p = self.packed.to(device=device)
        s = self.scales.to(device=device)
        if p.dtype != torch.uint8 or s.dtype != torch.uint8:
            raise ValueError("MXFP4 payload and scales must both be uint8")
        if p.ndim < 2 or p.shape[:-1] != s.shape[:-1] or p.shape[-1] * 2 != s.shape[-1] * 32:
            raise ValueError("Invalid MXFP4 tensor layout")
        if tuple(self.shape) != (*p.shape[:-1], p.shape[-1] * 2):
            raise ValueError("MXFP4 logical shape mismatch")
        # E2M1: sign bit 3, magnitude codes 0..7 map to these values.
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                             dtype=torch.float32, device=p.device)
        lo = torch.bitwise_and(p, 15).long()
        hi = torch.bitwise_right_shift(p, 4).long()
        code = torch.stack([lo, hi], dim=-1).reshape(*p.shape[:-1], -1)
        values = table[code].reshape(*p.shape[:-1], s.shape[-1], 32)
        exponent = s.float() - 127.0
        factors = torch.exp2(exponent)
        # E8M0 255 is a NaN sentinel, not 2**128.
        factors = torch.where(s == 255, torch.full_like(factors, float("nan")), factors)
        return (values * factors.unsqueeze(-1)).reshape(self.shape).to(dtype=self.dtype)


class MXFP4Index:
    """Present local native MXFP4 pairs as logical floating-point weights.

    No remote code or checkpoint conversion. A missing partner is rejected, and
    nonpacked tensors are delegated to the safe local safetensors reader.
    """

    def __init__(self, source: Any, *, dtype: Any = None):
        torch = _torch()
        self.source = source
        self.dtype = torch.bfloat16 if dtype is None else dtype
        keys = source.keys()
        self._pairs: dict[str, tuple[str, str]] = {}
        self._raw: set[str] = set(keys)
        for k in keys:
            if k.endswith(".weight_packed"):
                scale = k[:-len("weight_packed")] + "weight_scale"
                if scale not in keys:
                    raise ValueError(f"MXFP4 scale tensor missing for {k}")
                logical = k[:-len("weight_packed")] + "weight"
                if logical in keys:
                    raise ValueError(f"Both packed and full weights present: {logical}")
                self._pairs[logical] = (k, scale)
                self._raw.remove(k)
                self._raw.remove(scale)
        orphans = {k for k in self._raw if k.endswith(".weight_scale")}
        if orphans:
            raise ValueError(f"MXFP4 packed tensor missing for {sorted(orphans)[0]}")

    def keys(self) -> set[str]:
        return self._raw | set(self._pairs)

    def _underlying(self, names: list[str]) -> list[str]:
        raw: list[str] = []
        for name in dict.fromkeys(names):
            if name in self._pairs:
                raw.extend(self._pairs[name])
            elif name in self._raw:
                raw.append(name)
            else:
                raise KeyError(f"Checkpoint tensor missing: {name}")
        return list(dict.fromkeys(raw))

    def tensor_byte_sizes(self, names: list[str]) -> dict[str, int]:
        sizes = self.source.tensor_byte_sizes(self._underlying(names))
        return {name: sum(sizes[k] for k in self._pairs[name]) if name in self._pairs else sizes[name]
                for name in dict.fromkeys(names)}

    def read(self, names: list[str]) -> dict[str, Any]:
        torch = _torch()
        payload = self.source.read(self._underlying(names))
        result: dict[str, Any] = {}
        for name in dict.fromkeys(names):
            if name in self._pairs:
                k, sk = self._pairs[name]
                packed, scales = payload[k], payload[sk]
                if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
                    raise ValueError(f"Expected U8 MXFP4 tensor pair: {name}")
                if packed.ndim < 2 or scales.shape[:-1] != packed.shape[:-1] or packed.shape[-1] * 2 != scales.shape[-1] * 32:
                    raise ValueError(f"Bad MXFP4 tensor dimensions: {name}")
                result[name] = MXFP4Tensor(packed, scales,
                                          (*packed.shape[:-1], packed.shape[-1] * 2), self.dtype)
            else:
                result[name] = payload[name]
        return result
