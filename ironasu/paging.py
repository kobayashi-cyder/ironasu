"""Out-of-core safetensors inference: materialize one module, then evict it.

This is an independent, deliberately restricted reference implementation of
AirLLM-style layer streaming. Unlike GPU model-parallelism, it does NOT reduce
FLOPs: every decoder block is executed for every generated token.

Only local safetensors checkpoints, ordinary dense PyTorch parameters and a
known set of decoder paths are supported. No pickle checkpoints or custom code.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Iterator
import json


class PagingError(RuntimeError):
    """Checkpoint cannot safely be streamed by this implementation."""


def _deps():
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise PagingError("Install ironasu[stream] (requires torch and safetensors)") from exc
    return torch, safe_open


class SafetensorsIndex:
    """Read original HF shard files directly, without duplicating the checkpoint.

    Only keys are indexed in memory; tensors are fetched on demand. A malicious
    weight_map path cannot escape the selected local directory.
    """

    def __init__(self, root: str | Path):
        _, safe_open = _deps()
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise PagingError("checkpoint must be a local directory")
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            with index.open("r", encoding="utf-8") as inp:
                mapping = json.load(inp).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise PagingError("invalid safetensors index: missing weight_map")
        elif (self.root / "model.safetensors").is_file():
            with safe_open(str(self.root / "model.safetensors"), framework="pt", device="cpu") as source:
                mapping = {key: "model.safetensors" for key in source.keys()}
        else:
            raise PagingError("Only local model.safetensors or model.safetensors.index.json supported")
        checked: dict[str, Path] = {}
        for key, file_name in mapping.items():
            if not isinstance(key, str) or not key or not isinstance(file_name, str):
                raise PagingError("invalid safetensors index entry")
            target = (self.root / file_name).resolve(strict=True)
            if not target.is_file() or target.suffix != ".safetensors" or not target.is_relative_to(self.root):
                raise PagingError("checkpoint shard is missing, unsafe, or is not safetensors")
            checked[key] = target
        self.paths = checked
        self.shards = sorted(set(checked.values()))

    def keys(self) -> set[str]:
        return set(self.paths)

    def read(self, names: list[str]) -> dict[str, Any]:
        _, safe_open = _deps()
        # Reading per-file avoids repeatedly opening the same large shard for a module.
        grouped: dict[Path, list[str]] = defaultdict(list)
        for name in names:
            if name not in self.paths:
                raise PagingError(f"checkpoint tensor missing: {name}")
            grouped[self.paths[name]].append(name)
        tensors = {}
        for path, keys in grouped.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for key in keys:
                    tensors[key] = source.get_tensor(key)
        return tensors


@dataclass(frozen=True)
class _Slot:
    owner: Any
    local: str
    tensor_name: str
    checkpoint_name: str
    shape: tuple[int, ...]
    dtype: Any
    buffer: bool


@dataclass
class _Stage:
    name: str
    module: Any
    slots: list[_Slot]


def _lookup(root: Any, path: str) -> Any:
    try:
        return root.get_submodule(path)
    except (AttributeError, ValueError) as exc:
        raise PagingError(f"Unsupported module path: {path}") from exc


def decoder_stage_names(model: Any) -> list[str]:
    """Conservative model-family detection. Reject unknown rather than miscompute."""
    prefixes = (
        ("model.layers", ("model.embed_tokens",), ("model.norm", "lm_head")),
        ("transformer.h", ("transformer.wte", "transformer.wpe"), ("transformer.ln_f", "lm_head")),
        ("gpt_neox.layers", ("gpt_neox.embed_in",), ("gpt_neox.final_layer_norm", "embed_out")),
    )
    for layers, leading, trailing in prefixes:
        try:
            sequence = _lookup(model, layers)
        except PagingError:
            continue
        if not hasattr(sequence, "__len__") or len(sequence) == 0:
            continue
        return [*leading, *(f"{layers}.{i}" for i in range(len(sequence))), *trailing]
    raise PagingError("Unsupported architecture: expected model.layers / transformer.h / gpt_neox.layers")


class LayerPager:
    """PyTorch hooks that load each stage just-in-time and free it after forward.

    A session owns exclusive access to the model; direct model.forward calls
    outside session must not race or be used after close(). The model must be
    constructed with meta *parameters* (e.g. accelerate.init_empty_weights).
    """

    def __init__(self, model: Any, index: SafetensorsIndex, *, device: str = "cpu", sync_timing: bool = False, prefetch: bool = False):
        torch, _ = _deps()
        self.model = model
        self.index = index
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise PagingError("Only cpu and cuda inference devices supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PagingError("CUDA device requested but CUDA is not available")
        self.sync_timing = sync_timing
        self.prefetch = prefetch
        self._executor: ThreadPoolExecutor | None = None
        self._ahead: tuple[str, Future] | None = None
        self._run_started: dict[str, float] = {}
        self._lock = RLock()
        self._hooks: list[Any] = []
        self._busy = False
        self._closed = False
        self._traces: dict[str, dict[str, float | int]] = defaultdict(lambda: {
            "disk_seconds": 0.0, "transfer_seconds": 0.0, "compute_seconds": 0.0,
            "loads": 0, "peak_simultaneous_stages": 0, "prefetch_wait_seconds": 0.0,
        })
        self._resident: set[str] = set()
        self._stages: list[_Stage] = []
        stage_names = decoder_stage_names(model)
        # Detect parameter aliases (notably tied embedding and lm_head weights).
        aliases: dict[int, list[str]] = defaultdict(list)
        all_parameters: dict[str, Any] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases[id(parameter)].append(name)
            all_parameters[name] = parameter
            if parameter.device.type != "meta":
                raise PagingError("Parameters must start on meta: create model with init_empty_weights")
        names = index.keys()
        seen: set[str] = set()
        for path in stage_names:
            module = _lookup(model, path)
            slots: list[_Slot] = []
            for local, parameter in module.named_parameters(remove_duplicate=False):
                qualified = f"{path}.{local}"
                source_key = qualified if qualified in names else next(
                    (candidate for candidate in aliases[id(parameter)] if candidate in names), None
                )
                if source_key is None:
                    raise PagingError(f"No checkpoint tensor (or tied alias) for {qualified}")
                parent, _, attr = local.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                slots.append(_Slot(owner, attr, qualified, source_key, tuple(parameter.shape), parameter.dtype, False))
                seen.add(qualified)
            for local, tensor in module.named_buffers(recurse=True):
                qualified = f"{path}.{local}"
                if qualified not in names:
                    # Nonpersistent attention/rotary buffers are not part of weight streaming.
                    continue
                parent, _, attr = local.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                slots.append(_Slot(owner, attr, qualified, qualified, tuple(tensor.shape), tensor.dtype, True))
            self._stages.append(_Stage(path, module, slots))
        unexpected = sorted(set(all_parameters) - seen)
        if unexpected:
            raise PagingError(f"Unsupported non-streamed parameters: {unexpected[:5]}")
        if not self._stages or not any(s.slots for s in self._stages):
            raise PagingError("No streamable parameters found")
        # Model-level nonpersistent buffers (rotary cache, attention masks) can be
        # needed outside streamed modules. Keep these small buffers resident.
        # GPU inference must not mix a CUDA activation with a CPU attention mask.
        if self.device.type == "cuda":
            for parent in model.modules():
                for name, value in parent._buffers.items():
                    if value is not None and value.device.type != "meta":
                        parent._buffers[name] = value.to(self.device)
        self._next = {a.name: b for a, b in zip(self._stages, self._stages[1:])}
        # Start the worker only after successful validation, to avoid leaking
        # worker threads when a checkpoint is rejected.
        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        # Register hooks only after all preflight validation passed.
        for stage in self._stages:
            self._hooks.append(stage.module.register_forward_pre_hook(self._make_pre(stage)))
            self._hooks.append(stage.module.register_forward_hook(self._make_post(stage), always_call=True))

    def _sync(self) -> None:
        torch, _ = _deps()
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _load_stage_cpu(self, stage: _Stage):
        started = perf_counter()
        data = self.index.read(list(dict.fromkeys(x.checkpoint_name for x in stage.slots)))
        return data, perf_counter() - started

    def _make_pre(self, stage: _Stage):
        def hook(module: Any, arguments: Any) -> None:
            if self._closed or not self._busy:
                raise PagingError("Use pager.session() for inference; pager is not active")
            self._sync()
            wait_started = perf_counter()
            if self._ahead is not None and self._ahead[0] != stage.name:
                # Some architectures conditionally skip modules; never retain an
                # unbounded stale prefetched stage in that case.
                _, stale = self._ahead
                self._ahead = None
                stale.cancel()
                if not stale.cancelled():
                    stale.result()
            if self._ahead is not None and self._ahead[0] == stage.name:
                _, future = self._ahead
                self._ahead = None
                tensors, disk_seconds = future.result()
                self._traces[stage.name]["prefetch_wait_seconds"] += perf_counter() - wait_started
            else:
                tensors, disk_seconds = self._load_stage_cpu(stage)
            self._traces[stage.name]["disk_seconds"] += disk_seconds
            # At most one future holds the *next* stage's weights on CPU.
            if self._executor is not None and stage.name in self._next:
                following = self._next[stage.name]
                self._ahead = (following.name, self._executor.submit(self._load_stage_cpu, following))
            start = perf_counter()
            try:
                for slot in stage.slots:
                    tensor = tensors[slot.checkpoint_name]
                    if tuple(tensor.shape) != slot.shape:
                        raise PagingError(f"Tensor shape mismatch for {slot.tensor_name}")
                    # Use the checkpoint dtype, not the meta placeholder dtype.
                    loaded = tensor.to(device=self.device)
                    if slot.buffer:
                        slot.owner._buffers[slot.local] = loaded
                    else:
                        torch, _ = _deps()
                        slot.owner._parameters[slot.local] = torch.nn.Parameter(loaded, requires_grad=False)
                self._sync()
                self._resident.add(stage.name)
                self._traces[stage.name]["loads"] += 1
                self._traces[stage.name]["peak_simultaneous_stages"] = max(
                    self._traces[stage.name]["peak_simultaneous_stages"], len(self._resident)
                )
                self._traces[stage.name]["transfer_seconds"] += perf_counter() - start
                self._run_started[stage.name] = perf_counter()
            except BaseException:
                self._evict(stage)
                raise
        return hook

    def _evict(self, stage: _Stage) -> None:
        torch, _ = _deps()
        for slot in stage.slots:
            meta = torch.empty(slot.shape, dtype=slot.dtype, device="meta")
            if slot.buffer:
                slot.owner._buffers[slot.local] = meta
            else:
                slot.owner._parameters[slot.local] = torch.nn.Parameter(meta, requires_grad=False)
        self._resident.discard(stage.name)

    def _make_post(self, stage: _Stage):
        def hook(module: Any, arguments: Any, output: Any) -> None:
            try:
                if stage.name in self._resident:
                    self._sync()
                    self._traces[stage.name]["compute_seconds"] += perf_counter() - self._run_started.pop(stage.name, perf_counter())
            finally:
                self._evict(stage)
        return hook

    @contextmanager
    def session(self) -> Iterator[Any]:
        with self._lock:
            if self._closed:
                raise PagingError("pager already closed")
            if self._busy:
                raise PagingError("nested pager sessions are not supported")
            self._busy = True
            try:
                yield self.model
            finally:
                for stage in self._stages:
                    self._evict(stage)
                self._run_started.clear()
                if self._ahead is not None:
                    _, future = self._ahead
                    future.cancel()
                    if not future.cancelled():
                        # A worker can already be reading. Wait before discarding its buffer.
                        future.result()
                    self._ahead = None
                self._busy = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "stage_count": len(self._stages),
            "prefetch": self.prefetch,
            "resident_stage_count": len(self._resident),
            "per_stage": {stage: dict(data) for stage, data in self._traces.items()},
            "warning": "Load times include page cache; CUDA computation times are enqueue-only unless sync_timing=True. Model FLOPs are unchanged.",
        }

    def close(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            for stage in self._stages:
                self._evict(stage)
            self._closed = True

    def __enter__(self) -> "LayerPager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
