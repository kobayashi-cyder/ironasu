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
from threading import Lock, RLock
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

    def tensor_byte_sizes(self, names: list[str]) -> dict[str, int]:
        """Estimate exact checkpoint payload sizes without loading full tensors.

        Inspects safetensors headers using ``get_slice``, not ``get_tensor``.
        These byte counts exclude headers, allocator overhead and page cache.
        """
        _, safe_open = _deps()
        sizes = {
            "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
            "I64": 8, "U64": 8, "I32": 4, "U32": 4,
            "I16": 2, "U16": 2, "I8": 1, "U8": 1,
            "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        }
        grouped: dict[Path, list[str]] = defaultdict(list)
        for name in dict.fromkeys(names):
            if name not in self.paths:
                raise PagingError(f"checkpoint tensor missing: {name}")
            grouped[self.paths[name]].append(name)
        result: dict[str, int] = {}
        for path, keys in grouped.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for key in keys:
                    view = source.get_slice(key)
                    dtype = view.get_dtype()
                    if dtype not in sizes:
                        raise PagingError(f"Unsupported checkpoint dtype for byte estimate: {dtype}")
                    count = 1
                    for dimension in view.get_shape():
                        count *= dimension
                    result[key] = count * sizes[dtype]
        return result

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

    def __init__(self, model: Any, index: SafetensorsIndex, *, device: str = "cpu",
                 sync_timing: bool = False, prefetch: bool = False,
                 cpu_cache_mb: int = 0, pin_stages: list[str] | None = None):
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
        if isinstance(cpu_cache_mb, bool) or not isinstance(cpu_cache_mb, int) or cpu_cache_mb < 0:
            raise PagingError("cpu_cache_mb must be a nonnegative integer")
        if pin_stages is not None and cpu_cache_mb == 0:
            raise PagingError("pin_stages requires cpu_cache_mb > 0")
        self._cpu_cache_budget = cpu_cache_mb * (1024 ** 2)
        self._cpu_cache_bytes = 0
        self._cpu_cache_lock = Lock()
        self._cpu_cache: dict[str, dict[str, Any]] = {}
        self._cpu_cache_hit_count = 0
        self._cpu_cache_miss_count = 0
        self._disk_read_bytes = 0
        self._avoided_read_bytes = 0
        self._pin_stage_names: set[str] = set()
        self._pin_stage_budget: dict[str, int] = {}
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
            "cache_hits": 0, "cache_misses": 0, "disk_read_bytes": 0,
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
                # `save_pretrained(safe_serialization=True)` drops duplicate
                # tied lm_head weights. On meta, accelerate/Transformers may not
                # preserve Python Parameter identity even when config says tied.
                if (source_key is None and qualified == "lm_head.weight"
                        and getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
                    for candidate in ("model.embed_tokens.weight", "transformer.wte.weight",
                                      "gpt_neox.embed_in.weight"):
                        if candidate in names:
                            source_key = candidate
                            break
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
        if self._cpu_cache_budget:
            # Static admission is deliberate. An LRU over a model much larger
            # than RAM may evict *every* layer during each decode token and
            # achieve zero hits. Pin a stable subset across token iterations.
            stage_by_name = {stage.name: stage for stage in self._stages}
            if pin_stages is not None:
                if len(set(pin_stages)) != len(pin_stages):
                    raise PagingError("duplicate pin_stages entry")
                missing = set(pin_stages) - stage_by_name.keys()
                if missing:
                    raise PagingError(f"Unknown pin_stages: {sorted(missing)}")
                candidates = [stage_by_name[name] for name in pin_stages]
            else:
                # Prioritize output head and embedding, then sequential blocks.
                # Stage sizes are derived from tensor headers; tied weights in
                # different stages are budgeted independently, conservatively.
                priority = {"lm_head": 0, "embed_out": 0,
                            "model.embed_tokens": 1, "transformer.wte": 1,
                            "gpt_neox.embed_in": 1}
                candidates = sorted(self._stages,
                                    key=lambda item: priority.get(item.name, 2))
            # A single header scan per shard avoids reopening the same shard
            # for every decoder block during the cache admission pass.
            checkpoint_sizes = index.tensor_byte_sizes([
                slot.checkpoint_name for stage in candidates for slot in stage.slots
            ])
            remaining = self._cpu_cache_budget
            for stage in candidates:
                if not stage.slots:
                    continue
                keys = list(dict.fromkeys(slot.checkpoint_name for slot in stage.slots))
                estimated = sum(checkpoint_sizes[key] for key in keys)
                if estimated <= remaining:
                    self._pin_stage_names.add(stage.name)
                    self._pin_stage_budget[stage.name] = estimated
                    remaining -= estimated
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
        # The worker thread may be reading the next stage while the main thread
        # performs model.forward. Protect statistics and dictionary mutation.
        keys = list(dict.fromkeys(x.checkpoint_name for x in stage.slots))
        with self._cpu_cache_lock:
            cached = self._cpu_cache.get(stage.name)
            if cached is not None:
                self._cpu_cache_hit_count += 1
                self._traces[stage.name]["cache_hits"] += 1
                self._avoided_read_bytes += self._pin_stage_budget[stage.name]
                return cached, 0.0
            self._cpu_cache_miss_count += 1
            self._traces[stage.name]["cache_misses"] += 1
        started = perf_counter()
        data = self.index.read(keys)
        elapsed = perf_counter() - started
        size_bytes = sum(t.numel() * t.element_size() for t in data.values())
        # Conservative upper bound of what this stage has read from safetensors.
        with self._cpu_cache_lock:
            self._disk_read_bytes += size_bytes
            self._traces[stage.name]["disk_read_bytes"] += size_bytes
            if (stage.name in self._pin_stage_names and
                    self._cpu_cache_bytes + size_bytes <= self._cpu_cache_budget):
                # Do NOT cache a corrupt/incompatible checkpoint tensor.
                expected = {slot.checkpoint_name: slot.shape for slot in stage.slots}
                if all(tuple(data[key].shape) == shape for key, shape in expected.items()):
                    self._cpu_cache[stage.name] = data
                    self._cpu_cache_bytes += size_bytes
        return data, elapsed

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
            "cpu_cache": {
                "policy": "fixed_admission",
                "budget_bytes": self._cpu_cache_budget,
                "occupied_bytes": self._cpu_cache_bytes,
                "planned_stages": sorted(self._pin_stage_names),
                "cached_stages": sorted(self._cpu_cache),
                "hits": self._cpu_cache_hit_count,
                "misses": self._cpu_cache_miss_count,
                "disk_read_bytes": self._disk_read_bytes,
                "avoided_read_bytes": self._avoided_read_bytes,
            },
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
            self._cpu_cache.clear()
            self._cpu_cache_bytes = 0
            self._closed = True

    def __enter__(self) -> "LayerPager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
