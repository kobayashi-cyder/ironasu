"""Experimental exact-token greedy prefill KV reuse, bounded per-engine."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any


def cache_tensors(past: Any) -> list[Any]:
    """Only recognized tuple caches and DynamicCache layouts are admissible."""
    import torch
    def walk(obj: Any) -> list[Any]:
        if torch.is_tensor(obj):
            return [obj]
        if obj is None:
            return []
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                out.extend(walk(item))
            return out
        raise TypeError("unknown cache payload")
    if isinstance(past, (list, tuple)):
        tensors = walk(past)
    elif isinstance(getattr(past, "layers", None), (list, tuple)):
        tensors = []
        for layer in past.layers:
            key, val = getattr(layer, "keys", None), getattr(layer, "values", None)
            if not (torch.is_tensor(key) and torch.is_tensor(val)):
                raise TypeError("unsupported cache layer")
            tensors.extend((key, val))
    elif hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        tensors = walk(past.key_cache) + walk(past.value_cache)
    else:
        raise TypeError("unsupported cache object")
    if not tensors:
        raise TypeError("empty cache")
    return tensors


class PrefixKVCache:
    """Exact token IDs/mask key, independent snapshot clones, payload-byte LRU."""

    def __init__(self, *, max_bytes: int, max_entries: int = 8):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be >=0")
        if type(max_entries) is not int or not 1 <= max_entries <= 128:
            raise ValueError("max_entries must be 1..128")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries = OrderedDict()
        self._occupied = 0
        self._generation = 0
        self._hits = self._misses = self._rejected = self._insertions = 0
        self._saved_prefill_tokens = 0
        self._lock = RLock()

    def reset(self) -> None:
        """O(1) logical generation switch; deallocation timing is unspecified."""
        with self._lock:
            self._entries = OrderedDict()
            self._occupied = 0
            self._generation += 1

    def get(self, tokens: tuple[int, ...], mask: tuple[int, ...]) -> tuple[Any, Any] | None:
        with self._lock:
            key = (tokens, mask)
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            past, logits, size = entry
            try:
                result = deepcopy(past), logits.detach().clone()
            except (TypeError, ValueError, RuntimeError):
                self._entries.pop(key)
                self._occupied -= size
                self._misses += 1
                self._rejected += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            self._saved_prefill_tokens += len(tokens)
            return result

    def put(self, tokens: tuple[int, ...], mask: tuple[int, ...],
            past: Any, logits: Any) -> bool:
        if not self.max_bytes:
            return False
        if len(tokens) == 0 or len(tokens) != len(mask):
            raise ValueError("nonempty tokens/mask of equal length required")
        with self._lock:
            try:
                import torch
                if not torch.is_tensor(logits):
                    raise TypeError("logits must be tensor")
                all_tensors = cache_tensors(past) + [logits]
                # Deduplicate shared tensor objects when computing payload size.
                size = sum(t.numel() * t.element_size()
                           for t in {id(t): t for t in all_tensors}.values())
                if size > self.max_bytes:
                    self._rejected += 1
                    return False
                entry = (deepcopy(past), logits.detach().clone(), size)
            except (TypeError, ValueError, RuntimeError):
                self._rejected += 1
                return False
            key = (tokens, mask)
            old = self._entries.pop(key, None)
            if old:
                self._occupied -= old[2]
            while self._entries and (self._occupied + size > self.max_bytes
                                     or len(self._entries) >= self.max_entries):
                _, old = self._entries.popitem(last=False)
                self._occupied -= old[2]
            self._entries[key] = entry
            self._occupied += size
            self._insertions += 1
            return True

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "generation": self._generation,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "occupied_bytes": self._occupied,
                "entries": len(self._entries),
                "hits": self._hits, "misses": self._misses,
                "rejected": self._rejected, "insertions": self._insertions,
                "saved_prefill_tokens": self._saved_prefill_tokens,
            }


def greedy_decode_with_exact_prefix(model: Any, inputs: Any, *,
                                    max_new_tokens: int, kv_cache: PrefixKVCache,
                                    eos_token_id: int | list[int] | None = None) -> Any:
    """Single batch, unpadded greedy decoding; not full HF GenerationConfig."""
    import torch
    if not 1 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be 1..1024")
    if set(inputs) - {"input_ids", "attention_mask"}:
        raise ValueError("unexpected tokenizer inputs")
    ids, mask = inputs["input_ids"], inputs.get("attention_mask")
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.size(0) != 1 or ids.size(1) < 1:
        raise ValueError("only nonempty batch=1 input_ids supported")
    if mask is None:
        mask = torch.ones_like(ids)
    if not torch.is_tensor(mask) or mask.shape != ids.shape or not bool((mask == 1).all()):
        raise ValueError("requires an unpadded attention mask")
    if type(eos_token_id) is int:
        eos = {eos_token_id}
    elif eos_token_id is None:
        eos = set()
    elif isinstance(eos_token_id, (list, tuple)) and all(type(v) is int for v in eos_token_id):
        eos = set(eos_token_id)
    else:
        raise ValueError("invalid eos_token_id")
    key = tuple(int(x) for x in ids[0].tolist())
    key_mask = tuple(int(x) for x in mask[0].tolist())
    hit = kv_cache.get(key, key_mask)
    if hit is None:
        output = model(input_ids=ids, attention_mask=mask,
                       use_cache=True, return_dict=True)
        past, logits = output.past_key_values, output.logits[:, -1, :]
        kv_cache.put(key, key_mask, past, logits)
    else:
        past, logits = hit
    generated = []
    for step in range(max_new_tokens):
        token = logits.argmax(dim=-1, keepdim=True)
        generated.append(token)
        if int(token.item()) in eos or step == max_new_tokens - 1:
            break
        mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        output = model(input_ids=token, attention_mask=mask,
                       past_key_values=past, use_cache=True, return_dict=True)
        past, logits = output.past_key_values, output.logits[:, -1, :]
    return torch.cat(generated, dim=1)
