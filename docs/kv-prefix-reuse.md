# Exact-token KV prefill reuse (experimental)

This module is opt-in for repeated identical tokenized prompts with greedy decoding. It extends the existing LayerPager engine without changing its weight paging or one-stage prefetch strategy.

Example:
    ironasu-stream LOCAL_MODEL_DIR --prompt "Write a short function" --kv-cache-mb 256 --kv-cache-entries 4 --repeat 2 --tokens 8

With the KV cache disabled (default budget 0), PagedCausalLM uses the original Transformers model.generate path. When enabled, it uses a restricted batch-1 greedy forward loop, not full GenerationConfig. Sampling, beams, custom logits processors, token_type_ids, arbitrary shared prefixes, cross-model reuse and multimodal inputs are not supported.

A cached entry contains copies of full-prompt KV tensors and final prefill logits. Full token-id and attention-mask tuples are compared for equality. Copies are used for decoding because DynamicCache objects may mutate in place. Unknown cache layouts are not stored. The LRU limits payload tensor bytes and entry count, but allocator and Python overhead are excluded. GPU cache entries reside on GPU, so configure a conservative budget.

reset_kv_cache() logically invalidates entries. Always reset before changing weights, adapters, tokenizer or position semantics. Cache is not persistent across program restarts. Stats can be read with kv_cache_snapshot(): hits, misses, saved_prefill_tokens, occupied_bytes and rejection count.

A successful hit skips the full prompt forward, not autoregressive continuation steps. Cache cloning and eviction have costs. It may be slower than no cache.

Validation: run python -m unittest discover -s tests -v; optional integration tests compare generated tokens for local tiny GPT-2 and Qwen2 checkpoints against unmodified Transformers greedy generation. Benchmark real TTFT, p50/p95 wall-time, peak RAM/VRAM, SSD reads and generated-code correctness; do not infer saved FLOPs or SSD reads from hit counts.

Model checkpoint loading is local-only. Do not enable this experimental feature for unverified custom architectures or nonstandard generation configurations.
