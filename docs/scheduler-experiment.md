# Experimental speculation and prefetch scheduler

Status: **standalone Python policy components and tests**. There is NO
integrated AirLLM runtime, CUDA transfer code, target-model verifier, model
weight decoder, or speedup benchmark yet. Running this module alone cannot
accelerate a local LLM.

## Three separate mechanisms

1. **OrderedLayerPrefetcher:** a bounded two-stage pipeline. While `apply`
   executes layer L, a background worker may call `load` for L+1. Layer
   execution order is preserved. Estimated staged weight bytes are limited,
   but CUDA allocations, the KV cache, loader overhead, and decompression
   transients are NOT included in this estimate. For real offloading, `load`
   must copy and synchronize correctly with the compute stream.
2. **NgramDraftIndex:** a bounded per-session hash table that proposes
   continuations from previously approved token sequences. These are **draft
   guesses, never valid outputs** until verified by the target model.
3. **AdaptiveSpeculationScheduler:** compares real decode-only seconds per
   verified output token at several draft lengths with a measured target-only
   baseline. It explores occasionally, chooses a measured improvement only
   above a threshold, and disables speculation if the backend cannot verify
   or memory/queue pressure prevents it.

## Next integration boundary

A compatible backend must provide:
- exact, tokenizer-compatible, target-model speculative verification and
  corrected sampling; **never** accept guessed draft tokens unconditionally;
- target-only and speculative decode-cycle time, proposed length, verified
  accepted draft count, and total verified emitted tokens;
- layer ID, estimated bytes, and a `load(id)` function plus a synchronous
  `apply(id, weights, state)` function; handle transfer synchronization;
- KV-cache rollback for rejected prefixes and cancellation safety;
- separate per-session n-gram indices (do not leak token histories between users).

### Minimal offline smoke test

```bash
python -m unittest discover -s tests -v
```

### Minimal policy example

```python
from ironasu import AdaptiveSpeculationScheduler

policy = AdaptiveSpeculationScheduler(draft_lengths=(2, 4))
for _ in range(3):
    policy.record_baseline(decode_seconds=2.0, verified_tokens=1)

choice = policy.choose(backend_supports_speculation=True,
                       queue_busy=False, memory_headroom=True)
# First eligible choice may be a bounded experiment with 2 draft tokens.
# Never emit the proposal without target-model verification.
print(choice)
```

## What to benchmark

Run identical model revision, tokenizer, decoding policy, prompts, and target
hardware. Compare target-only vs speculative under the same batch size and
offload layout; report p50/p95 end-to-end latency, tokens/s, accepted draft
tokens/target pass, SSD bytes read per emitted token, peak RAM/VRAM, and coding
task pass rates. The theoretical benefits do not guarantee a real speedup.
Speculative decoding may increase compute and memory use. For MoE models,
draft-token blocks may activate more *distinct* experts, increasing weight
traffic. Only benchmarking can settle that tradeoff.
