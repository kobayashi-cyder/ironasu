# Speculative initialization experiment

This implementation adds a bounded, CPU-only metadata-state primitive. It is not a model-execution speedup, GPU KV cache, tensor cache, AirLLM reimplementation, draft-token decoder, or durable storage.

StateStore.reset() atomically switches to an empty generation in O(1); previous snapshots stay readable. StateStore.fork() starts an immutable copy-on-write overlay. try_promote() evaluates a caller-provided validation callback and performs compare-and-swap under a lock; stale candidates cannot publish. Every 32 commits, the overlay is compacted with an O(live-state) copy to keep read traversal bounded.

State values must be immutable bytes. Delta payload bytes and modified-key count are independently bounded (Python object overhead not included). No side effects such as file writes, external tool actions, or in-place tensor mutations may occur in speculative branches.

SpeculativeInitializer races 1–8 pure candidate builders. The first VERIFIED candidate can publish. Invalid candidates, exceptions, and stale candidates are rejected. Losing builders may continue until finished; a follow-on speculative run is refused until they drain. Closing the pool waits for all work.

Example:

~~~python
from ironasu.speculative_state import StateStore, SpeculativeInitializer

state = StateStore()
with SpeculativeInitializer(state) as initializer:
    outcome = initializer.run({
        "reuse": lambda branch: branch.write("prefix", b"cached"),
        "recompute": lambda branch: branch.write("prefix", b"verified"),
    }, verify=lambda name, snap: snap.get("prefix") == b"verified")
assert outcome.committed
assert state.snapshot().get("prefix") == b"verified"
~~~

Validation:

~~~sh
python -m unittest discover -s tests -v
python -m ironasu.speculative_state --keys 10000 --rounds 100
~~~

The CLI times fundamentally different operations (full dict rebuild versus empty-root reset); any ratio is NOT an inference speedup. Before integrating the primitive with LayerPager or speculative KV reuse, measure real TTFT, tokens/sec, peak RAM, SSD IO, rejected candidates, correctness, and target-model parity. Require checkpoint/tokenizer/position/mask/adapter identity for KV state. Keep this path opt-in until reproducible improvement is measured.
