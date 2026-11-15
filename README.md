# ironasu

**An adaptive compute-saving gateway for local code-assistant LLMs.**

The original goal is to use huge models on low-memory hardware, inspired by AirLLM. That goal has two distinct limits:

- **Memory/IO:** AirLLM streams weights layer by layer; it avoids loading all parameters at once, but repeatedly reading model weights is expensive.
- **Arithmetic:** A dense model normally still executes its layers for each generated token. **No arbitrary, lossless reduction in the FLOPs of a single huge-model inference is claimed here.**

This initial version targets **total inference work across real requests**, rather than claiming to accelerate each individual 70B/405B token. It sits in front of LocalAI, llama.cpp's OpenAI-compatible server, or any compatible service.

## Current implementation status

The original README described a future v0.1 gateway as if it were already
implemented; the repository previously contained only this README. In the
`feature/adaptive-speculation-prefetch` branch, the actual implemented code
is the **experimental, backend-independent** module
`ironasu/scheduling.py`, including:

- A bounded, per-session **hash-table N-gram draft proposer**. Its output is
  unverified and must **never** be sent directly to the user.
- A **measurement-gated speculation policy** that chooses a draft length only
  when a verifier-capable backend is explicitly available, resource pressure
  is low, and measured decode-cycle latency justifies it.
- An **ordered, bounded layer prefetch pipeline** that overlaps a layer's
  loading with another layer's execution through an injected backend loader.
- Unit tests: `python -m unittest discover -s tests -v`.

**Not yet implemented:** the OpenAI-compatible proxy, AirLLM-compatible
weight streaming and CUDA synchronization, GPU-side multi-token verification,
real target/draft model integration, or any measured hardware speedup.
See [scheduler experiment design](docs/scheduler-experiment.md).

## Planned gateway capabilities (not implemented)

1. **Exact response cache:** identical, deterministic (`temperature: 0`) non-tool calls reuse responses within a configurable TTL; repeated inference is skipped.
2. **Singleflight:** concurrent identical requests share one upstream call rather than all computing the same result.
3. **Adaptive routing:** narrowly classified simple coding questions go to a small model; complex/security/multifile work stays on the quality model. Explicit `x_ironasu_mode: quality` is always respected.
4. **Pressure detection:** slow requests or high queue wait trigger short-request routing to the small model; three normal observations reset this state.
5. **Overload protection:** bounded upstream concurrency and queue timeout prevent unbounded memory/compute contention.
6. **Observability:** `/metrics` reports per-route calls, latency samples, wait times, cache/singleflight hits, errors, and pressure state.

**Important:** heuristic routing can reduce answer quality. Compare answers against a fixed coding-task evaluation set before changing defaults. There is no automatic layer-skipping or training in this prototype.

## Planned gateway quick start (not yet executable)

Requires Python 3.10+; proxy itself has **no runtime dependencies**.

```bash
python -m unittest discover -s tests -v
export IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1
export IRONASU_FAST_MODEL=qwen2.5-coder-1.5b
export IRONASU_QUALITY_MODEL=qwen2.5-coder-7b
python -m ironasu
```

The backend must run separately and expose `/v1/chat/completions`. Model names must match the IDs registered on your backend; the defaults are examples, not auto-downloads. On Windows PowerShell, use `$env:IRONASU_BACKEND_URL='http://127.0.0.1:8080/v1'` instead of `export`.

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ironasu","messages":[{"role":"user","content":"explain this Python function"}],"temperature":0}'
curl http://127.0.0.1:8765/metrics
```

Connect Continue/VS Code to `http://127.0.0.1:8765/v1` with model name `ironasu`. **Streaming is not supported yet**; configure non-streaming requests.

## Planned request controls

- `x_ironasu_mode`: `auto` (default), `fast`, `quality`.
- `x_ironasu_cache`: `true` (default) or `false`. Caching applies only when `temperature=0` and there are no tools/tool_choice. Time-sensitive requests should set `false`.
- Response `_ironasu.route`: `fast` or `quality`; `_ironasu.source`: `upstream`, `cache`, or `singleflight`.
- `stream=true` currently returns HTTP 400 instead of silently pretending streaming works.

## Proposed gateway configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `IRONASU_BACKEND_URL` | `http://127.0.0.1:8080/v1` | LocalAI/llama.cpp URL |
| `IRONASU_BACKEND_KEY` | empty | Bearer token for upstream |
| `IRONASU_FAST_MODEL` | `qwen2.5-coder-1.5b` | Small-model identifier |
| `IRONASU_QUALITY_MODEL` | `qwen2.5-coder-7b` | Quality-model identifier |
| `IRONASU_MAX_CONCURRENCY` | `2` | Maximum simultaneous upstream requests |
| `IRONASU_SLOW_REQUEST_S` | `20` | Pressure signal threshold |
| `IRONASU_QUEUE_TIMEOUT_S` | `10` | Reject saturated queue after this wait |
| `IRONASU_REQUEST_TIMEOUT_S` | `600` | Upstream timeout |
| `IRONASU_CACHE_ENTRIES` | `128` | LRU bound |
| `IRONASU_CACHE_TTL_S` | `300` | Maximum cache age (seconds) |
| `IRONASU_SERVER_KEY` | empty | Optional inbound bearer token |
| `IRONASU_HOST` / `IRONASU_PORT` | `127.0.0.1` / `8765` | Listener |

The proxy binds to loopback by default. A server key is required for non-loopback binding. Cache stays in RAM only; do not expose the proxy on the public internet without an authenticated reverse proxy and TLS. The backend URL is administrator-controlled, not supplied per request.

## Benchmark plan (measure; do not invent speedups)

Run the *same* fixed collection of programming tasks with the same token budget on the quality model alone and through ironasu. Report:

- `backend_calls` per model; cache/singleflight hit rates (directly measurable saved calls).
- p50/p95 **end-to-end latency**, queue wait, peak RAM/VRAM, and SSD reads.
- code compilation/tests passed, security/regression outcomes, human review for correctness.
- per-task total generated tokens, energy consumed and cost, where measurable.

The first easy win is skipping entire duplicate calls (exactly one avoided invocation per cache hit). Routing can reduce FLOPs but its savings depend on model and prompt; benchmark locally. **Zero speedup guarantees** for new/unique, complex requests routed to the large model.

## Next runtime integration experiments (not implemented)

- Pluggable AirLLM profiling: separate disk load time, host↔device transfer, attention, MLP/MoE and decoding.
- CPU↔GPU layer prefetch and bounded asynchronous I/O with hardware measurements.
- Compatible backend speculative decoding (e.g. llama.cpp draft or n-gram), accepting that it may improve latency without reducing total FLOPs.
- Architecture-aware, calibrated approximate layer/expert skipping, only behind an explicit opt-in and tested quality constraints.
- Streaming SSE and comprehensive tool-call behavior before production VS Code integration.

License: MIT (add license text when distributing publicly).
