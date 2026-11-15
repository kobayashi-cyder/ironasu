# ironasu

**An adaptive compute-saving gateway for local code-assistant LLMs.**

The original goal is to use huge models on low-memory hardware, inspired by AirLLM. That goal has two distinct limits:

- **Memory/IO:** AirLLM streams weights layer by layer; it avoids loading all parameters at once, but repeatedly reading model weights is expensive.
- **Arithmetic:** A dense model normally still executes its layers for each generated token. **No arbitrary, lossless reduction in the FLOPs of a single huge-model inference is claimed here.**

This initial version targets **total inference work across real requests**, rather than claiming to accelerate each individual 70B/405B token. It sits in front of LocalAI, llama.cpp's OpenAI-compatible server, or any compatible service.

## Implemented (v0.1 prototype)

1. **Exact response cache:** identical, deterministic (`temperature: 0`) non-tool calls reuse responses within a configurable TTL; repeated inference is skipped.
2. **Singleflight:** concurrent identical requests share one upstream call rather than all computing the same result.
3. **Adaptive routing:** narrowly classified simple coding questions go to a small model; complex/security/multifile work stays on the quality model. Explicit `x_ironasu_mode: quality` is always respected.
4. **Pressure detection:** slow requests or high queue wait trigger short-request routing to the small model; three normal observations reset this state.
5. **Overload protection:** bounded upstream concurrency and queue timeout prevent unbounded memory/compute contention.
6. **Observability:** `/metrics` reports per-route calls, latency samples, wait times, cache/singleflight hits, errors, and pressure state.

**Important:** heuristic routing can reduce answer quality. Compare answers against a fixed coding-task evaluation set before changing defaults. There is no automatic layer-skipping or training in this prototype.

## Quick start

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

## Request controls

- `x_ironasu_mode`: `auto` (default), `fast`, `quality`.
- `x_ironasu_cache`: `true` (default) or `false`. Caching applies only when `temperature=0` and there are no tools/tool_choice. Time-sensitive requests should set `false`.
- Response `_ironasu.route`: `fast` or `quality`; `_ironasu.source`: `upstream`, `cache`, or `singleflight`.
- `stream=true` currently returns HTTP 400 instead of silently pretending streaming works.

## Configuration

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

## Next experiments (not implemented)

- Pluggable AirLLM profiling: separate disk load time, host↔device transfer, attention, MLP/MoE and decoding.
- CPU↔GPU layer prefetch and bounded asynchronous I/O with hardware measurements.
- Compatible backend speculative decoding (e.g. llama.cpp draft or n-gram), accepting that it may improve latency without reducing total FLOPs.
- Architecture-aware, calibrated approximate layer/expert skipping, only behind an explicit opt-in and tested quality constraints.
- Streaming SSE and comprehensive tool-call behavior before production VS Code integration.

License: MIT (add license text when distributing publicly).

## v0.2: Code-context pruning + AirLLM measurements (experimental)

v0.2 adds **opt-in reduction of prompt context** and **a real AirLLM benchmark harness**. It does **not** replace AirLLM, perform model-layer skipping, guarantee quality, or claim to reduce the FLOPs needed for a dense output-token forward pass. The earlier v0.1 behaviors and their tests remain intact.

### Save prompt-prefill work on coding tasks

Pass a list of source files as `x_ironasu_files` (array of `{ "path": "...", "content": "..." }`), and optionally `x_ironasu_context_chars` (1000–100000, default 12000). The proxy finds 32-line windows with the most overlapping *literal identifiers* with the last user question, and sends only those excerpts to the configured model. This is an **approximate lexical retrieval heuristic**, not semantic search or dependency analysis; it may omit relevant code. **It is never enabled implicitly**. Disable it for correctness-sensitive multi-file changes until tested.

Example request body (send to `/v1/chat/completions`):

```json
{
  "messages": [{"role": "user", "content": "Explain validate_user"}],
  "temperature": 0,
  "x_ironasu_files": [
    {"path": "src/auth.py", "content": "def validate_user(token):\n    return verify_signature(token)\n"},
    {"path": "src/other.py", "content": "def unrelated():\n    return 1\n"}
  ],
  "x_ironasu_context_chars": 2000
}
```

The returned `_ironasu.context_retrieval` includes original and selected **source character counts**; those are not tokens or FLOPs. `/metrics` adds aggregated character counts. Private code remains on your machine unless the explicitly configured upstream backend is remote. For manual preview against your own local repository (no inference request):

```bash
python -m ironasu.context_cli --repo . --query 'fix validate_user in auth.py' --budget-chars 4000
```

The collector skips hidden files, common vendor directories and symlinks, limits file sizes, and only includes known text-source extensions. The proxy itself **never reads repository files**; you decide what to send.

### Benchmark AirLLM on the actual hardware

Install compatible AirLLM, PyTorch, and model dependencies in a separate environment. AirLLM models may require large downloads and disk space for layer sharding; benchmark a small model first.

```bash
python -m ironasu.airllm_benchmark \
  --model Qwen/Qwen3-4B --device cpu --max-new-tokens 8 \
  --prefetch both --output airllm-results.json
```

The benchmark records model load time, generation time, output tokens, Linux process block-read bytes (when available), optional CUDA peak allocated bytes, and best-effort timing of AirLLM's shard load/decode and device materialization methods. It compares prefetch ON/OFF **only if the generated token IDs match and effective prefetch states differ**. If `compression` disables prefetching in the installed AirLLM, no prefetch speedup is reported.

**Important limitations:** disk cache warms during the first run and influences the second, GPU work is asynchronous, and the method probes depend on AirLLM's internal names. Stage totals can overlap and must not be interpreted as percentages of wall time or GPU FLOPs. The script never auto-activates a slower or quality-changing alternative. Repeat with reversed prefetch order to validate speedups; no actual performance measurements from the ASUS are included here.

### Diagnose measured stages

Save a trace JSON (from trusted measurements of a *single* inference interval):

```json
{
  "wall_seconds": 10.0,
  "events": [
    {"stage": "disk_io", "seconds": 6.5},
    {"stage": "compute", "seconds": 2.1},
    {"stage": "transfer", "seconds": 0.9}
  ]
}
```

Then:

```bash
python -m ironasu.diagnostics trace.json
```

A stage is identified as a bottleneck when its observed duration is at least 40% of wall time; otherwise the diagnosis says `unresolved`. Possible stages are `disk_io`, `transfer`, `compute`, `prefill`, `queue`, `other`. AirLLM's probe outputs separate best-effort categories named `disk_io_plus_decode` and `transfer_plus_setup`; **they must not be naively converted to exact GPU compute time**. Use external tooling (such as a CUDA profiler or measured token-prefill timings) for the missing phases.

### Evaluation gate before enabling globally

Test with a stable set of coding prompts and projects. Compare baseline vs reduced context on actual compiled/tested results, not just answer similarity. Report original/selected token counts from the backend, correctness, p50/p95 latency, stage times, I/O bytes and peak RAM/VRAM. Revert the optimization if correctness regresses. In this repository's tests, the context-selection logic is exercised with fixtures, **not a running 70B model**.

## v0.3: Experimental native out-of-core inference (AirLLM-style)

**Scope:** Unlike the earlier proxy and AirLLM benchmarking wrapper, this contains
an **independent layer-streaming implementation** (`ironasu.paging`). For a trusted
**local** Hugging Face `safetensors` checkpoint, it constructs a Transformers model
with parameters on the `meta` device, then uses forward hooks to load the
embedding, individual decoder blocks, final norm and output head from SSD **one
module at a time**. Each module's weights are returned to `meta` immediately after
its forward pass. Original checkpoint shards are indexed in place, without a
second layer-shard copy. Optional one-stage CPU prefetch trades extra RAM for
potential I/O overlap; disabled by default.

This is a **limited reference implementation**, **not** AirLLM v4 feature parity.
The package does **not** copy AirLLM source code; see its Apache-2.0 original at
https://github.com/lyogavin/airllm . A dense model still executes all decoder
layers for each generated token. CPU offload is not CPU FLOPs reduction.

### Installation and use

```bash
python -m pip install -e '.[stream]'
# Download an openly licensed safetensors model separately using a trusted source,
# e.g. `hf download <HF-ID> --local-dir ./models/model-name`.
python -m ironasu.stream ./models/model-name \
  --prompt 'Write a Python binary search function.' --tokens 8 --device cpu
# CUDA (only if an NVIDIA CUDA runtime and driver are available):
python -m ironasu.stream ./models/model-name \
  --prompt 'Write a Python binary search function.' --tokens 8 --device cuda --profile
# Bounded CPU prefetch, extra RAM required:
python -m ironasu.stream ./models/model-name \
  --prompt 'Explain Python generators.' --tokens 4 --prefetch
```

`ironasu-stream` is the equivalent console entrypoint. `--profile` synchronizes
CUDA kernels to obtain more meaningful stage timings, at a performance cost.
The normal `ironasu` command still runs the separate OpenAI-compatible **proxy**;
the paged model is not yet exposed through its HTTP API.

**Supported in this version:** `model.layers` (Llama/Qwen2-style dense decoder),
`transformer.h` (GPT-2 style), and `gpt_neox.layers` (GPT-NeoX style), **subject to
checkpoint-layout checks**. Untied and tied embedding/head checkpoints are handled.
PyTorch forward parity is tested on a tiny locally generated model. There is no
claim that all real pretrained model versions or full generation paths have been
validated on the target ASUS hardware.

**Not supported:** automatic HF model downloads, pickle `.bin` weights, arbitrary
remote model Python code, quantized GGUF/FP8/MXFP4 or MoE weight dispatch, 4/8-bit
checkpoint recompression, GPU-specific kernels, streamed LoRA fine-tuning,
multimodal models, AirLLM's many vendor-specific architectures, or production
concurrency/streaming HTTP serving. Models with parameters outside recognized
stages fail closed instead of silently returning incorrect predictions.

**Memory caveats:** Although all *model weights* need not be resident at once,
peak memory still depends on the **largest layer**, current layer's disk tensors,
activation and **KV caches**, PyTorch overhead and (when enabled) one prefetched
layer. Model files must fit on disk; weights are not automatically deleted.
Model and code must be from a trusted source. For safer operation, local-only,
`trust_remote_code=False` is the default and remote checkpoints cannot be fetched.

### Correctness and bottleneck measurement

```bash
python -m unittest discover -s tests -v
python -m ironasu.diagnostics results.json
```

The pager reports per-module `disk_seconds`, `transfer_seconds`, `compute_seconds`,
load count, and peak simultaneous loaded stages. Disk timing includes OS page-cache
behavior and deserialization; when GPU synchronization is disabled compute timing
mostly covers enqueue and host bookkeeping. **No throughput improvement or FLOPs
saving is claimed until measured against a matching baseline and checked for
correct outputs**. The optional HF smoke test builds a tiny model locally without
network downloads; it runs only when `transformers` and `accelerate` are installed.
