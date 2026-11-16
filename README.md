# ironasu

**v1.9.0 isolated development missions:** Opt-in concurrent Git worktree editing (up to 16 agents) with dependency-aware commits, separate baseline/candidate verification, explicit conflict reporting, durable SQLite metadata and manual fast-forward approval. The checked-out branch is **not** edited by a successful isolated run. Worktree isolation is **not an OS sandbox**; run only in trusted repositories. See [isolated mission guide](docs/isolated-missions.md).

**v1.8.0 — Durable Mission Controller:** GitHub-style software missions can now
create a bounded task DAG, run up to 16 read-only agent sessions (2 model calls
by default), persist progress in SQLite, verify approved project checks, and
attempt up to two explicit permission-gated repairs. Qt6 gains a **ミッション**
tab; `ironasu-mission` offers the same workflow from the terminal. Editing and
test execution are **off by default**; see [mission documentation](docs/mission.md).
This is a bounded development harness, not an unrestricted computer agent.

**ironasu Studio v1.7.0:** Native PySide6 / Qt6 desktop interface replaces the previous Tk version as the default on Windows. The UI uses no browser or embedded WebView and provides drag-resizable panels, high-DPI support, Japanese-font-aware typography, rounded dark controls, chat/Reasoning telemetry, model downloads and the 16-agent corpus workflow. Install the bundled Windows EXE or use `pip install -e ".[desktop]"`. Previous UI entrypoints remain as `ironasu-studio-tk` and `ironasu-studio-web`. See [Qt6 desktop documentation](docs/qt-desktop.md).

## Q4 adoption quality gate

**New:** `ironasu-q4-gate` combines the existing `ironasu-benchmark-q4`
timing/checkpoint report with **independent, paired task-quality scores**.
It rejects inadequate evidence, per-task quality regressions, unexpectedly
slow quantization and negligible checkpoint savings. Exact-text parity
is not mistaken for semantic correctness, and missing GPU/energy measurements
are never invented. See [Q4 quality gate and validation workflow](docs/q4-quality-gate.md).

## Adaptive execution budget + evidence dashboard

**New:** the existing gateway now accepts an optional `x_ironasu_budget`
object to preserve auto-route quality under pressure, bound per-request
queue waiting and disable exact-cache reuse when requested. Desktop and
Web Studio default to pressure-safe auto routing. The native/Web dashboards
show **measured upstream calls avoided** through successful cache/Singleflight
reuse and a separately derived reuse ratio; no fictitious GPU/FLOP savings
are presented. See [budget API, evidence definitions and limits](docs/adaptive-budget.md).

## v1.6 — independent desktop GUI + 16 corpus agents

**ironasu Studio now opens its own native window rather than a browser.**
The Windows ironasu-studio.exe uses built-in Tk controls and starts without
Chrome, Edge, WebView2, an HTTP UI server, or a separate Python installation.
The browser interface remains available via ironasu-studio-web.

The desktop includes an エージェント ×16 tab: 4/16 lossless ordered corpus
partitions, up to 16 read-only analysis workers, separately bounded model
requests (two by default) and a final synthesis with JSON export.
See [native desktop and corpus agents](docs/desktop.md).


## ironasu Studio (offline Web UI)

**v1.6.1 agent optimisation:** Native 16-worker desktop corpus analysis now uses 4/16 lossless partitions, adaptive subchunks, same-run single-flight reuse of identical text, bounded hierarchical synthesis and source-indexed JSON exports with measured call savings. See [corpus optimisation](docs/corpus-optimization.md).

**New in v1.5:** TTFT/tok/s/Reasoning diagnostics, separate model-provided
reasoning output, opt-in LM Studio autoload and **one-click startup of a saved
GGUF** using an installed llama-server runtime. Model download and startup are
managed in the local Web UI. The Windows package does not bundle llama.cpp;
point `IRONASU_LLAMA_SERVER` to a trusted runner executable before using
native GGUF chat. See [managed GGUF and performance guide](docs/runtime.md).


**New in v1.4: Native GGUF installation without LM Studio or Ollama.**
Search public Hugging Face models, inspect downloadable GGUF sizes, download
verified weights to ironasu-managed local storage, resume incomplete transfers,
cancel active jobs and remove installed files. Downloads are SHA-256 verified.
No model-runtime backend is required for installation, but a GGUF-compatible
runtime is still required for inference. See [Native GGUF installation](docs/native-install.md).


**New in v1.3: Model installation from the UI.** Click **「＋ モデルを追加」** to download models directly into LM Studio (native v1 model-management API) or Ollama (native Pull API), view download progress, inspect installed models, and explicitly load LM Studio models. No direct installation support is claimed for other backend families; see [モデルの直接インストール](docs/model-install.md).

**Windowsのインストーラー（Python不要）とオフラインwheel配布に対応。** 詳細は [インストールと対応バックエンド](docs/install.md) を参照してください。`ironasu-studio` でLM Studio・Ollama等のローカルサーバーを自動検出して起動できます。

LM Studio風の**オフライン起動可能なWeb UI**を同梱しています。`python -m ironasu` で起動して、ブラウザで **http://127.0.0.1:8765/ui** を開いてください。CDN、外部フォント、Node.jsは不要です。チャット、モデルルーティング、SSEストリーミング（上流対応時）、推論設定、実行メトリクスを利用できます。モデル推論にはローカルOpenAI互換バックエンドが別途必要です。詳しくは [オフラインWeb UIの使い方](docs/studio.md)。


## v1.1 experimental runtime connection

- **Verified greedy N-gram decoding** now has an opt-in `ironasu-stream --speculative-ngram` path. Candidate tokens are verified in a real paged target-model forward pass; rejected KV prefixes are cropped or invalidated and replayed. This is greedy-only, tested on a local tiny Qwen2, and **not** validated for Kimi K3.
- **16-session DAG team scheduler** via `ironasu-team --tasks tasks.json --workers 16 --model-slots 2` supports shared briefs, dependency summaries, bounded model calls and independent persistent read-only sessions. This does not share attention/KV state between agents.
- **A/B benchmark** via `ironasu-benchmark-ab` reports exact response parity, p50/p95, and tokens/s where available for two servers with the same target model. No 3T/70B hardware result is asserted.

**New: fail-closed performance regression gate.** Run `ironasu-benchmark-ab` against
baseline and candidate inference servers, then `ironasu-regression-gate ab.json`
to reject output mismatches, insufficient samples, p95 latency regression or
sequential tokens/s regression. Speculative cycle timing now includes draft
lookup and scheduler overhead rather than only target execution. See
[measurement commands and limitations](docs/performance-regression.md).

**New: bounded admission and concurrent A/B load measurement.** The gateway
limits waiting upstream calls and singleflight followers (default 32;
`IRONASU_MAX_WAITING_REQUESTS=0` disables waiting); `/metrics` includes
active/waiting gauges. `ironasu-benchmark-load` and `ironasu-load-gate`
measure and validate actual overlapping HTTP workloads at user-selected
concurrency levels. See [concurrent performance evaluation](docs/concurrent-load.md).

See [implementation and validation](docs/runtime-validation.md). The
production out-of-core performance, optimized CUDA weight prefetch, and
architecture-specific Kimi KV handling still require real-device integration.


## Unified ironasu v1.0 (integrated package)

The project now ships several independently tested capabilities in **one Python
package**; they are not yet a single automatic GPU execution pipeline.

| Component | Interface | Status |
| --- | --- | --- |
| Adaptive inference gateway | `python -m ironasu` or `ironasu` | Functional HTTP proxy (model backend required) |
| Out-of-core / sparse MoE / Kimi text experiments | `ironasu-stream`, `ironasu-moe-plan`, `ironasu-kimi-inspect` | Experimental; opt-in dependencies / trusted local code |
| Persistent coding-agent harness | `ironasu-harness` or `python -m ironasu.harness` | Functional, opt-in writes and tests; no OS sandbox |
| Adaptive speculative draft scheduling | `ironasu.scheduling` | Standalone policy primitives; no target verifier |
| Logical state reset and speculative initialization | `ironasu.speculative_state` | Standalone metadata primitive; NOT KV cache rollback |

Install from the checkout with `python -m pip install -e .`. The gateway
remains the default `ironasu` executable; harness is a separate command
to avoid silently changing existing applications.

Example (run the LLM upstream separately, then start gateway in terminal 1):

```sh
python -m ironasu
```

Run the harness in terminal 2 through the gateway, with a tool-calling-capable
quality model available in the upstream:

```sh
ironasu-harness "Explain this repository" --workspace . --session demo --endpoint http://127.0.0.1:8765/v1
```

For direct backend use, omit the harness `--endpoint` override. The gateway
routes tool calls to its upstream and does not cache tool-call responses.
See [harness safety and usage](docs/HARNESS.md),
[speculation scheduler](docs/scheduler-experiment.md), and
[logical initialization](docs/speculative-initialization.md).

**Correctness boundary:** candidate tokens from `NgramDraftIndex` are **not
verified generations**. The scheduler is not wired to a target-model
speculative verification kernel, CUDA memory transfers or a real AirLLM
checkpoint loader; no hardware speedup is established. The logical state
initializer does not restore KV caches or roll back file/tool side effects.
Experimental model support may require custom architecture adaptations.


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

Connect Continue/VS Code to `http://127.0.0.1:8765/v1` with model name `ironasu`. SSE streaming is supported when the configured upstream backend supports OpenAI-style `text/event-stream` responses; otherwise use non-streaming requests.

## Request controls

- `x_ironasu_mode`: `auto` (default), `fast`, `quality`.
- `x_ironasu_cache`: `true` (default) or `false`. Caching applies only when `temperature=0` and there are no tools/tool_choice. Time-sensitive requests should set `false`.
- Response `_ironasu.route`: `fast` or `quality`; `_ironasu.source`: `upstream`, `cache`, or `singleflight`.
- `stream=true` passes through backend SSE with existing admission limits. It bypasses exact caching and cannot be combined with `x_ironasu_files`; unsupported backends fail explicitly instead of pretending streaming works.

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
- Comprehensive streaming tool-call semantics and production VS Code integration testing (basic SSE passthrough now exists).

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


## v0.4: Bounded CPU weight cache (no 70B performance claim)

For long autoregressive generations, a very large model can re-read the same
weights from checkpoint shards for **every** token. The new optional
`--cpu-cache-mb` setting retains a **fixed subset of complete stages** in CPU
memory. It does not cache arbitrary weight slices, skip decoder blocks, change
model outputs, or reduce the model's arithmetic FLOPs.

```bash
python -m ironasu.stream ./models/qwen2-local --prompt 'Explain this code' \
  --device cpu --tokens 8 --cpu-cache-mb 1024

# Explicitly prioritize useful stages within the same memory budget:
python -m ironasu.stream ./models/qwen2-local --prompt 'Hello' \
  --device cuda --tokens 8 --cpu-cache-mb 2048 \
  --pin-stage lm_head --pin-stage model.embed_tokens
```

The cache is **off by default**, to avoid unexpected RAM use on laptops with
16GB total system memory. When enabled, checkpoint tensor *headers* are
inspected without materializing the entire checkpoint. The static admission
policy prioritizes the output head and token embedding, then decoder stages;
explicit `--pin-stage` overrides that order. A stage is only retained if its
**entire** checkpoint payload fits within the remaining fixed byte budget.
Entries are never evicted just because a much larger model is streamed through
RAM; that would cause a sequential LRU scan to thrash. All cached tensors are
released by `close()`.

`pager.snapshot()['cpu_cache']` reports `budget_bytes`, `occupied_bytes`,
`planned_stages`, `cached_stages`, `hits`, `misses`, `disk_read_bytes`, and
`avoided_read_bytes`. These byte counters measure the **logical safetensors
payloads requested** and avoided by this cache; they are *not* measurements of
physical SSD I/O. OS page cache, safetensors memory mapping, and GPU transfers
make physical I/O different. `prefetch=True` remains compatible with caching,
but may temporarily allocate an additional stage-sized CPU buffer beyond the
configured cache budget. Peak overall RSS includes model activations, KV cache,
Python/PyTorch allocation, and staging, so `--cpu-cache-mb` is **not a total
process RAM limit**.

The cache uses CPU RAM even for CUDA inference. It avoids checkpoint reads on
cache hits, but GPU weights are still copied for each forward; USB4/eGPU
transfer costs are **not** eliminated. For SSD-bound workloads, compare output
parity and wall time with `--cpu-cache-mb 0` against a fixed cache budget on
identical prompts. If useful weights are larger than the budget or the bottleneck
is compute/transfer, a speedup may be zero. Synthetic unit tests validate
correctness and cache hits. Optional CI integration generates tiny GPT-2 and
Qwen2 models locally and verifies greedy multi-token parity, but full-size Qwen,
70B, MoE, FP8, CUDA and the target ASUS remain unverified.


## v0.6 — 重みIDハッシュテーブル（既存キャッシュの重複排除）

`ironasu` の固定CPUキャッシュを、ステージ名でなく **safetensorsのテンソル名** をキーとする辞書（ハッシュテーブル）に変更しました。
モデル全体をハッシュ計算するのではなく、検証済みのチェックポイント内でテンソル名を同一性の根拠として扱います。Pythonの辞書がキーの等価性も確認するため、通常のハッシュ衝突による誤共有は起こしません。

- 重み共有（tied embedding / output head）がある場合、同じテンソルをCPU RAMに二重保持しません。
- どのレイヤーからでもキャッシュ済みテンソルを参照でき、ディスク読み込みが不要になります。
- キャッシュの容量予算は**ユニークなテンソル**に対して計算します。`--cpu-cache-mb` は従来どおり有効です。
- `pager.snapshot()["cpu_cache"]` に `granularity`, `cached_tensors`, `tensor_hits`, `tensor_misses`, `deduplicated_planned_bytes`, `avoided_read_bytes` が追加されました。

```bash
ironasu-stream ./models/my-model --device cpu --tokens 8 --cpu-cache-mb 512 \
  --prompt "Write Python unit tests"
```

**再現可能な小型テスト**：入力Embeddingと出力ヘッドが同じテンソルを指すToyDecoderで、3回のforwardについて、そのテンソルのcheckpoint読み込みがキャッシュなしの6回から1回に減り、出力はビット単位で一致しました。これは重み共有テンソルに関する**読み込み回数の比較**であり、実際のSSD I/O量・時間・FLOPsの削減率ではありません。重み共有が少ないモデルでは効果は限定的です。

**適用限界**：ハッシュテーブル自体で70Bの計算量が減るわけではありません。モデルの大規模化は層単位のSSDストリーミングが実現し、ハッシュテーブルは重複するRAM使用量と一部I/Oを削減します。KVキャッシュは別のしくみであり、任意のプロンプトをハッシュキーだけで安全に再利用できるわけではありません。キャッシュは現在プロセス内・モデルインスタンス内に限定されています。実機の推論速度はベンチマークで評価してください。

## v0.5: **IroNasu Compass — 測る・選ぶ・守る・戻す**

ironasuはAirLLMの単なる模倣ではありません。「大きいモデルを動かす」だけでなく、
**どこが遅いかを測り、実際に改善する設定だけを選ぶ**ための実験装置です。

1. **測る** — 同じプロンプト・同じトークン数で、基準・先読み・固定キャッシュ・両方を実測する。
2. **選ぶ** — ベースラインのレイヤー計測から、CPUキャッシュに固定する候補を提案して試す。
3. **守る** — 出力の一致を確認し、最低改善率（既定10%）を満たさない候補は不採用にする。
4. **戻す** — エラー・応答不一致・基準結果の不安定化では、ベースラインを選択する。

開始時の自動チューニングは**明示的なオプトイン**であり、モデルやチェックポイントを書き換えません。

```bash
# メモリ追加なしで安全側の比較: 先読みON/OFF、各2回＋ベースライン再測定
ironasu-tune ./models/my-model --device cpu --tokens 2 --repeats 2

# CPUキャッシュ最大512 MiBで4〜5候補を比較（追加RAMが必要）
ironasu-tune ./models/my-model --device cpu --cache-mb 512 \
  --prompt 'Explain the purpose of the function.' --tokens 2 --repeats 2 \
  --output evidence.json

# 自動選択した候補で生成まで行う（毎回チューニングするため重い）
ironasu-stream ./models/my-model --device cpu --prompt 'Review this code' \
  --tokens 8 --cpu-cache-mb 512 --auto-tune
```

**評価の限界と注意**：文字列の出力一致は一つの入力での確認であって、意味的正しさ・
全入力での精度・ロジット一致を保証しません。OSのページキャッシュ、熱制御、GPUの非同期
計算、バックグラウンド負荷により計測が偏る場合があります。GPUでは同期プロファイリングを
有効にして試験します。容量制限を超えてRAMを使わない保証でもありません。CLIは実機の
計算能力を検証するものではなく、70Bなどに対する高速化を断定するものでもありません。
`ironasu-tune` は各候補ごとにモデルを初期化して推論します。巨大モデルでは試験そのものが
非常に長くなります。なお、この実装はモデルのFLOPsを直接削減せず、先読みや重み再利用の
候補を試すものです。品質評価には別途コードテストが必要です。

## v0.7: experimental Q4 out-of-core weights (opt-in)

`ironasu` now has its own small **symmetric group-wise Q4** checkpoint format.
This is an experimental, *lossy* weight packer, **not GGUF/GPTQ/AWQ compatibility**
and not a fused 4-bit inference kernel. Q4 is stored as packed 4-bit values with
per-group FP32 scales in per-tensor `safetensors` files; small and nonfloating
weights are kept verbatim. The original checkpoint is left unchanged.

**Why:** the out-of-core pager rereads weights on every generated token when RAM
cannot cache them. Reducing source bytes can reduce storage and CPU-to-GPU transfer,
though unpacking adds CPU/GPU computation and temporary memory. There is **no
speedup guarantee**; numerical values and generated text can change.

```bash
pip install -e '.[stream]'

# Convert a trusted LOCAL safetensors checkpoint. Output must be outside
# the input directory and nonexisting/empty; this does NOT delete originals.
ironasu-quantize-q4 ./models/qwen2-model --output ./models/qwen2-q4 --group-size 64

# Architecture configuration/tokenizer still come from the original directory.
ironasu-stream ./models/qwen2-model --q4-dir ./models/qwen2-q4 \
  --device cpu --tokens 8 --prompt 'Write a Python function to sort a list.'

# Compare the same prompt/repeats; baseline runs first (page-cache-order bias).
ironasu-benchmark-q4 ./models/qwen2-model --q4-dir ./models/qwen2-q4 \
  --prompt 'Write a Python function to sort a list.' --tokens 4 \
  --repeats 2 --output q4-evidence.json
```

The benchmark records baseline/Q4 median generation latency, indexed tensor
payload bytes, cache read counters and decoded text equality. The benchmark
**does not auto-activate Q4**: verify code correctness using a test suite and
run counterbalanced A/B trials on the actual target hardware before claiming
faster inference. Output equality for one prompt does not prove equivalence.

Limitations and resource cautions:
- A full original checkpoint is required during the conversion. The Q4 output
  is additional disk usage; plan enough free space for both while converting.
- Conversion is one tensor at a time, not a whole-model tensor allocation; the
  largest individual tensor and temporary FP32 intermediates still need RAM.
- Dequantization currently materializes the full layer weights at compute time.
  Peak activation + GPU weight + unpack intermediate allocations may still OOM.
- This is an additive experimental format with one file per tensor; huge models
  may encounter filesystem metadata and open-file overhead. It is **not** a
  proven faster substitute for optimized GGUF/llama.cpp kernels.
- For now use dense, supported Transformers models with `safetensors` weights.
  FP8/MoE layouts, arbitrary custom architectures and 70B+ model correctness
  have not been validated.
- `--q4-dir` cannot currently be combined with `--auto-tune`; the existing
  autotuner requires bit-identical output, which is inappropriate for lossy Q4.
- The checkpoint manifest uses local path validation and `safetensors` rather
  than pickle. Keep the original checkpoint secure; do not run untrusted code.

## v0.8: demand-paged MoE experts and native MXFP4 reference decoder (experimental)

**Purpose:** the full expert bank of a large MoE layer cannot be loaded in 16GB
RAM. If a router selects a small number of expert modules for a token, ironasu
can page **only those executed modules** out of local safetensors, evict their
live parameters immediately after forward, and optionally cache hot expert
weights in a bounded CPU LRU table. The enclosing decoder layer retains its
router and attention weights only for its own forward. No source tensor names
or checkpoint bytes need to be hashed.

**Safety and support boundary:** this mode only works for a recognized decoder
architecture with individually-called `nn.ModuleList` containers named
`experts`, under `model.layers.N...experts`. Models with vectorized packed
experts, fused operators that access expert parameters directly, or unsupported
architectures **are not runnable** through this adapter. It does not implement
Kimi K3's KDA/Gated-MLA/Attention Residuals, SiTU-GLU routing, multimodal stack,
or model-specific custom execution. Kimi K3's weights cannot currently be fed
to `ironasu-stream` to obtain an answer. Full Kimi inference is **unfinished**.

Try a compatible small, local ModuleList-MoE checkpoint (no downloads or remote
code) with:

```bash
pip install -e '.[stream]'
python -m unittest discover -s tests -p test_moe.py -v
python -m unittest discover -s tests -p test_mxfp4.py -v

ironasu-stream ./models/supported-moe --device cpu \
  --expert-paging --expert-cache-mb 256 \
  --prompt 'Explain this Python function' --tokens 4
```

`--expert-cache-mb` bounds **stored CPU expert tensor payloads**, not temporary
expert materialization, CUDA allocations, shard OS page cache or the base
model layers. A too-large single expert is streamed without caching.
`--expert-paging` is explicitly opt-in and rejects models without recognized
`ModuleList` experts; `--expert-cache-mb` is invalid unless paging is enabled.
For larger prefill batches many experts may be selected: sparse loading does
**not** promise only 16 experts total per forward, or any speedup.

### Metadata-only Kimi checkpoint planner

```bash
# Needs only model.safetensors.index.json. No tensor shards, GPU or PyTorch.
ironasu-moe-plan ./models/Kimi-K3 \
  --layer language_model.model.layers.1.block_sparse_moe.experts \
  --expert 0 --expert 5 --expert 17
```

The planner reports the selected tensor key count and indexed shard count for
known expert IDs. This does **not** identify the router's actual decisions or
estimate physical SSD bytes. It refuses missing IDs; it never downloads weights
or executes untrusted model code. It recognizes `.weight_packed` and
`.weight_scale` index names.

### Native MXFP4 (reference implementation, not optimized kernels)

`ironasu.mxfp4.MXFP4Index` wraps `SafetensorsIndex` and presents an original
MXFP4 pair `<name>.weight_packed` (U8 low nibble first) + `<name>.weight_scale`
(U8 E8M0 exponent per 32 elements) as a *logical* `<name>.weight`. The
`MXFP4Tensor` decoder uses OCP E2M1 FP4 values, no re-quantization or original
FP16 checkpoint. It produces BF16 tensors by default (change dtype through
Python API), with FP32 intermediate dequantization. It does **not** perform
low-bit matrix multiplies: it fully expands one active matrix to the target
CPU/GPU for computation. This is distinct from ironasu's own symmetric Q4
format, which is not native Kimi MXFP4.

For a compatible local ModuleList MoE model whose keys use the native pairing:

```bash
ironasu-stream ./models/compatible-native-mxfp4-moe \
  --expert-paging --native-mxfp4 --expert-cache-mb 512 \
  --device cpu --prompt 'hello' --tokens 4
```

The implementation is validated using synthetic native-format expert weights,
exact OCP FP4 values, and a small real Torch MoE forward with output parity;
**no 2.8T-model runtime or accuracy/speed benchmark has been performed**.
MXFP4 conversion expands intermediate tensors, so peak memory and wall time may
still be substantial. For Kimi, the remaining engineering task is an executable
Kimi architecture adapter that can dispatch the top-k routed experts while
handling its special attention/residuals and multimodal components, plus an
optimized MXFP4 GEMM implementation and full end-to-end testing.

## v0.9 — Kimi K3 text-path adapter (experimental, not full K3 support)

The official Kimi K3 checkpoint is **multimodal**. Its `language_model.`
namespace holds the Kimi Linear text model. `ironasu-kimi-inspect` reads only
local `config.json` and the checkpoint **index JSON** (when present). It does
not import the official model code, read tensor payloads, or download weights:

```bash
ironasu-kimi-inspect ./models/Kimi-K3
```

The optional `--kimi-text-only --trust-local-code` path uses **the original
Kimi Python code supplied in the local model directory** to perform its KDA,
Gated MLA, depth attention residual, and latent MoE mathematics. Ironasu
supplies safetensors weights through its out-of-core hooks. It does not vendor
Kimi code or reimplement its kernels. It strips the `language_model.` prefix
*in the reader*, skips all vision tensors, loads output-attention-residual
weights for their full enclosing text-model forward, and pages only the expert
modules actually invoked via `ModuleList.__call__`.

```bash
# Requires full K3 local checkpoint and its locally audited architecture files.
# The Kimi implementation also requires its own runtime dependencies
# (including fla-core and the appropriate flash-attention/CUDA backend).
ironasu-stream ./models/Kimi-K3 --kimi-text-only --trust-local-code \
  --device cuda --expert-cache-mb 1024 --tokens 1 \
  --prompt 'def binary_search(a, x):'
```

**Security / compatibility:** `--trust-local-code` executes model-supplied
Python, including all dependencies. Review and pin these files first. The
entrypoint refuses unfamiliar local `auto_map` and missing Python files, but
that does not make untrusted code safe. All checkpoint reads remain local;
there is no network downloader. In normal mode, `trust_remote_code=False`
remains mandatory. **This mode has not yet been run against Kimi K3's 1.56 TB
checkpoint** and no 2.8T token has been generated by ironasu. CPU support,
KDA/MLA/AttnRes cache compatibility, genuine MXFP4 bit layout across all
K3 tensors, device peak RAM, and final numerical equivalence with the
official inference engine require real-model tests. The model's license and
upstream runtime prerequisites also apply. Small synthetic text models only
verify stage lifecycle, routing, key remapping and memory-release invariants.
