# ironasu v1.1 runtime execution and validation

This work connects previously independent components to **actual model forward
passes**, provides conservative per-request KV handling, and runs independently
bounded coding-agent teams. It does **not** establish a 3T parameter benchmark,
hardware speedup, CUDA weight-transfer pipelining, or sampled speculative
decoding. Those require the actual target model and machine.

## 1. Verified greedy N-gram speculation on a local paged model

Requires optional dependencies (`pip install -e '.[stream]'`), a trusted *local*
Transformers-compatible safetensors model, and a compatible target forward
implementation (not the experimental Kimi text-only adapter).

```sh
ironasu-stream /path/to/local/checkpoint \
  --prompt "Explain this function" --tokens 32 --device cuda \
  --speculative-ngram --profile --cpu-cache-mb 1024
```

The implementation runs the target model on a **multi-token causal block**
(`last_unprocessed_token + candidate tokens`, with already validated KV prefix
when safe). Candidate N-grams are per-request and untrusted. It compares every
candidate with the corresponding target argmax and emits only the verified
prefix plus the target's correction or bonus. An early mismatch never commits
the unverified suffix. When safe cropping of the target KV cache is not
supported, it invalidates the cache and recomputes the next step against the
full accepted prefix. Token generation is **greedy only**, without sampling or
repetition penalties; don't claim parity with custom decoding processors.

A `--speculative-ngram` invocation prints counters including target forward
passes and accepted/proposed drafts. The adaptive policy starts with measured
non-speculative requests and uses speculation only after enough observations.
It does not guarantee speculation will be chosen or that there will be enough
N-gram matches to benefit. The optional path is deliberately OFF by default.

**Important:** A model-specific cache may have sliding-window, KDA/MLA,
quantization, or positional semantics that prevent safe rollback. Never
reuse an unrecognized cache. This path has unit-test fake-cache coverage and
an optional tiny Qwen2 real-pager regression; it has NOT been validated on
Kimi K3 or on a huge model. A mismatch is a correctness bug, not an acceptable
approximation.

## 2. Bounded task scheduler up to 16 independent coding agents

```json
[
  {"task_id": "inspect", "prompt": "Summarize architecture", "priority": 10},
  {"task_id": "security", "prompt": "Identify risks in code", "priority": 10},
  {"task_id": "synthesis", "prompt": "Synthesize the findings",
   "depends_on": ["inspect", "security"]}
]
```

Save as `tasks.json`, then:

```sh
ironasu-team --tasks tasks.json --shared-context PROJECT_BRIEF.txt \
  --workspace . --workers 16 --model-slots 2 \
  --endpoint http://127.0.0.1:8765/v1
```

There can be 16 **agent sessions**, but only two concurrent LLM requests in
this example. This protects a single shared GPU from 16 simultaneous large
model calls. Each agent's SQLite journal/session is separate. Dependencies only
receive completed predecessor results; failed/uncertain/paused steps block
downstream tasks. Read-only behavior is enforced by disabling file writes and
command execution in every team worker. A unique `run_id` is printed for
manual recovery; no autonomous recovery or exactly-once external side effects
are claimed.

Shared context is copied into each model prompt. There is **no magical**
cross-agent shared attention state, and token overhead may remain high without
backend prefix caching.

## 3. Measured baseline-vs-speculative inference

Use *two separately configured* OpenAI-compatible servers with the **same**
target model/tokenizer/quantization/prompt template. The candidate server must
actually enable a supported target-side speculation implementation. For
example, llama.cpp server has explicit speculative options such as
`--spec-type ngram-mod`, `--spec-draft-model`, and `--spec-draft-n-max`
(check your installed server version; options differ across releases).

Create `prompts.json` as a JSON array of test prompts:

```sh
ironasu-benchmark-ab --prompts prompts.json \
  --baseline-url http://127.0.0.1:8081/v1 \
  --candidate-url http://127.0.0.1:8082/v1 \
  --model YOUR_MATCHING_MODEL_ID --repeat 3 --warmup 1 \
  --max-tokens 64 --output report.json
```

The report includes exact output matching, p50/p95 end-to-end latency, and
output tokens/s **only when the server provides completion token counts**.
Prompt/output text is not placed in the report. Do not interpret these as
SSD transfer, GPU memory or electrical power improvements: those require
real system monitoring. Compare the `ironasu-stream --speculative-ngram`
path against standard `ironasu-stream` using repeated fixed prompts on the
same hardware if available, but the A/B HTTP benchmark measures two upstream
servers, not this local CLI directly.

## Acceptance gates for a 3T-class claim

1. Actual checkpoint with correct custom architecture and tokenizer (not only
   metadata-only inspection).
2. Deterministic greedy target parity over prompts, stop token behavior,
   rejected draft correction, and KV mismatch/fallback.
3. Repeated measurements of target calls, accepted draft tokens, p50/p95,
   decode tokens/s, RAM/VRAM peaks, and SSD/PCIe bytes per emitted token.
4. Multiple batch, draft-length, and memory-pressure conditions; no unbounded
   queues, caches or expert prefetch. Compare baseline on the same hardware.
5. No acceptance of guessed tokens; no untrusted model code executed unless
   explicitly opted in and reviewed.

No claim of 3T-class model operation or speedup is made in this version.
