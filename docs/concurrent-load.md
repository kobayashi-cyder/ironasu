# Concurrent load A/B and bounded gateway admission

The existing `ironasu-benchmark-ab` performs **serial** requests. Its
tokens/s is the total number of generated tokens divided by the sum of
per-request HTTP latency, so it cannot establish throughput under load.
The new `ironasu-benchmark-load` runs bounded concurrent HTTP requests at
specified concurrency levels and measures wall-clock batch throughput.

## Run

Start two separately configured OpenAI-compatible model servers. Keep model,
checkpoint, tokenizer, prompt template, hardware, precision, max output length
and sampling policy equivalent; change only the intended optimization.
Use the same representative prompt JSON array on both sides.

```bash
ironasu-benchmark-load \
  --prompts prompts.json \
  --baseline-url http://127.0.0.1:8001/v1 \
  --candidate-url http://127.0.0.1:8002/v1 \
  --model your-model-id \
  --concurrency 1,8,32 \
  --repeat 5 --warmup 2 --output load.json

ironasu-load-gate load.json \
  --min-samples 100 \
  --max-p95-regression 0.05 \
  --min-throughput-gain 0.03 \
  --output load-verdict.json
```

Supply at least as many prompt entries as the largest requested concurrency\n(e.g. at least 32 prompts when using `--concurrency 1,8,32`). The client\nrefuses higher worker counts rather than silently reporting unachievable\nconcurrency.\n\nThe benchmark alternates which server runs first on each repetition. It
never drives both servers simultaneously during the same trial: this limits
competition for resources but does not eliminate order, temperature, cache,
network or other environmental effects.

For each concurrency, the report includes p50/p95 per-request HTTP wall
latencies, requests/s, generated output tokens/elapsed batch wall second
(only if the server returns valid token counts), exact decoded-text match
fraction, and prompt indices of mismatches. It does not include user prompts
or output strings. Each request uses a separate HTTP connection from the
standard-library client; worker count bounds client overlap, not server
microbatch size.

The gate requires **every tested concurrency** to satisfy:

- At least `--min-samples` requests per side (default 10).
- 100% exact output-text match and equal positive token totals.
- Candidate p95 HTTP request latency no more than 5% worse than baseline,
  unless configured otherwise.
- Candidate concurrent output-tokens/batch-wall-second not below baseline,
  unless a higher gain threshold is specified.

Gate exit codes: 0 pass, 1 regression/insufficient samples, 2 malformed or
missing evidence. The benchmark returns 1 if any output mismatches occur and
still writes the JSON report. Use independent coding-task correctness and
long-context evaluations when changing quantization or model quality:
exact text match is a strict but incomplete criterion.

For production evaluation run more than 100 diverse prompts, repeat trials
at different times, report confidence intervals, and trace TTFT/TPOT with
a dedicated streaming profiler if needed. This benchmark does **not**
measure GPU utilization, KV cache occupancy, VRAM, power, energy, request
admission quality, or task-level semantic correctness. Do not treat small
fixture results as real hardware speedups.

## Bounded gateway queue

`IRONASU_MAX_CONCURRENCY` continues to cap active upstream calls. The new
`IRONASU_MAX_WAITING_REQUESTS` caps queued upstream calls **and** duplicate
Singleflight followers waiting for an in-flight cacheable request. Default
is 32, valid range 0–1024. Set 0 to reject new work immediately when all
upstream slots are occupied; idle upstream slots are still usable.

```bash
export IRONASU_MAX_CONCURRENCY=2
export IRONASU_MAX_WAITING_REQUESTS=8
export IRONASU_QUEUE_TIMEOUT_S=10
python -m ironasu
```

Excess work receives HTTP 503; slot acquisition timeout also receives 503.
The `/metrics` endpoint reports `active_requests`, `waiting_requests`,
`peak_active_requests`, `peak_waiting_requests` and `overloads`.
Counters expose no prompts or decoded contents. The pending limit avoids
unbounded **admitted** waiting but the built-in ThreadingHTTPServer itself
still spawns a thread for each accepted connection; deploy behind a
rate-limited, authenticated front end for hostile or public networks.
The current semaphore is not strict FIFO and is not a token-aware GPU
batch scheduler.

## Next integration step

Couple these measurements with TTFT/TPOT tracing and real per-backend
batch/KV scheduling controls. Do not automatically enable FP8 KV, REAL Loss,
speculative decoding or model-layer skipping until accuracy and hardware
constraints are validated with separate evidence.
