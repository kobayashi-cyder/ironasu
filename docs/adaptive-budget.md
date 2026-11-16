# Adaptive request budgets and measured work avoidance

The first stage of ironasu's adaptive controller coordinates **enforceable**
request decisions across the existing gateway: auto routing under pressure,
cache use and maximum upstream wait. This does **not** imply native GPU
scheduling, tensor/KV memory budgets, FLOP counting or verified model-quality
equivalence. It never suppresses model-provided Reasoning.

## Request-scoped policy

Submit an optional `x_ironasu_budget` object to
`POST /v1/chat/completions` (works with both ordinary and streaming replies):

```json
{
  "model": "ironasu",
  "messages": [{"role": "user", "content": "Explain this function"}],
  "temperature": 0,
  "x_ironasu_budget": {
    "preserve_auto_quality": true,
    "allow_cache": true,
    "max_queue_ms": 500
  }
}
```

Supported options:

| Field | Default when budget object present | Meaning |
| --- | --- | --- |
| `preserve_auto_quality` | `true` | Suppress **only** the existing pressure-triggered downgrade of ambiguous short automatic requests. Explicit `fast` routing and known simple-task routing still work. |
| `allow_cache` | `true` | Allow deterministic exact-cache and Singleflight reuse. Set `false` to force a new upstream response. Existing tool/nondeterministic cache restrictions remain. |
| `max_queue_ms` | not set | Maximum waiting time for an upstream slot or in-flight duplicate response. Effective upstream slot limit is the smaller of this budget and `IRONASU_QUEUE_TIMEOUT_S`. Value 0 means no waiting; it does not block access to an already-free slot. Valid range 0–120000. |

Bad types, unknown fields and out-of-range limits receive HTTP 400.
When the budget object is **absent**, the previous legacy routing and
queue policy is preserved. The Studio Web and native desktop chat clients
send `{"preserve_auto_quality": true}` to avoid silently downgrading
ambiguous requests under pressure; neither UI imposes an extra queue cap.

Queue limits cannot cancel upstream decoding after execution starts, change
a model's context length, cap VRAM, bound TTFT/TPOT or guarantee an end-to-end
latency SLA. Even `max_queue_ms:0` may require time for request parsing,
cache checks and already-started backend work. Streaming responses remain
uncached and model-provided Reasoning is not disabled.

Nonstreaming JSON replies include `_ironasu.execution_budget` when an
explicit budget was submitted. The metadata contains policy, selected route,
queue budget and cache policy only; prompts, credentials and model internals
are not copied. Streaming clients can inspect aggregate counters instead.

## Evidence-based savings, not invented GPU metrics

The existing authenticated `GET /metrics` and `GET /ui/status` expose:

```json
{
  "execution_budget": {
    "explicit_plans": 4,
    "quality_guard_requested": 4,
    "queue_limit_requested": 1,
    "cache_bypass_requested": 0
  },
  "efficiency": {
    "observed": {
      "avoided_upstream_requests": 2,
      "cache_responses": 1,
      "singleflight_responses": 1,
      "successful_upstream_responses": 2,
      "reused_reported_output_tokens": 14,
      "reuse_missing_token_usage": 0
    },
    "derived": {"response_reuse_fraction": 0.5},
    "unmeasured": [
      "GPU compute avoided",
      "GPU memory saved",
      "FLOPs saved",
      "energy saved",
      "counterfactual latency saved"
    ]
  }
}
```

These are **illustrative values**, not measured hardware performance.

- `avoided_upstream_requests` counts **successfully returned** responses
  served through exact cache or Singleflight without a separate upstream
  completion request. Failed or timed-out duplicate waiters do not count.
- `successful_upstream_responses` counts completed upstream attempts;
  streaming completions are included when their context closes normally.
- `response_reuse_fraction` is avoided / (avoided + successful upstream
  responses), or `null` when none have completed. It is derived from counts,
  not a measured speedup. It is **not** a cache hit probability under a
  different workload.
- `reused_reported_output_tokens` sums the original response's
  `usage.completion_tokens` only when present for **every** reused reply;
  otherwise it is `null` and the number of missing-usage events is reported.
  It is not a measurement of saved GPU computation.
- All counters are process-local and reset when the gateway restarts.
  No prompt text, cache keys, user-identifying tags or generated answers
  appear in these counters. Under concurrent load, requests and wall time
  overlap: do not add request durations to assert energy or latency savings.

The Web Studio inference panel and native desktop Chat sidebar display
avoided upstream responses and derived reuse fraction. Their existing
TTFT/tokens-per-second indicators remain client observations, not hardware
telemetry.

## Validation workflow

Run the standard-library unit tests, including `tests/test_budget.py`
and `tests/test_studio.py`, then on an actual supported inference server
test with cache on/off, low/high queue contention, and pressure before
enabling more aggressive optimization:

```sh
python -m unittest discover -s tests -v
ironasu-benchmark-load --help
ironasu-load-gate --help
```

Before adopting GPU paging, speculative Drafts, KV reuse or quantization,
measure those options using the existing A/B regression gates, task quality
checks, p95 latency and memory limits. This policy layer does **not** implement
backend-specific GPU/KV controls or REAL Loss training; these remain separate
opt-in validation projects.
