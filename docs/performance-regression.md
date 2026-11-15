# Evidence-gated performance regression checks

`ironasu-benchmark-ab` measures two explicitly configured OpenAI-compatible
servers on the **same fixed prompts**, using sequential HTTP completion requests.
`ironasu-regression-gate` reads its JSON and exits **0=pass, 1=regression or
insufficient evidence, 2=malformed input/invalid policy**. No GPU is needed to
exercise the policy in unit tests. A real performance claim does require a real
model and reproducible hardware measurements.

## Run on your own hardware

Start two servers with the same weights, tokenizer, prompt template,
precision, decoding settings and model ID. The baseline should have the
optimization disabled; the candidate should change only one factor (e.g. FP8
KV, prefill scheduling, draft model). Use an actual supported backend;
ironasu's own gateway does not implement FP8 KV or native GPU kernels.

Save representative, non-sensitive prompts in `prompts.json`:

```json
["Explain a simple function.", "Write a small unit test."]
```

Example (use more prompts/repetitions than the minimal illustration):

```sh
ironasu-benchmark-ab \
  --prompts prompts.json \
  --baseline-url http://127.0.0.1:8001/v1 \
  --candidate-url http://127.0.0.1:8002/v1 \
  --model my-local-model \
  --repeat 5 --warmup 2 --output ab.json

ironasu-regression-gate ab.json \
  --min-samples 10 \
  --max-p95-regression 0.05 \
  --min-throughput-gain 0.03 \
  --output verdict.json
```

The A/B command exits 1 on text mismatches while writing the report; the
separate gate also rejects mismatches. Inspect the A/B report before deciding
whether to proceed to the gate.

**Default fail-closed thresholds**: at least 10 paired measured completions,
100% exact decoded-text parity, p95 end-to-end request latency no worse than
5%, and no regression in serialized output-tokens/HTTP-wall-second. Change
`--min-throughput-gain` to require a positive improvement. The baseline and
candidate must produce the **same positive total measured output token count**.
If your server omits usage counts, the gate rejects the report by default.
Only use `--latency-only` when explicitly choosing to skip the throughput
assertion; that verdict records throughput as *not evaluated*.

The JSON verdict includes each check, observed ratio, requirements and
limitations without copying prompts or generated text. Commit only redacted
reports and record the test's hardware/driver/model/backend versions separately.
Do not evaluate on confidential prompts unless storage and endpoints are trusted.

## What the gate does *not* prove

- Exact decoded text equality is not code correctness, semantic quality,
  probability parity or calibration. Use separate evaluation sets, tests and
  human review before changing precision or model weights.
- The benchmark is **sequential**: its tokens/s is not throughput under 1/8/32/128
  concurrent users. HTTP wall time includes network and queuing, not just
  GPU decode. The current report does not separately collect TTFT, TPOT,
  VRAM, GPU utilization, energy or power.
- Two different hosts with different CPU/GPU, network, page-cache state or
  thermal pressure may produce incomparable numbers. Warm both models,
  alternate order, repeat at different times and record context lengths and
  output lengths. A p95 from 10 samples is only a smoke check; use many more
  requests and confidence intervals for a production change.
- REAL Loss, TQIR and FP8 KV require backend-specific training/kernel support.
  They are **not implemented** by this gate. Validate them on supported
  hardware; never present theoretical memory reductions as measured speedups.

## Speculative decode timing

The experimental greedy paged decoder now records the entire per-cycle cost:
draft selection/proposal, synchronization, target verification and index
maintenance. Previously it measured target-forward time only, potentially
overstating speculative benefits. Its scheduler still requires a non-speculative
measured baseline, and all draft tokens are verified by the target before
emission. No native sampled speculative decoding, cross-request KV sharing
or general 3-trillion-parameter support is implied.
