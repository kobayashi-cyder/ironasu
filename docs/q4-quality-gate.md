# Q4 adoption: require quality, payload-size and timing evidence

`ironasu-q4-gate` is an **opt-in deployment decision gate**. It does not
quantize weights, run an evaluator or enable Q4 automatically. It combines:
(1) the existing `ironasu-benchmark-q4` checkpoint/timing report and
(2) independent, paired quality scores from a task suite.

A Q4 model is lossy. Matching output text is *not* sufficient proof of task
quality, and different but correct text is not an automatic failure. The
reported speedup from `ironasu-benchmark-q4` comes from generation wall
time, not decode-only tokens/s or GPU kernel time.

## 1. Benchmark the actual checkpoints

```sh
ironasu-benchmark-q4 /path/to/base-model \
  --q4-dir /path/to/q4-checkpoint \
  --device cuda --prompt "Implement binary search and explain it" \
  --tokens 64 --repeats 5 --output q4_bench.json
```

Choose the correct supported `--device` for the current implementation
(`cpu` or `cuda`). These devices are not equivalent to arbitrary ROCm,
Vulkan or Windows GPU installations. The Q4 pager is not a fused low-bit
matrix kernel. Quantization may reduce checkpoint payload while *slowing*
generation due to dequantization overhead.

The current benchmark runs the uncompressed model first and Q4 second.
OS page-cache/order effects can bias the comparison. For production
decisions, run independent trials with counterbalanced order and more
representative prompts using a separate experiment harness; do not treat
this single-prompt benchmark as a population-wide confidence estimate.

## 2. Produce independent scored results

Example `quality.json` (illustrative **format**, not real results):

```json
{
  "suite_id": "ironasu-coding-regression-2026-11",
  "evaluator": "deterministic unit tests and fixed acceptance assertions",
  "cases": [
    {"id": "task-001", "baseline_score": 1.0, "q4_score": 1.0},
    {"id": "task-002", "baseline_score": 1.0, "q4_score": 0.9}
  ]
}
```

Use at least **10 independent cases** in real trials. Scores must be
numeric and between 0 and 1 inclusive, with *higher better*, using the
**same evaluator, dataset, prompt templates, inference settings and scoring
rules** for baseline and Q4. Case IDs must be unique and stable. Use genuine
task-level evaluation (e.g. passing deterministic unit tests with defined
coverage), not a model judging its own output without calibration.

Scores are **provided by the operator**. This gate cannot establish their
provenance or verify that the scored checkpoints match the timing report.
Document model IDs, checkpoint hashes, evaluator version, decoding settings,
dataset, hardware and sample selection separately; verify them before
accepting the verdict. The two-line JSON above deliberately does **not**
satisfy the default minimum case requirement.

## 3. Run the fail-closed adoption gate

```sh
ironasu-q4-gate q4_bench.json --quality quality.json \
  --min-cases 10 --min-repeats 3 \
  --min-bytes-saved 0.10 \
  --max-median-slowdown 0.05 \
  --max-mean-quality-loss 0.01 \
  --max-case-quality-loss 0.05 \
  --output q4_verdict.json
```

Default requirements:

| Check | Requirement |
| --- | --- |
| Paired independent quality cases | At least 10 |
| Timing repetitions | At least 3 each, equal count |
| Checkpoint payload saved | At least 10% |
| Q4 median generation slowdown | At most 5% |
| Greedy repeat stability | Both variants stable across repeated runs |
| Mean quality loss | At most 0.01 absolute points on [0,1] |
| Worst individual-case quality loss | At most 0.05 absolute points on [0,1] |

Timing and checkpoint sizes are recomputed from raw samples; inconsistent
reported ratios cause a malformed-report failure. Missing or invalid scores
are **not silently inferred** from output equality. A severe individual-task
regression can block adoption even if the average quality is unchanged.

Exit codes: 0 = meets all evidence thresholds, 1 = fails at least one
threshold, 2 = invalid JSON/policy or missing report. A passing verdict is
a **conditional local decision**, not a proof that the Q4 model is safe or
faster for all tasks/hardware. The existing Q4 benchmark still does *not*
report end-to-end production p95 latency, GPU utilization, energy savings,
quantization-search cost or semantically verified correctness.

## 4. Next experiments

For the latest research on automatic quantization parameter selection
(AutoQS) and pre-quantization quality prediction, begin with a bounded,
fixed candidate set (e.g. several supported Q4 group sizes), use this gate
per candidate, record CPU/GPU memory and decode throughput externally, and
select a candidate only after held-out task evaluation. A reference to a
paper's speedup or predicted KL divergence does not replace actual ironasu
measurement. This implementation is **not** AutoQS or a Fisher/KL predictor.
