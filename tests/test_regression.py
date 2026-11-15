"""Hardware-independent tests for the fail-closed regression gate."""
import json
from pathlib import Path
import tempfile
import unittest

from ironasu.benchmark import Completion, compare
from ironasu.regression import GatePolicy, evaluate_ab, main


def report(*, baseline_seconds=2.0, candidate_seconds=1.5,
           same_output=True, with_usage=True, prompts=5, repeats=2):
    return compare(
        [f"case {i}" for i in range(prompts)],
        baseline=lambda p: Completion(p.upper(), baseline_seconds, 10 if with_usage else None),
        candidate=lambda p: Completion(p.upper() if same_output else p, candidate_seconds,
                                       10 if with_usage else None),
        repeats=repeats, warmup=0,
    )


class RegressionGateTests(unittest.TestCase):
    def test_passes_quality_p95_and_throughput(self):
        result = evaluate_ab(report())
        self.assertTrue(result["passed"])
        self.assertEqual(result["p95_latency_ratio"], 0.75)
        self.assertAlmostEqual(result["sequential_throughput_ratio"], 4 / 3)
        self.assertTrue(result["throughput_evaluated"])

    def test_p95_slowdown_fails(self):
        result = evaluate_ab(report(candidate_seconds=2.2))
        self.assertFalse(result["passed"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "p95_latency")["passed"])

    def test_mismatched_outputs_fail_even_when_faster(self):
        result = evaluate_ab(report(candidate_seconds=0.1, same_output=False))
        self.assertFalse(result["passed"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "exact_output_parity")["passed"])

    def test_insufficient_samples_fail(self):
        result = evaluate_ab(report(prompts=1, repeats=1))
        self.assertFalse(result["passed"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "sample_count")["passed"])

    def test_missing_usage_rejected_without_explicit_opt_out(self):
        r = report(with_usage=False)
        with self.assertRaises(ValueError):
            evaluate_ab(r)
        result = evaluate_ab(r, GatePolicy(latency_only=True))
        self.assertTrue(result["passed"])
        self.assertIsNone(result["sequential_throughput_ratio"])
        self.assertFalse(result["throughput_evaluated"])

    def test_equal_tokens_required(self):
        r = report()
        r["candidate"]["measured_output_tokens"] += 1
        result = evaluate_ab(r)
        self.assertFalse(result["passed"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "equal_token_counts")["passed"])

    def test_malformed_and_nonfinite_metrics_rejected(self):
        r = report()
        r["candidate"]["p95_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate_ab(r)
        r = report()
        r["candidate"]["samples"] = True
        with self.assertRaises(ValueError):
            evaluate_ab(r)
        r = report()
        r["mismatches"] = [{"prompt_index": 0, "repeat": 0}]
        self.assertFalse(evaluate_ab(r)["passed"])
        with self.assertRaises(ValueError):
            GatePolicy(min_samples=0)
        with self.assertRaises(ValueError):
            GatePolicy(max_p95_regression=float("inf"))
        with self.assertRaises(ValueError):
            GatePolicy(min_throughput_gain=-0.1)

    def test_min_gain_is_enforced(self):
        r = report(baseline_seconds=2.0, candidate_seconds=1.98)
        gated = evaluate_ab(r, GatePolicy(min_throughput_gain=0.10))
        self.assertFalse(gated["passed"])
        self.assertTrue(next(c for c in gated["checks"] if c["name"] == "p95_latency")["passed"])

    def test_cli_writes_machine_readable_verdict_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ab.json"
            output = Path(tmp) / "gate.json"
            source.write_text(json.dumps(report()), encoding="utf-8")
            self.assertEqual(main([str(source), "--output", str(output)]), 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])
            source.write_text(json.dumps(report(candidate_seconds=4.0)), encoding="utf-8")
            self.assertEqual(main([str(source), "--output", str(output)]), 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])
            source.write_text("not json", encoding="utf-8")
            with self.assertRaises(SystemExit) as exc:
                main([str(source)])
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
