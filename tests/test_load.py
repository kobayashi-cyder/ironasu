"""Concurrent-load correctness and fail-closed gate tests (no GPU required)."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from ironasu.benchmark import Completion
from ironasu.load import compare_load, evaluate_load, gate_main
from ironasu.regression import GatePolicy


def fixture(*, mismatch=False, usage=True):
    def baseline(prompt):
        time.sleep(0.005)
        return Completion(prompt.upper(), 0.005, 8 if usage else None)
    def candidate(prompt):
        time.sleep(0.005)
        return Completion(prompt if mismatch else prompt.upper(), 0.005,
                          8 if usage else None)
    result = compare_load(
        ["SECRET prompt", "another"], baseline=baseline, candidate=candidate,
        concurrencies=(1, 4), repeats=2, warmup=0)
    # Normalize the fake candidate's batch timing for deterministic gate tests.
    # Actual overlap and wall-clock measurement are exercised separately.
    for row in result["profiles"]:
        b, c = row["baseline"], row["candidate"]
        c["batch_wall_seconds"] = b["batch_wall_seconds"] / 2
        if c["measured_output_tokens"] is not None:
            c["concurrent_output_tokens_per_wall_second"] = (
                c["measured_output_tokens"] / c["batch_wall_seconds"])
    return result


class ConcurrentLoadTests(unittest.TestCase):
    def test_concurrent_batch_measures_overlapping_work_and_keeps_parity(self):
        lock = threading.Lock()
        active = [0]
        peak = [0]
        def fake(prompt):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                time.sleep(0.012)
                return Completion(prompt.upper(), 0.012, 9)
            finally:
                with lock:
                    active[0] -= 1
        report = compare_load(["a", "b", "c", "d"], baseline=fake,
                              candidate=fake, concurrencies=(1, 4),
                              repeats=2, warmup=0)
        self.assertGreaterEqual(peak[0], 2)
        self.assertEqual(len(report["profiles"]), 2)
        for row in report["profiles"]:
            self.assertEqual(row["exact_output_match_rate"], 1)
            self.assertEqual(row["baseline"]["samples"], 8)
            self.assertEqual(row["candidate"]["samples"], 8)
            self.assertEqual(row["baseline"]["measured_output_tokens"], 72)
            self.assertGreater(row["baseline"]["concurrent_output_tokens_per_wall_second"], 0)
            self.assertGreater(row["baseline"]["requests_per_wall_second"], 0)
        self.assertEqual(active[0], 0)
        # No pass claim: tiny real wall timings are noisy on shared CI runners.

    def test_mismatch_is_rejected_without_disclosing_prompts(self):
        report = fixture(mismatch=True)
        result = evaluate_load(report, GatePolicy(min_samples=4))
        self.assertFalse(result["passed"])
        self.assertFalse(result["profiles"][0]["checks"]["output_parity"])
        self.assertNotIn("SECRET prompt", json.dumps(report))

    def test_missing_usage_is_rejected(self):
        report = fixture(usage=False)
        self.assertIsNone(report["profiles"][0]["baseline"]["concurrent_output_tokens_per_wall_second"])
        with self.assertRaises(ValueError):
            evaluate_load(report)

    def test_detects_throughput_regression_independently_at_each_concurrency(self):
        report = fixture()
        row = report["profiles"][1]
        row["candidate"]["batch_wall_seconds"] *= 4
        row["candidate"]["concurrent_output_tokens_per_wall_second"] /= 4
        verdict = evaluate_load(report, GatePolicy(min_samples=4))
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["profiles"][0]["passed"])
        self.assertFalse(verdict["profiles"][1]["checks"]["concurrent_throughput"])

    def test_detects_latency_regression_and_missing_samples(self):
        report = fixture()
        report["profiles"][0]["candidate"]["p95_seconds"] *= 3
        verdict = evaluate_load(report, GatePolicy(min_samples=4))
        self.assertFalse(verdict["profiles"][0]["checks"]["p95_latency"])
        report = fixture()
        self.assertFalse(evaluate_load(report)["passed"])  # only four samples per level
        report["profiles"][0]["candidate"]["samples"] -= 1
        self.assertFalse(evaluate_load(report, GatePolicy(min_samples=4))["passed"])

    def test_rejects_falsified_or_nonfinite_throughput(self):
        report = fixture()
        report["profiles"][0]["candidate"]["concurrent_output_tokens_per_wall_second"] = 99999
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            evaluate_load(report)
        report = fixture()
        report["profiles"][0]["candidate"]["p95_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate_load(report)
        report = fixture()
        report["profiles"][0]["concurrency"] = 1
        report["profiles"][1]["concurrency"] = 1
        with self.assertRaises(ValueError):
            evaluate_load(report)

    def test_rejects_unsupported_settings_and_bad_input(self):
        fake = lambda p: Completion(p, 1, 1)
        for levels in [(), (0,), (1, 1), (129,), (True,)]:
            with self.assertRaises(ValueError):
                compare_load(["x"], baseline=fake, candidate=fake,
                             concurrencies=levels, warmup=0)
        with self.assertRaises(ValueError):
            compare_load([], baseline=fake, candidate=fake, warmup=0)
        with self.assertRaises(ValueError):
            evaluate_load({"prompts": 1})
        with self.assertRaises(ValueError):
            evaluate_load(fixture(), GatePolicy(latency_only=True))

    def test_gate_cli_json_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "load.json", Path(tmp) / "verdict.json"
            report = fixture()
            source.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(gate_main([str(source), "--min-samples", "4", "--output", str(dest)]), 0)
            self.assertTrue(json.loads(dest.read_text(encoding="utf-8"))["passed"])
            report["profiles"][0]["candidate"]["p95_seconds"] *= 2
            source.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(gate_main([str(source), "--min-samples", "4"]), 1)
            source.write_text("not json", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                gate_main([str(source)])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
