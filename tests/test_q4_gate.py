"""CPU-only contract tests for Q4 evidence gating. No model or GPU required."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ironasu.q4_gate import Q4GatePolicy, evaluate_q4, main


def fixture():
    benchmark = {
        "checkpoint_payload": {
            "baseline_bytes": 1000, "q4_bytes": 450, "q4_fraction": 0.45,
        },
        "baseline": {"generation_s": [1.0, 1.1, 0.9], "stable_output": True,
                     "generated_text": "before"},
        "q4": {"generation_s": [0.90, 0.95, 1.00], "stable_output": True,
               "generated_text": "after (lossy is allowed)"},
        "speed_ratio_q4_vs_baseline": 1.0 / 0.95,
        "identical_generated_text": False,
        "recommendation": "validate_on_programming_test_suite",
    }
    quality = {
        "suite_id": "fixed-code-eval-v1",
        "evaluator": "deterministic unit-test pass rate",
        "cases": [
            {"id": f"case-{i}", "baseline_score": 0.8, "q4_score": 0.8}
            for i in range(10)
        ],
    }
    return benchmark, quality


class Q4GateTests(unittest.TestCase):
    def test_pass_with_independent_quality_even_when_text_differs(self):
        bench, quality = fixture()
        result = evaluate_q4(bench, quality)
        self.assertTrue(result["passed"])
        self.assertFalse(bench["identical_generated_text"])
        self.assertEqual(result["quality_cases"], 10)
        self.assertAlmostEqual(result["checkpoint_payload_saved_fraction"], 0.55)
        self.assertGreater(result["median_generation_speedup_q4_over_baseline"], 1)

    def test_missing_or_insufficient_quality_fails_closed(self):
        bench, quality = fixture()
        quality["cases"] = []
        verdict = evaluate_q4(bench, quality)
        self.assertFalse(verdict["passed"])
        self.assertIsNone(verdict["baseline_mean_score"])
        self.assertFalse(next(x["passed"] for x in verdict["checks"] if x["name"] == "mean_quality_loss"))
        quality["cases"] = fixture()[1]["cases"][:9]
        self.assertFalse(evaluate_q4(bench, quality)["passed"])

    def test_loss_beyond_limit_blocks_adoption(self):
        bench, quality = fixture()
        for case in quality["cases"]:
            case["q4_score"] = 0.77
        result = evaluate_q4(bench, quality)
        self.assertFalse(result["passed"])
        self.assertIn("mean_quality_loss", [x["name"] for x in result["checks"] if not x["passed"]])

    def test_single_severe_regression_blocks_average_pass(self):
        bench, quality = fixture()
        # The average drop is 0.009; the individual drop is 0.09.
        quality["cases"][0]["q4_score"] = 0.71
        result = evaluate_q4(bench, quality)
        self.assertFalse(result["passed"])
        rejected = [x["name"] for x in result["checks"] if not x["passed"]]
        self.assertIn("worst_case_quality_loss", rejected)
        self.assertNotIn("mean_quality_loss", rejected)

    def test_late_latency_or_insufficient_saving_blocks(self):
        bench, quality = fixture()
        slow = deepcopy(bench)
        slow["q4"]["generation_s"] = [1.2, 1.2, 1.2]
        slow["speed_ratio_q4_vs_baseline"] = 1 / 1.2
        self.assertIn("median_generation_slowdown",
                      [x["name"] for x in evaluate_q4(slow, quality)["checks"] if not x["passed"]])
        large = deepcopy(bench)
        large["checkpoint_payload"] = {
            "baseline_bytes": 1000, "q4_bytes": 990, "q4_fraction": 0.99,
        }
        self.assertFalse(evaluate_q4(large, quality)["passed"])

    def test_unstable_run_and_insufficient_repeat_blocks(self):
        bench, quality = fixture()
        bench["q4"]["stable_output"] = False
        self.assertFalse(evaluate_q4(bench, quality)["passed"])
        bench["q4"]["stable_output"] = True
        bench["baseline"]["generation_s"] = [1.0]
        bench["q4"]["generation_s"] = [0.95]
        self.assertFalse(evaluate_q4(bench, quality)["passed"])

    def test_invalid_provenance_missing_fields_and_malformed_scores_fail(self):
        bench, quality = fixture()
        for update in [
            lambda q: q.pop("suite_id"),
            lambda q: q.update({"evaluator": ""}),
            lambda q: q["cases"][0].update({"q4_score": None}),
            lambda q: q["cases"][0].update({"q4_score": float("nan")}),
            lambda q: q["cases"][0].update({"q4_score": 1.2}),
            lambda q: q["cases"][0].update({"baseline_score": True}),
            lambda q: q["cases"][0].update({"id": q["cases"][1]["id"]}),
            lambda q: q.update({"cases": "all"}),
        ]:
            q = deepcopy(quality)
            update(q)
            with self.subTest(q=q), self.assertRaises(ValueError):
                evaluate_q4(bench, q)

    def test_report_integrity_validation(self):
        bench, quality = fixture()
        for mutator in [
            lambda b: b["checkpoint_payload"].update({"q4_fraction": 0.2}),
            lambda b: b.update({"speed_ratio_q4_vs_baseline": 3}),
            lambda b: b["baseline"].update({"generation_s": [1.0, -1.0, 1.0]}),
            lambda b: b["q4"].update({"generation_s": [1.0, 1.0]}),
            lambda b: b["baseline"].update({"stable_output": 1}),
            lambda b: b["checkpoint_payload"].update({"q4_bytes": True}),
        ]:
            b = deepcopy(bench)
            mutator(b)
            with self.subTest(benchmark=b), self.assertRaises(ValueError):
                evaluate_q4(b, quality)

    def test_explicit_policy_can_change_thresholds(self):
        bench, quality = fixture()
        for case in quality["cases"]:
            case["q4_score"] = 0.79
        strict = Q4GatePolicy(max_mean_quality_loss=0)
        forgiving = Q4GatePolicy(max_mean_quality_loss=0.02)
        self.assertFalse(evaluate_q4(bench, quality, strict)["passed"])
        self.assertTrue(evaluate_q4(bench, quality, forgiving)["passed"])
        for kwargs in [
            {"min_cases": 0}, {"min_repeats": -1},
            {"min_bytes_saved": 1.01}, {"max_mean_quality_loss": -0.1},
            {"max_median_slowdown": float("inf")},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Q4GatePolicy(**kwargs)

    def test_cli_exit_codes_and_verdict_artifact(self):
        b, q = fixture()
        with tempfile.TemporaryDirectory() as root:
            bench_path = Path(root, "q4.json")
            quality_path = Path(root, "quality.json")
            verdict_path = Path(root, "verdict.json")
            bench_path.write_text(json.dumps(b), encoding="utf-8")
            quality_path.write_text(json.dumps(q), encoding="utf-8")
            args = [str(bench_path), "--quality", str(quality_path),
                    "--output", str(verdict_path)]
            self.assertEqual(main(args), 0)
            self.assertTrue(json.loads(verdict_path.read_text(encoding="utf-8"))["passed"])
            q["cases"][0]["q4_score"] = 0
            quality_path.write_text(json.dumps(q), encoding="utf-8")
            self.assertEqual(main(args), 1)
            self.assertFalse(json.loads(verdict_path.read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    unittest.main()
