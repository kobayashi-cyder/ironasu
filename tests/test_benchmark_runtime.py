import unittest
from ironasu.benchmark_runtime import percentile, summarize


class BenchmarkTests(unittest.TestCase):
    def test_correctness_gates_gain(self):
        results = {
            "baseline": [{"output": "ok", "seconds": 2.}, {"output": "ok", "seconds": 2.}],
            "kv": [{"output": "ok", "seconds": 3.}, {"output": "ok", "seconds": 1.}],
            "bad": [{"output": "not ok", "seconds": .01}],
            "baseline_after": [{"output": "ok", "seconds": 2.},
                               {"output": "ok", "seconds": 2.}],
        }
        report = summarize(results)
        self.assertTrue(report["reference_stable"])
        self.assertEqual(report["comparisons"]["kv"]["warm_relative_gain"], .5)
        self.assertIsNone(report["comparisons"]["bad"]["warm_relative_gain"])

    def test_unstable_baseline_suppresses_gains(self):
        data = {
            "baseline": [{"output": "ok", "seconds": 2.}],
            "kv": [{"output": "ok", "seconds": .1}],
            "baseline_after": [{"output": "different", "seconds": 2.}],
        }
        result = summarize(data)
        self.assertFalse(result["reference_stable"])
        self.assertIsNone(result["comparisons"]["kv"]["warm_relative_gain"])

    def test_reject_bad_inputs_and_percentiles(self):
        with self.assertRaises(ValueError):
            summarize({"baseline": [{"output": "ok", "seconds": 0}]})
        self.assertEqual(percentile([1, 3, 5], 50), 3)
        with self.assertRaises(ValueError):
            percentile([], 95)
