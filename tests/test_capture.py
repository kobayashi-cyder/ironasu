"""Sequential low-RAM benchmark capture and validation tests."""
import json
import unittest

from ironasu.benchmark import Completion
from ironasu.capture import record_capture, compare_captures


class CaptureTests(unittest.TestCase):
    def test_reproducible_without_raw_output(self):
        prompts = ["private: abc", "private: def"]
        a = record_capture(prompts, lambda p: Completion(p.upper(), 2.0, 4),
                           model="same", max_tokens=64, repeats=2, warmup=0)
        b = record_capture(prompts, lambda p: Completion(p.upper(), 1.0, 4),
                           model="same", max_tokens=64, repeats=2, warmup=0)
        report = compare_captures(a, b)
        self.assertEqual(report["exact_output_match_rate"], 1.0)
        self.assertEqual(report["p50_latency_speedup"], 2.0)
        self.assertEqual(report["baseline"]["measured_output_tokens"], 16)
        saved = json.dumps(a)
        self.assertNotIn("private:", saved)
        self.assertNotIn("PRIVATE:", saved)

    def test_reject_mismatched_model_and_content(self):
        a = record_capture(["secret"], lambda p: Completion("yes", 1, None),
                           model="x", max_tokens=5, warmup=0)
        b = record_capture(["secret"], lambda p: Completion("no", 1, None),
                           model="x", max_tokens=5, warmup=0)
        self.assertEqual(compare_captures(a, b)["exact_output_match_rate"], 0)
        b["model"] = "other"
        with self.assertRaises(ValueError):
            compare_captures(a, b)

    def test_reject_mismatched_prompts_and_row_order(self):
        a = record_capture(["alpha", "beta"], lambda p: Completion("same", 1, 1),
                           model="x", max_tokens=5, warmup=0, repeats=1)
        b = record_capture(["gamma", "beta"], lambda p: Completion("same", 1, 1),
                           model="x", max_tokens=5, warmup=0, repeats=1)
        with self.assertRaises(ValueError):
            compare_captures(a, b)
        b["prompt_hashes"] = a["prompt_hashes"]
        b["rows"] = list(reversed(b["rows"]))
        with self.assertRaises(ValueError):
            compare_captures(a, b)


if __name__ == "__main__":
    unittest.main()
