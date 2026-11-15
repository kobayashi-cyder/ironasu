"""A/B benchmark statistics and privacy regressions."""
import unittest

from ironasu.benchmark import Completion, compare, make_client, summarize, _url


class BenchmarkTests(unittest.TestCase):
    def test_identical_outputs_and_ratio(self):
        report = compare(
            ["hello", "test"],
            baseline=lambda p: Completion(p.upper(), 4.0, 8),
            candidate=lambda p: Completion(p.upper(), 2.0, 8),
            repeats=3, warmup=1,
        )
        self.assertEqual(report["exact_output_match_rate"], 1.0)
        self.assertEqual(report["p50_latency_speedup"], 2.0)
        self.assertEqual(report["candidate"]["measured_output_tokens"], 48)
        self.assertEqual(report["candidate"]["throughput_tokens_per_wall_second"], 4.0)

    def test_disagreement_exposes_indices_only(self):
        report = compare(["SECRET PROMPT"], baseline=lambda p: Completion("PRIVATE", 1, None),
                         candidate=lambda p: Completion("DIFFERENT", 1, None),
                         repeats=1, warmup=0)
        self.assertEqual(report["exact_output_match_rate"], 0)
        self.assertEqual(report["mismatches"], [{"prompt_index": 0, "repeat": 0}])
        self.assertNotIn("SECRET PROMPT", str(report))
        self.assertNotIn("PRIVATE", str(report))
        self.assertIsNone(report["candidate"]["throughput_tokens_per_wall_second"])

    def test_missing_usage_disables_token_throughput(self):
        report = summarize([Completion("x", 2, 10), Completion("x", 2, None)])
        self.assertIsNone(report["measured_output_tokens"])
        self.assertIsNone(report["throughput_tokens_per_wall_second"])

    def test_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            Completion("x", -1, 2)
        with self.assertRaises(ValueError):
            compare([], baseline=lambda p: None, candidate=lambda p: None)
        with self.assertRaises(ValueError):
            compare(["hello"], baseline=lambda p: None, candidate=lambda p: None, repeats=0)
        with self.assertRaises(ValueError):
            make_client("ftp://host/v1", "model")

    def test_endpoint_rewrite(self):
        self.assertEqual(_url("http://localhost:8080/v1/"),
                         "http://localhost:8080/v1/chat/completions")
        self.assertEqual(_url("http://localhost:8080/v1/chat/completions"),
                         "http://localhost:8080/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
