"""Unit tests for local benchmark reporting; no model server required."""
import unittest
from unittest.mock import patch

from ironasu.local_benchmark import measure, record


class LocalBenchmarkTests(unittest.TestCase):
    def test_record_preserves_optional_usage(self):
        row = record("uncached", 1, {
            "choices": [{"message": {"content": "ok"}}],
            "_ironasu": {"source": "upstream", "route": "quality"},
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }, 1.23456)
        self.assertEqual(row["elapsed_s"], 1.2346)
        self.assertEqual(row["source"], "upstream")
        self.assertEqual(row["total_tokens"], 14)

    def test_missing_usage_and_metadata(self):
        row = record("uncached", 1, {"choices": []}, 0.1)
        self.assertEqual(row["source"], "unknown")
        self.assertIsNone(row["completion_tokens"])

    def test_measure_sends_cache_opt_out_and_probe(self):
        received = []

        def fake_post(url, payload, timeout):
            received.append(payload)
            src = "cache" if len(received) == 4 else "upstream"
            return {"choices": [], "_ironasu": {"source": src}}, 0.5

        with patch("ironasu.local_benchmark.post_json", fake_post):
            output = measure("http://127.0.0.1:8765/v1", "test-model", 2, 2.0)
        self.assertEqual(len(received), 4)
        self.assertFalse(received[0]["x_ironasu_cache"])
        self.assertFalse(received[1]["x_ironasu_cache"])
        self.assertNotIn("x_ironasu_cache", received[2])
        self.assertEqual(received[2], received[3])
        self.assertEqual(output["cache_probe_hits"], 1)
        self.assertEqual(output["uncached_median_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
