"""Bridge tests without a real LLM or Android SDK."""
import os
import time
import unittest
from unittest.mock import patch
from dataclasses import dataclass

from ironasu.mobile_agents import MobileJobs
from ironasu.core import Config, InferenceError


class FakeEngine:
    config = Config(max_concurrency=2)


@dataclass
class FakePart:
    index: int = 0
    segment_count: int = 1
    status: str = "succeeded"
    summary: str = "OK"
    error: str = ""


class FakeTeam:
    def __init__(self, engine):
        self.engine = engine

    def run(self, corpus, objective, *, partitions, workers, model_slots,
            model, cancel, progress):
        from types import SimpleNamespace
        assert model_slots <= 2
        progress(0, 1, 1, 1)
        return SimpleNamespace(
            complete=True, partitions=partitions, workers=workers,
            model_slots=model_slots, corpus_chars=len(corpus),
            total_segments=1, unique_segments=1, cache_hits=0,
            model_calls=1, synthesis="Result", parts=[FakePart()],
            evidence=[])


class MobileJobsTest(unittest.TestCase):
    def setUp(self):
        self.jobs = MobileJobs(FakeEngine())

    def test_validate_concurrency_and_parts(self):
        request = {"corpus": "abcd", "objective": "Review", "workers": 17}
        with self.assertRaises(InferenceError):
            self.jobs.start_corpus(request)
        request["workers"] = 16
        request["partitions"] = 16
        with self.assertRaises(InferenceError):
            self.jobs.start_corpus(request)

    @patch("ironasu.mobile_agents.CorpusAgentTeam", FakeTeam)
    def test_async_corpus_and_bounded_slots(self):
        job = self.jobs.start_corpus({
            "corpus": "abcdefghijklmnop", "objective": "Review",
            "partitions": 16, "workers": 16, "model_slots": 16})
        self.assertEqual(job["kind"], "corpus")
        for _ in range(100):
            state = self.jobs.get(job["id"])
            if state["state"] not in ("queued", "running"):
                break
            time.sleep(.01)
        self.assertEqual(state["state"], "completed")
        self.assertEqual(state["result"]["model_slots"], 2)
        self.assertEqual(state["result"]["synthesis"], "Result")
        self.assertEqual(state["completed"], 1)

    def test_unknown_job(self):
        with self.assertRaises(InferenceError):
            self.jobs.get("not-a-real-job")

    def test_harness_off_by_default(self):
        with patch.dict(os.environ, {"IRONASU_MOBILE_HARNESS_ENABLED": "0"}):
            with self.assertRaises(InferenceError) as ctx:
                self.jobs.start_harness({"task": "Review code"})
            self.assertEqual(ctx.exception.status, 403)


if __name__ == "__main__":
    unittest.main()
