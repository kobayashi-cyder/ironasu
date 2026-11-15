"""Corpus chunking and <=16 read-only agent scheduling tests."""
from __future__ import annotations

from threading import Event, Lock
import time
import unittest

from ironasu.corpus_agents import (
    CorpusAgentTeam, MAX_CORPUS_CHARS, split_corpus, split_segments,
)


class FakeEngine:
    def __init__(self):
        self.lock = Lock()
        self.calls = []
        self.active = 0
        self.peak = 0

    def complete(self, payload):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(payload)
        time.sleep(0.015)
        with self.lock:
            self.active -= 1
        return {"choices": [{"message": {"content": "検証済みの短い要約"}}]}


class CorpusTest(unittest.TestCase):
    def test_lossless_corpus_partition_4_and_16(self):
        text = ("日本語文章\n" * 502) + "末尾"
        for n in (4, 16):
            with self.subTest(n=n):
                parts = split_corpus(text, n)
                self.assertEqual(len(parts), n)
                self.assertEqual("".join(parts), text)
                self.assertTrue(all(len(part) > 0 for part in parts))
                self.assertEqual("".join(split_segments(text, 256)), text)

    def test_minimum_partitions_and_input_limits(self):
        for invalid in (0, 2, 8, 17, "16", True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                split_corpus("a" * 100, invalid)
        with self.assertRaises(ValueError):
            split_corpus("too short", 16)
        with self.assertRaises(ValueError):
            split_corpus("a" * (MAX_CORPUS_CHARS + 1), 4)

    def test_16_agents_are_bounded_by_two_model_slots(self):
        engine = FakeEngine()
        report = CorpusAgentTeam(engine).run(
            "A line\n" * 100, "Evidence analysis",
            partitions=16, workers=16, model_slots=2, model="quality",
        )
        self.assertTrue(report.complete)
        self.assertEqual(report.partitions, 16)
        self.assertEqual(report.workers, 16)
        self.assertEqual(len(report.parts), 16)
        self.assertEqual(report.total_segments, 16)
        self.assertEqual(len(engine.calls), 17)  # 16 independent + 1 synthesis
        self.assertLessEqual(engine.peak, 2)
        self.assertGreater(engine.peak, 1)
        self.assertTrue(report.synthesis)
        for payload in engine.calls:
            self.assertFalse(payload["stream"])
            self.assertFalse(payload["x_ironasu_cache"])
            self.assertEqual(payload["x_ironasu_mode"], "quality")
            self.assertEqual(len(payload["messages"]), 2)
            self.assertNotIn("tools", payload)

    def test_long_partition_processed_by_multiple_segments(self):
        engine = FakeEngine()
        report = CorpusAgentTeam(engine).run(
            "long_line " * 4000,
            "Find claims",
            partitions=4, workers=4, model_slots=2,
            model="my-local-model",
        )
        self.assertTrue(report.complete)
        self.assertGreater(report.total_segments, 4)
        self.assertGreater(len(engine.calls), 5)
        self.assertEqual(report.corpus_chars, len("long_line " * 4000))
        self.assertTrue(all(payload.get("x_ironasu_model") == "my-local-model"
                            for payload in engine.calls))

    def test_cancel_before_start_makes_no_calls(self):
        engine = FakeEngine()
        done = Event()
        done.set()
        report = CorpusAgentTeam(engine).run(
            "a" * 100, "Objective", partitions=4,
            workers=4, model_slots=2, cancel=done,
        )
        self.assertFalse(report.complete)
        self.assertFalse(engine.calls)

    def test_limit_validation(self):
        team = CorpusAgentTeam(FakeEngine())
        for args in (
            {"workers": 17}, {"workers": 0}, {"model_slots": 0},
            {"model_slots": 17}, {"partitions": 8},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                team.run("x" * 100, "Analyze", **args)


if __name__ == "__main__":
    unittest.main()
