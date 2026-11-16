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
        self.assertEqual(len(engine.calls), report.model_calls)
        self.assertEqual(report.cache_hits, report.total_segments - report.unique_segments)
        self.assertGreater(report.cache_hits, 0)
        self.assertLess(report.model_calls, 17)  # repeated chunks are mapped only once
        self.assertEqual(len(report.evidence), report.total_segments)
        self.assertEqual(len({(x.partition, x.segment) for x in report.evidence}), 16)
        self.assertLessEqual(engine.peak, 2)
        # Exact duplicate reuse can legitimately reduce peak model concurrency to one.
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

    def test_adaptive_salience_keeps_all_bytes_and_prioritizes_dense_sections(self):
        ordinary = ("This is regular background text. " * 450)
        salient = ("WARNING: CRITICAL issue, check the evidence.\n" * 360)
        base = split_segments(ordinary)
        highlights = split_segments(salient)
        self.assertEqual("".join(base), ordinary)
        self.assertEqual("".join(highlights), salient)
        self.assertGreater(len(highlights), len(base))
        self.assertTrue(all(len(x) <= 8000 for x in highlights))
        self.assertTrue(all(len(x) <= 8000 for x in base))

    def test_shared_evidence_preserves_occurrences_and_provenance(self):
        engine = FakeEngine()
        corpus = ("repeated line\n" * 160)
        report = CorpusAgentTeam(engine).run(
            corpus, "Trace evidence", partitions=16, workers=16, model_slots=2,
        )
        self.assertTrue(report.complete)
        self.assertEqual(report.total_segments, len(report.evidence))
        self.assertEqual(report.cache_hits, report.total_segments - report.unique_segments)
        self.assertGreater(report.cache_hits, 0)
        self.assertLessEqual(report.unique_segments, report.total_segments)
        ordered = list(report.evidence)
        self.assertEqual([(x.partition, x.segment) for x in ordered],
                         sorted((x.partition, x.segment) for x in ordered))
        self.assertEqual(sum(x.end - x.start for x in ordered), len(corpus))
        self.assertEqual("".join(corpus[e.start:e.end] for e in ordered), corpus)
        from hashlib import sha256
        self.assertTrue(all(sha256(corpus[e.start:e.end].encode()).hexdigest() == e.sha256
                            for e in ordered))
        self.assertEqual(len({x.label for x in ordered}), report.total_segments)

    def test_synthesis_does_not_silently_cut_off_sixteen_partitions(self):
        engine = FakeEngine()
        report = CorpusAgentTeam(engine).run(
            "Part\n" * 1000, "Find contradictions",
            partitions=16, workers=16, model_slots=2,
        )
        self.assertTrue(report.complete)
        # At least some synthesis requests carry actual partition names
        # into bounded model prompts; no 48k-character single-pass synthesis.
        prompts = [p["messages"][1]["content"] for p in engine.calls]
        self.assertTrue(any("Partition 1" in p for p in prompts))
        self.assertTrue(all(len(p) < 16000 for p in prompts))

    def test_failed_shared_map_cannot_forge_complete_report(self):
        class FailedEngine(FakeEngine):
            def complete(self, payload):
                with self.lock:
                    self.calls.append(payload)
                raise RuntimeError("test backend failure")
        engine = FailedEngine()
        report = CorpusAgentTeam(engine).run(
            "same content\n" * 120, "Read", partitions=16,
            workers=16, model_slots=2,
        )
        self.assertFalse(report.complete)
        self.assertFalse(report.synthesis)
        self.assertTrue(all(p.status == "failed" for p in report.parts))
        self.assertGreater(report.cache_hits, 0)
        self.assertEqual(report.model_calls, report.unique_segments)

    def test_run_cache_is_not_persistent_across_requests(self):
        engine = FakeEngine()
        team = CorpusAgentTeam(engine)
        first = team.run("repeat\n" * 100, "A", partitions=4, workers=4)
        initial = len(engine.calls)
        second = team.run("repeat\n" * 100, "B", partitions=4, workers=4)
        self.assertTrue(first.complete and second.complete)
        self.assertEqual(second.model_calls, len(engine.calls) - initial)
        self.assertGreater(second.model_calls, 0)

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
