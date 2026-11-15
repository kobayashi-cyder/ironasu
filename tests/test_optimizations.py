import time
import unittest

from ironasu.context import select_code
from ironasu.diagnostics import PhaseTrace, StageEvent, diagnose, from_json
from ironasu.airllm_benchmark import summarize
from ironasu.core import AdaptiveEngine, Config, InferenceError


class CaptureBackend:
    def __init__(self):
        self.calls = []

    def complete(self, payload):
        self.calls.append(payload)
        return {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class RetrievalTests(unittest.TestCase):
    def fixtures(self):
        return [
            {"path": "src/arithmetic.py", "content": "\n".join([f"def make_{i}(): return {i}" for i in range(400)])},
            {"path": "src/security.py", "content": "\n".join([f"def placeholder_{i}(): return None" for i in range(96)] +
                                                            ["def verify_password(password, expected_hash):", "    return check_hash(password, expected_hash)"] +
                                                            [f"def placeholder_after_{i}(): return None" for i in range(90)])},
        ]

    def test_retrieval_returns_relevant_source_and_omits_unrelated(self):
        files = self.fixtures()
        s = select_code(files, "Explain verify_password in security.py", 2500)
        self.assertIn("verify_password", s.text)
        self.assertIn("src/security.py", s.text)
        self.assertGreater(s.original_chars, s.selected_chars * 5)
        self.assertLessEqual(len(s.text), 2600)
        self.assertGreater(s.chunks_total, s.chunks_selected)

    def test_selection_is_deterministic(self):
        files = self.fixtures()
        self.assertEqual(select_code(files, "verify_password", 3000), select_code(files, "verify_password", 3000))

    def test_missing_or_malformed_context_rejected(self):
        with self.assertRaises(ValueError):
            select_code([{"path": "x.py", "content": "x", "unexpected": 1}], "x")
        with self.assertRaises(ValueError):
            select_code([{"path": "x\nsecret", "content": "foo"}], "foo")
        with self.assertRaises(ValueError):
            select_code([{"path": "x", "content": "foo"}], "foo", 100)

    def test_engine_context_opt_in_and_original_unchanged(self):
        backend = CaptureBackend()
        engine = AdaptiveEngine(Config(), backend)
        files = self.fixtures()
        payload = {"messages": [{"role": "user", "content": "Explain verify_password"}],
                   "x_ironasu_files": files, "x_ironasu_context_chars": 2500, "temperature": 0}
        out = engine.complete(payload)
        request = backend.calls[-1]
        self.assertEqual(payload["messages"][0]["content"], "Explain verify_password")
        self.assertNotIn("x_ironasu_files", request)
        self.assertNotIn("x_ironasu_context_chars", request)
        self.assertIn("verify_password", request["messages"][0]["content"])
        meta = out["_ironasu"]["context_retrieval"]
        self.assertGreater(meta["source_chars_omitted"], 0)
        snap = engine.diagnostic_snapshot()
        self.assertEqual(snap["context_retrieval"]["requests"], 1)
        self.assertEqual(engine.complete(payload)["_ironasu"]["source"], "cache")
        self.assertEqual(len(backend.calls), 1)

    def test_opt_out_does_not_append_any_source(self):
        backend = CaptureBackend()
        engine = AdaptiveEngine(Config(), backend)
        engine.complete({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(backend.calls[0]["messages"][0]["content"], "Hi")
        self.assertNotIn("context_retrieval", engine.complete({"messages": [{"role": "user", "content": "Hi again"}]})["_ironasu"])

    def test_invalid_engine_context_is_http_400(self):
        engine = AdaptiveEngine(Config(), CaptureBackend())
        with self.assertRaises(InferenceError) as ctx:
            engine.complete({"messages": [{"role": "user", "content": "Check"}],
                             "x_ironasu_files": [{"path": "a.py", "content": "x"}], "x_ironasu_context_chars": 2})
        self.assertEqual(ctx.exception.status, 400)


class DiagnosticsTests(unittest.TestCase):
    def test_dominant_disk_io_and_advice(self):
        report = diagnose([StageEvent("disk_io", 6), StageEvent("compute", 2)], 10)
        self.assertEqual(report["bottleneck"], "disk_io")
        self.assertIn("NVMe", report["advice"])

    def test_no_guesses_when_uninstrumented(self):
        self.assertEqual(diagnose([], 4)["bottleneck"], "unresolved")
        self.assertEqual(diagnose([StageEvent("disk_io", 1), StageEvent("compute", 1)], 10)["bottleneck"], "unresolved")

    def test_overlap_explicitly_not_additive(self):
        report = diagnose([StageEvent("disk_io", 7), StageEvent("compute", 8)], 10)
        self.assertGreater(sum(report["observed_seconds_by_stage"].values()), report["wall_seconds"])
        self.assertIn("Overlapping", report["warning"])

    def test_trace_span_and_json(self):
        trace = PhaseTrace()
        with trace.span("queue"):
            time.sleep(.001)
        report = trace.report(.005)
        self.assertGreater(report["observed_seconds_by_stage"]["queue"], 0)
        self.assertEqual(from_json({"wall_seconds": 5, "events": [{"stage": "compute", "seconds": 3}]})["bottleneck"], "compute")

    def test_invalid_trace_rejected(self):
        with self.assertRaises(ValueError):
            diagnose([StageEvent("wrong", 3)], 5)
        with self.assertRaises(ValueError):
            diagnose([StageEvent("queue", -1)], 5)
        with self.assertRaises(ValueError):
            diagnose([], 0)


class BenchmarkTests(unittest.TestCase):
    def test_comparison_checks_identical_output(self):
        both = summarize([
            {"prefetch_requested": False, "output_token_ids": [1, 2], "generation_seconds": 10},
            {"prefetch_requested": True, "output_token_ids": [1, 2], "generation_seconds": 5},
        ])
        self.assertTrue(both["exact_token_match"])
        self.assertEqual(both["prefetch_speedup_ratio"], 2)

    def test_output_changes_invalidate_speedup_ratio(self):
        both = summarize([
            {"prefetch_requested": False, "output_token_ids": [1], "generation_seconds": 10},
            {"prefetch_requested": True, "output_token_ids": [2], "generation_seconds": 5},
        ])
        self.assertFalse(both["exact_token_match"])
        self.assertIsNone(both["prefetch_speedup_ratio"])


if __name__ == "__main__":
    unittest.main()

class StageProbeTests(unittest.TestCase):
    def test_probe_preserves_results_and_tracks_real_calls(self):
        from ironasu.airllm_probe import AirLLMStageProbe
        class Model:
            def _load_streamed_layer(self, x):
                return x * 2
            def move_layer_to_device(self, value):
                return value + 3
        m = Model()
        original = m._load_streamed_layer
        with AirLLMStageProbe(m) as probe:
            self.assertEqual(m.move_layer_to_device(m._load_streamed_layer(4)), 11)
        self.assertEqual(m._load_streamed_layer(4), 8)
        self.assertEqual(m._load_streamed_layer, original)
        snapshot = probe.snapshot()
        self.assertEqual(snapshot["counts"]["disk_io_plus_decode"], 1)
        self.assertEqual(snapshot["counts"]["transfer_plus_setup"], 1)

    def test_probe_tracks_exceptions_and_restores(self):
        from ironasu.airllm_probe import AirLLMStageProbe
        class Broken:
            def load_layer_to_cpu(self, name):
                raise FileNotFoundError(name)
        m = Broken()
        with self.assertRaises(FileNotFoundError):
            with AirLLMStageProbe(m) as probe:
                m.load_layer_to_cpu("missing")
        self.assertEqual(probe.snapshot()["counts"]["disk_io_plus_decode"], 1)
        with self.assertRaises(FileNotFoundError):
            m.load_layer_to_cpu("missing")

class CollectorTests(unittest.TestCase):
    def test_source_collector_skips_hidden_vendor_and_symlinks(self):
        from ironasu.context_cli import collect_files
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'index.py').write_text('print(1)', encoding='utf-8')
            (root / '.env').write_text('SECRET=1', encoding='utf-8')
            (root / 'node_modules').mkdir()
            (root / 'node_modules' / 'bundle.js').write_text('secret', encoding='utf-8')
            try:
                (root / 'shortcut.py').symlink_to(root / 'index.py')
            except (OSError, NotImplementedError):
                pass
            self.assertEqual(collect_files(root), [{'path': 'index.py', 'content': 'print(1)'}])
