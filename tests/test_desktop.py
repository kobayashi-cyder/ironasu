"""Desktop stream parsing, reasoning and zero-browser launcher regression tests."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from ironasu.desktop import desktop_parser, desktop_self_test
from ironasu.desktop_logic import (
    GenerationStats, SSEEvents, extract_reasoning, extract_text,
    normalized_model_selection,
)


class TestNativeDesktop(unittest.TestCase):
    def test_native_launcher_accepts_existing_provider_flags(self):
        args = desktop_parser().parse_args(["--provider", "lmstudio"])
        self.assertEqual(args.provider, "lmstudio")
        self.assertIsNone(args.self_test)

    def test_self_test_is_offline_no_window(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "native.json"
            desktop_self_test(str(target))
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "native-tk-desktop")
            self.assertFalse(report["browser_required"])
            self.assertFalse(report["http_server_required"])
            self.assertIn("tkinter_available", report)

    def test_split_utf8_and_crlf_stream_events(self):
        parser = SSEEvents()
        message = 'data: {"choices":[{"delta":{"content":"日本語"}}]}\r\n\r\n'
        raw = message.encode("utf-8")
        events = []
        for b in raw:
            events.extend(parser.feed(bytes([b])))
        self.assertEqual(len(events), 1)
        self.assertEqual(extract_text(events[0]["choices"][0]["delta"]), "日本語")
        self.assertEqual(parser.feed(b"data: [DONE]\n\n"), [])
        self.assertTrue(parser.done)

    def test_reasoning_fields_and_think_tag_split(self):
        stats = GenerationStats(started=1)
        first = {
            "choices": [{"delta": {"reasoning_content": "step a"}}]
        }
        self.assertEqual(stats.ingest(first, 1.2), [("reasoning", "step a")])
        stats.ingest({"choices": [{"delta": {"content": "<thi"}}]}, 1.3)
        stats.ingest({"choices": [{"delta": {"content": "nk>step b</th"}}]}, 1.4)
        stats.ingest({"choices": [{"delta": {"content": "ink>Final"}}]}, 1.6)
        stats.ingest({"usage": {"completion_tokens": 20}}, 2)
        result = stats.finish(2.2)
        self.assertEqual(stats.reasoning, "step astep b")
        self.assertEqual(stats.answer, "Final")
        self.assertAlmostEqual(result["ttft_s"], 0.2)
        self.assertAlmostEqual(result["reasoning_s"], 0.4)
        self.assertAlmostEqual(result["tokens_per_s"], 20.0)
        self.assertEqual(extract_reasoning({"reasoning_details":[{"text":"hi"}]}), "hi")

    def test_no_token_rate_guess_without_usage(self):
        stats = GenerationStats(started=1)
        stats.append_content("Hey", 1.5)
        result = stats.finish(2)
        self.assertIsNone(result["tokens_per_s"])
        self.assertEqual(result["chars_per_s"], 3)
        self.assertIsNone(result["reasoning_s"])

    def test_alias_selector(self):
        self.assertEqual(normalized_model_selection("quality"), ("quality", None))
        self.assertEqual(normalized_model_selection("model-id"), ("auto", "model-id"))
        with self.assertRaises(ValueError):
            normalized_model_selection("bad\nalias")

    def test_module_does_not_start_browser_or_webserver(self):
        import inspect
        from ironasu import desktop
        source = inspect.getsource(desktop.main)
        self.assertNotIn("serve(", source)
        self.assertNotIn("webbrowser.open", source)


if __name__ == "__main__":
    unittest.main()
