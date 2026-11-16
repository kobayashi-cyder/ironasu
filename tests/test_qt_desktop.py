"""Tests for the standalone Qt6 Studio without requiring Qt for core Python users."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec

from ironasu.qt_desktop import VERSION, parser, self_test


class QtDesktopTests(unittest.TestCase):
    def test_default_desktop_parser_accepts_existing_cli(self):
        args = parser().parse_args(["--provider", "lmstudio", "--port", "8766"])
        self.assertEqual(args.provider, "lmstudio")
        self.assertEqual(args.port, 8766)

    def test_self_test_never_needs_browser(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "desktop.json"
            self_test(str(out))
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["mode"], "native-qt6-desktop")
            self.assertFalse(result["browser_required"])
            self.assertFalse(result["http_server_required"])
            self.assertEqual(result["version"], VERSION)
            self.assertEqual(result["agents_max"], 16)
            self.assertEqual(result["corpus_partitions"], [4, 16])

    def test_no_qt_import_at_entrypoint(self):
        code = ("import ironasu.qt_desktop as q; "
                "print(q.parser().parse_args(['--provider','lmstudio']).provider)")
        process = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "lmstudio")

    @unittest.skipUnless(find_spec("PySide6"), "Qt optional extra is not installed")
    def test_qt_widgets_construct_without_network_or_browser(self):
        with tempfile.TemporaryDirectory() as folder:
            result_file = Path(folder) / "qt-smoke.json"
            env = dict(os.environ)
            if sys.platform != "win32" and not env.get("DISPLAY"):
                env["QT_QPA_PLATFORM"] = "offscreen"
            process = subprocess.run(
                [sys.executable, "-m", "ironasu.qt_desktop",
                 "--smoke-gui", str(result_file)],
                capture_output=True, text=True, timeout=60, env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertTrue(report["native_window_created"])
            self.assertTrue(report["qt6"])
            self.assertEqual(report["splitter_panels"], 3)
            self.assertEqual(len(report["tabs"]), 3)
            self.assertIn("エージェント", report["tabs"][2])
            self.assertTrue(report["model_management"])
            self.assertFalse(report["browser_required"])


if __name__ == "__main__":
    unittest.main()
