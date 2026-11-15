"""Tool-schema minimization tests; no inference backend required."""
from pathlib import Path
import tempfile
import unittest

from ironasu.harness import Harness, Settings, TOOLS, available_tools


class CaptureBackend:
    def __init__(self):
        self.tools = None

    def complete(self, messages, tools):
        self.tools = tools
        return {"content": "finished", "tool_calls": []}


class LeanHarnessToolsTests(unittest.TestCase):
    def test_readonly_defaults_hide_unavailable_schemas(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(workspace=Path(folder))
            backend = CaptureBackend()
            self.assertEqual(Harness(settings, backend).run("readonly", "Inspect"), "finished")
            names = {x["function"]["name"] for x in backend.tools}
            self.assertEqual(names, {"list_files", "read_file", "search_text", "remember", "recall"})
            self.assertLess(len(backend.tools), len(TOOLS))

    def test_writes_and_checks_remain_available_only_when_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(workspace=Path(folder), allow_writes=True, allow_tests=True)
            backend = CaptureBackend()
            self.assertEqual(Harness(settings, backend).run("permitted", "Inspect"), "finished")
            self.assertEqual(backend.tools, TOOLS)

    def test_writes_without_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            names = {x["function"]["name"] for x in available_tools(Settings(
                workspace=Path(folder), allow_writes=True))}
            self.assertIn("edit_file", names)
            self.assertIn("write_file", names)
            self.assertNotIn("run_checks", names)


if __name__ == "__main__":
    unittest.main()
