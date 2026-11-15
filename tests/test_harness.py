import json
from pathlib import Path
import tempfile
import unittest

from ironasu.harness import Harness, Settings, Store, Tools, context_window


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(json.loads(json.dumps(messages)))
        return next(self.responses)


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        self.s = Settings(workspace=self.root)

    def test_default_denies_write_and_tests(self):
        t = Tools(self.s, Store(self.root), "x")
        self.assertIn("denied", t.execute("write_file", {"path": "x.py", "content": "print(1)"}))
        self.assertIn("denied", t.execute("run_checks", {"kind": "compileall"}))

    def test_write_read_and_cas(self):
        self.s.allow_writes = True
        t = Tools(self.s, Store(self.root), "x")
        self.assertIn("sha256=", t.execute("write_file", {"path": "src/test.py", "content": "old"}))
        self.assertIn("expected_sha256", t.execute("write_file", {"path": "src/test.py", "content": "new"}))
        read = json.loads(t.execute("read_file", {"path": "src/test.py"}))
        self.assertEqual(read["content"], "old")
        self.assertIn("wrote", t.execute("write_file", {"path": "src/test.py", "content": "new", "expected_sha256": read["sha256"]}))
        self.assertEqual((self.root / "src/test.py").read_text(), "new")

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "outside").symlink_to(outside, target_is_directory=True)
            self.s.allow_writes = True
            t = Tools(self.s, Store(self.root), "x")
            self.assertIn("escapes workspace", t.execute("write_file", {"path": "outside/out.py", "content": "x"}))

    def test_agent_calls_tool_then_finishes(self):
        backend = FakeBackend([
            {"content": None, "tool_calls": [call("id1", "list_files", {"path": "."})]},
            {"content": "Done", "tool_calls": []},
        ])
        agent = Harness(self.s, backend)
        self.assertEqual(agent.run("one", "Inspect files"), "Done")
        self.assertEqual(agent.store.load("one")["status"], "complete")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1][-1]["role"], "tool")
        self.assertEqual(agent.store.event("one", "id1")[0], "done")

    def test_persistent_resume(self):
        backend = FakeBackend([{"content": None, "tool_calls": [call("r1", "read_file", {"path": "src/nope"})]}])
        agent = Harness(Settings(workspace=self.root, max_steps=1), backend)
        self.assertIn("PAUSED", agent.run("two", "Inspect"))
        self.assertEqual(agent.store.load("two")["pending"][0]["id"], "r1")
        next_backend = FakeBackend([{"content": "Read attempted"}])
        resumed = Harness(Settings(workspace=self.root, max_steps=3), next_backend)
        self.assertEqual(resumed.run("two", None, resume=True), "Read attempted")

    def test_started_tool_not_replayed(self):
        c = call("uncertain1", "write_file", {"path": "x", "content": "text"})
        store = Store(self.root)
        store.save("three", self.root, "ready", [{"role": "system", "content": "s"},
            {"role": "user", "content": "u"}, {"role": "assistant", "content": None, "tool_calls": [c]}], [c])
        store.start("three", c)
        backend = FakeBackend([])
        agent = Harness(self.s, backend)
        self.assertIn("UNCERTAIN", agent.run("three", None, resume=True))
        self.assertEqual(backend.calls, [])
        store.resolve_uncertain("three", "uncertain1", "Verified: no file changed")
        self.assertEqual(store.load("three")["pending"], [])

    def test_notes_are_session_scoped(self):
        store = Store(self.root)
        store.remember("a", "decision", "Use sqlite")
        self.assertEqual(len(store.recall("a", "sqlite")), 1)
        self.assertEqual(store.recall("b", "sqlite"), [])

    def test_compaction_keeps_complete_call_group(self):
        history = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
        for i in range(20):
            history += [{"role": "assistant", "content": None, "tool_calls": [call(f"t{i}", "list_files", {"path": "."})]},
                        {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 500}]
        out = context_window(history, 4000)
        calls = {tc["id"] for m in out for tc in m.get("tool_calls", [])}
        results = {m["tool_call_id"] for m in out if m["role"] == "tool"}
        self.assertEqual(calls, results)
        self.assertEqual(out[0]["role"], "system")
        self.assertLess(len(json.dumps(out)), 4500)


if __name__ == "__main__":
    unittest.main()
