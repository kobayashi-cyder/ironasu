"""Real subprocess checks: bounds, cancellation, descendants and OS lock recovery."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Timer
import time
import unittest
from unittest.mock import patch

from ironasu.execution import exclusive_lock, run_command
from ironasu.harness import Harness, ModelError, Settings, Store, Tools
from ironasu.mission import check_acceptance


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def command(self, code, **kwargs):
        return run_command([sys.executable, "-c", code], cwd=self.root,
                           timeout=kwargs.pop("timeout", 3), **kwargs)

    def test_exit_decode_and_secret_filter(self):
        result = self.command(
            "import os,sys; os.write(1,b'bad:\\xff'); "
            "print(os.getenv('OPENAI_API_KEY','absent')); sys.exit(7)",
            env={**os.environ, "OPENAI_API_KEY": "secret"})
        self.assertFalse(result.passed)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.reason, "exited")
        self.assertIn("absent", result.output)
        self.assertNotIn("secret", result.output)

    def test_output_flood_is_bounded_and_fails_closed(self):
        result = self.command(
            "import os;\nwhile True: os.write(1,b'x'*8192)", max_output_bytes=4096)
        self.assertEqual(result.reason, "output_limit")
        self.assertEqual(result.output_bytes, 4096)
        self.assertLess(len(result.output), 4200)
        self.assertFalse(result.passed)

    def test_fast_success_with_too_much_output_still_fails(self):
        result = self.command("print('x'*200000)", max_output_bytes=4096)
        self.assertEqual(result.reason, "output_limit")
        self.assertFalse(result.passed)

    def test_timeout_kills_descendant(self):
        marker = self.root / "child-survived"
        child = ("import time; from pathlib import Path; time.sleep(1); "
                 f"Path({str(marker)!r}).write_text('alive')")
        parent = ("import subprocess,sys,time; "
                  f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)")
        before = time.monotonic()
        result = self.command(parent, timeout=0.3)
        self.assertEqual(result.reason, "timeout")
        self.assertLess(time.monotonic() - before, 3)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_success_also_cleans_background_children(self):
        marker = self.root / "background-survived"
        child = ("import time; from pathlib import Path; time.sleep(1); "
                 f"Path({str(marker)!r}).write_text('alive')")
        result = self.command("import subprocess,sys; "
                              f"subprocess.Popen([sys.executable,'-c',{child!r}])")
        self.assertTrue(result.passed)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_cancel_active_and_pre_cancelled(self):
        stop = Event()
        timer = Timer(0.15, stop.set)
        timer.start()
        self.addCleanup(timer.join)
        result = self.command("import time; time.sleep(10)", stop=stop)
        self.assertEqual(result.reason, "cancelled")
        self.assertFalse(result.passed)
        self.assertIsNone(self.command("raise Exception('must not start')",
                                      stop=stop).returncode)

    def test_invalid_timeouts(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.command("pass", timeout=timeout)

    def test_lock_excludes_another_process_and_recovers_after_crash(self):
        lock = self.root / "lock"
        ready = self.root / "ready"
        source = ("import time; from pathlib import Path; "
                  "from ironasu.execution import exclusive_lock; "
                  f"lock=Path({str(lock)!r}); ready=Path({str(ready)!r}); "
                  "\nwith exclusive_lock(lock):\n ready.write_text('yes'); time.sleep(30)")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.Popen([sys.executable, "-c", source], **options)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with exclusive_lock(lock):
                    self.fail("Lock allowed two owners")
        finally:
            process.kill()
            process.wait(timeout=5)
        with exclusive_lock(lock):
            pass

    def test_long_lock_path(self):
        directory = self.root
        while len(str(directory)) < 280:
            directory /= "long-lock-directory"
        with exclusive_lock(directory / "session.lock"):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with exclusive_lock(directory / "session.lock"):
                    self.fail("Long path bypassed lock ownership")

    def test_parallel_journal_initialization(self):
        def worker(i):
            store = Store(self.root)
            try:
                store.remember(str(i), "key", "value")
                return store.recall(str(i), "value")
            finally:
                store.close()
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(worker, range(32)))
        self.assertTrue(all(r == [{"key": "key", "value": "value"}] for r in results))

    def test_verification_rejects_cancelled_command_even_if_exit_zero(self):
        from ironasu.execution import CommandResult
        with patch("ironasu.mission.run_command", return_value=
                   CommandResult(0, "partial", "cancelled", 7)):
            result = check_acceptance(self.root, ("compileall",), (),
                                      allowed=True, stop=Event())
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["reason"], "cancelled")


class HarnessIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def agent(self, backend):
        agent = Harness(Settings(workspace=self.root), backend)
        self.addCleanup(agent.store.close)
        return agent

    def test_concurrent_resume_and_manual_resolution_are_excluded(self):
        class Backend:
            def complete(backend, messages, tools):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    rival.run("same", "overwrite")
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    rival.store.resolve_uncertain("same", "call", "overwrite")
                self.assertEqual(rival.store.load("same")["messages"][1]["content"], "original")
                return {"content": "done"}
        first = self.agent(Backend())
        rival = self.agent(None)
        self.assertEqual(first.run("same", "original"), "done")
        self.assertEqual(rival.run("same", None, resume=True), "done")

    def test_cancel_preserves_pending_side_effect_for_resume(self):
        stop = Event()
        class Backend:
            def complete(backend, messages, tools):
                stop.set()
                return {"tool_calls": [{"id": "write", "function": {
                    "name": "write_file", "arguments": '{"path":"result","content":"x"}'}}]}
        agent = self.agent(Backend())
        self.assertIn("PAUSED: cancelled", agent.run("cancel", "task", cancel=stop))
        self.assertEqual(len(agent.store.load("cancel")["pending"]), 1)
        self.assertIsNone(agent.store.event("cancel", "write"))
        self.assertFalse((self.root / "result").exists())

    def test_malformed_response_never_changes_history(self):
        invalid = [[], {"content": {}}, {"tool_calls": {}},
                   {"tool_calls": False}, {"role": "user", "content": "bad"},
                   {"tool_calls": [{"id": "x"*257, "function": {"name": "read_file"}}]}]
        for i, reply in enumerate(invalid):
            with self.subTest(reply=reply):
                class Backend:
                    def complete(self, messages, tools):
                        return reply
                agent = self.agent(Backend())
                with self.assertRaises(ModelError):
                    agent.run(str(i), "task")
                self.assertEqual(len(agent.store.load(str(i))["messages"]), 2)

    def test_reused_tool_id_is_rejected_before_new_pending_is_saved(self):
        reply = {"tool_calls": [{"id": "reused", "function": {
            "name": "list_files", "arguments": '{"path":"."}'}}]}
        class Backend:
            def complete(self, messages, tools):
                return reply
        agent = self.agent(Backend())
        with self.assertRaisesRegex(ModelError, "already used"):
            agent.run("reuse", "task")
        self.assertEqual(agent.store.load("reuse")["pending"], [])
        self.assertEqual(agent.store.event("reuse", "reused")[0], "done")

    def test_session_lock_releases_on_model_error(self):
        class Failing:
            def complete(self, messages, tools):
                raise ModelError("offline")
        class Success:
            def complete(self, messages, tools):
                return {"content": "recovered"}
        agent = self.agent(Failing())
        with self.assertRaises(ModelError):
            agent.run("recover", "task")
        agent.backend = Success()
        self.assertEqual(agent.run("recover", None, resume=True), "recovered")

    def test_oversized_task_is_not_sent_to_model(self):
        agent = self.agent(None)
        with patch.object(agent.backend, "complete") as complete:
            with self.assertRaisesRegex(ModelError, "context budget"):
                agent.run("large", "x" * 50000)
        complete.assert_not_called()

    def test_private_alias_is_denied(self):
        private = self.root / ".env.secret"
        private.write_text("not-for-model")
        alias = self.root / "public.txt"
        try:
            alias.symlink_to(private)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Creating symlinks requires Windows developer mode")
            raise
        agent = self.agent(None)
        output = Tools(agent.settings, agent.store, "alias").execute(
            "read_file", {"path": "public.txt"})
        self.assertIn("Private", output)
        self.assertNotIn("not-for-model", output)
