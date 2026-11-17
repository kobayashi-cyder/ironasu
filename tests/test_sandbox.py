"""Pure policy/snapshot tests plus mandatory real-container tests in sandbox CI."""
from dataclasses import replace
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Timer
import time
import unittest
from unittest.mock import patch

from ironasu.execution import CommandResult
from ironasu.harness import Harness, Settings, Tools
from ironasu.mission import MissionJournal, check_acceptance
from ironasu.sandbox import SandboxPolicy, Snapshot, run_sandbox

IMAGE = os.getenv("IRONASU_TEST_SANDBOX_IMAGE")
DUMMY = "sha256:" + "a" * 64


class SandboxPolicyTests(unittest.TestCase):
    def test_policy_rejects_mutable_images_and_invalid_limits(self):
        for kwargs in ({"image": "python:latest"}, {"image": DUMMY, "memory_mb": True},
                       {"image": DUMMY, "cpus": float("nan")},
                       {"image": DUMMY, "pids": 0}, {"image": DUMMY, "disk_mb": 32}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SandboxPolicy(**kwargs)

    def test_snapshot_excludes_private_and_generated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in (".env", "secret.key", "credentials.json", "normal.py"):
                (source / name).write_text("content")
            (source / ".ironasu").mkdir()
            (source / ".ironasu" / "session").write_text("private")
            Snapshot(10000).copy(source, root / "copy")
            self.assertEqual([p.name for p in (root / "copy").iterdir()], ["normal.py"])

    def test_snapshot_budget_and_symlinks_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            (root / "source" / "large").write_bytes(b"x" * 100)
            with self.assertRaisesRegex(ValueError, "budget"):
                Snapshot(10).copy(root / "source", root / "copy")

    def test_session_and_mission_policy_cannot_be_downgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Backend:
                def complete(self, messages, tools):
                    return {"content": "done"}
            settings = Settings(workspace=root, sandbox=SandboxPolicy(DUMMY))
            agent = Harness(settings, Backend())
            try:
                self.assertEqual(agent.run("secure", "first"), "done")
                settings.sandbox = None
                with self.assertRaisesRegex(ValueError, "policy changed"):
                    agent.run("secure", "second")
                self.assertEqual(len(agent.store.load("secure")["messages"]), 3)
                # Returning an already completed result executes no code.
                self.assertEqual(agent.run("secure", None, resume=True), "done")
            finally:
                agent.store.close()
            journal = MissionJournal(root)
            journal.bind_policy("mission", SandboxPolicy(DUMMY))
            with self.assertRaisesRegex(ValueError, "policy changed"):
                journal.bind_policy("mission", None)

    def test_requested_sandbox_never_runs_native_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = Harness(Settings(workspace=root, sandbox=SandboxPolicy(DUMMY),
                                     allow_tests=True))
            try:
                with patch("ironasu.harness.run_sandbox", side_effect=RuntimeError("no engine")), \
                     patch("ironasu.harness.run_command") as native:
                    with self.assertRaisesRegex(RuntimeError, "no engine"):
                        Tools(agent.settings, agent.store, "test").execute(
                            "run_checks", {"kind": "compileall"})
                    native.assert_not_called()
            finally:
                agent.store.close()


@unittest.skipUnless(IMAGE, "Real sandbox image is supplied by the sandbox-integration CI job")
class RealSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.policy = SandboxPolicy(IMAGE, memory_mb=256, disk_mb=48,
                                    snapshot_mb=8, pids=24, cpus=0.5)
        (self.root / "source.txt").write_text("original")
        (self.root / ".env").write_text("host secret")

    def command(self, code, **kwargs):
        return run_sandbox([sys.executable, "-c", code], cwd=self.root,
            timeout=kwargs.pop("timeout", 10), policy=kwargs.pop("policy", self.policy), **kwargs)

    def test_isolated_network_filesystem_identity_and_host_immutability(self):
        code = (
            "import os,socket; from pathlib import Path; "
            "assert os.getuid()==65534; "
            "assert not Path('/input/project/.env').exists(); "
            "assert not Path('/var/run/docker.sock').exists(); "
            "assert not [line for line in Path('/proc/net/route').read_text().splitlines()[1:] "
            "if line.split()[1]=='00000000']; "
            "Path('source.txt').write_text('changed in container'); "
            "\nfor p in ['/etc/ironasu-escape','/input/project/source.txt']:\n"
            " try: Path(p).write_text('bad'); raise AssertionError('write escaped')\n"
            " except OSError: pass\n"
            "assert not os.getenv('OPENAI_API_KEY'); print('isolated')")
        result = self.command(code)
        self.assertTrue(result.passed, result)
        self.assertIn("isolated", result.output)
        self.assertEqual((self.root / "source.txt").read_text(), "original")

    def test_total_memory_limit_is_enforced(self):
        result = self.command("x=bytearray(800*1024*1024); print(len(x))")
        self.assertFalse(result.passed, result)
        self.assertEqual(result.reason, "memory_limit", result)

    def test_total_writable_capacity_is_enforced(self):
        result = self.command(
            "import errno; f=open('fill','wb'); block=b'x'*1048576\n"
            "try:\n"
            " for _ in range(100): f.write(block); f.flush()\n"
            " raise AssertionError('disk budget not enforced')\n"
            "except OSError as exc:\n assert exc.errno==errno.ENOSPC,exc\n print('disk limited')")
        self.assertTrue(result.passed, result)
        self.assertIn("disk limited", result.output)

    def test_process_count_is_bounded(self):
        result = self.command(
            "import subprocess,sys\nchildren=[]\n"
            "try:\n"
            " for _ in range(60): children.append(subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']))\n"
            " raise AssertionError('pid limit not enforced')\n"
            "except OSError:\n assert len(children)<24; print('pids limited')\n"
            "finally:\n"
            " for child in children: child.terminate()\n"
            " for child in children: child.wait()\n")
        self.assertTrue(result.passed, result)
        self.assertIn("pids limited", result.output)

    def test_cancellation_removes_the_daemon_owned_container(self):
        stop = Event()
        # Allow image checks/snapshot/setup to finish before cancellation.
        timer = Timer(2, stop.set)
        timer.start()
        self.addCleanup(timer.join)
        before = time.monotonic()
        result = self.command("import time; time.sleep(30)", stop=stop, timeout=20)
        self.assertEqual(result.reason, "cancelled", result)
        self.assertLess(time.monotonic() - before, 8)
        output = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=ironasu-"], text=True)
        self.assertEqual(output.strip(), "")

    def test_output_flood_and_detached_descendants_cannot_outlive_container(self):
        result = self.command(
            "import os,subprocess,sys; subprocess.Popen([sys.executable,'-c',"
            "'import os,time; os.setsid(); time.sleep(30)']);\n"
            "while True: os.write(1,b'x'*8192)", max_output_bytes=4096)
        self.assertEqual(result.reason, "output_limit", result)
        self.assertEqual(result.output_bytes, 4096)
        output = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=ironasu-"], text=True)
        self.assertEqual(output.strip(), "")

    def test_harness_and_independent_verifier_use_the_same_sandbox(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_sample.py").write_text(
            "import unittest,os\nclass Sample(unittest.TestCase):\n"
            " def test_uid(self): self.assertEqual(os.getuid(),65534)\n")
        agent = Harness(Settings(workspace=self.root, sandbox=self.policy, allow_tests=True))
        try:
            result = Tools(agent.settings, agent.store, "test").execute(
                "run_checks", {"kind": "unittest"})
            self.assertIn("exit=0 reason=exited", result)
        finally:
            agent.store.close()
        verdict = check_acceptance(self.root, ("compileall",), (), allowed=True,
                                   stop=Event(), sandbox=self.policy)
        self.assertTrue(verdict["passed"], verdict)

    def test_sanitized_git_and_linked_worktree_checks(self):
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()
        git("init", "-q")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "Test")
        (self.root / ".gitignore").write_text(".ironasu/\n.env\n")
        git("add", "source.txt", ".gitignore")
        git("commit", "-qm", "base")
        worktree = self.root / ".ironasu" / "worktrees" / "candidate"
        git("worktree", "add", "--detach", str(worktree), "HEAD")
        (worktree / "source.txt").write_text("bad trailing whitespace \n")
        verdict = check_acceptance(worktree, ("git_diff_check",), (), allowed=True,
                                   stop=Event(), sandbox=self.policy)
        self.assertFalse(verdict["passed"], verdict)
        self.assertIn("trailing whitespace", verdict["checks"][0]["output"])
