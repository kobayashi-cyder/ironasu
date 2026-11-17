"""Model request cancellation is exercised over real HTTP, including response-body stalls."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import BoundedSemaphore, Event, Thread
import time
import unittest

from ironasu.execution import ModelCancelled, model_slot, run_command
from ironasu.harness import Harness, ModelError, OpenAIBackend, Settings
import sys


class ModelWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.seen, self.release = Event(), Event()
        case = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.request = body
                if self.server.mode == "stall":
                    self.send_response(200)
                    self.send_header("Content-Length", "500000")
                    self.end_headers()
                    self.wfile.write(b"{")
                    self.wfile.flush()
                    case.seen.set()
                    case.release.wait(5)
                    return
                if self.server.mode == "large":
                    payload = b"x" * (1024 * 1024 + 1)
                elif self.server.mode == "invalid":
                    payload = b"[]"
                else:
                    payload = json.dumps({"choices": [{"message": {
                        "role": "assistant", "content": "worker completed"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.mode = "normal"
        thread = Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            self.release.set()
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=3)
        self.addCleanup(cleanup)
        self.settings = Settings(workspace=self.root, timeout_s=10,
            endpoint=f"http://127.0.0.1:{self.server.server_port}/v1")

    def test_real_completion_roundtrip(self):
        result = OpenAIBackend(self.settings).complete(
            [{"role": "user", "content": "task"}], [])
        self.assertEqual(result["content"], "worker completed")
        self.assertEqual(self.request["messages"][0]["content"], "task")

    def test_cancel_interrupts_stalled_response_body_and_releases_session(self):
        self.server.mode = "stall"
        stop = Event()
        def cancel():
            if self.seen.wait(5):
                stop.set()
        thread = Thread(target=cancel)
        thread.start()
        self.addCleanup(thread.join, 6)
        agent = Harness(self.settings)
        self.addCleanup(agent.store.close)
        before = time.monotonic()
        result = agent.run("request", "task", cancel=stop)
        self.assertTrue(self.seen.is_set())
        self.assertIn("PAUSED: cancelled", result)
        self.assertLess(time.monotonic() - before, 3)
        row = agent.store.load("request")
        self.assertEqual(row["pending"], [])
        self.assertEqual(len(row["messages"]), 2)
        self.server.mode = "normal"
        self.assertEqual(agent.run("request", None, resume=True), "worker completed")

    def test_invalid_and_oversized_body_fail_without_reply(self):
        for mode in ("invalid", "large"):
            self.server.mode = mode
            with self.subTest(mode=mode), self.assertRaises(ModelError):
                OpenAIBackend(self.settings).complete([], [])

    def test_queued_model_slot_can_be_cancelled(self):
        gate, stop = BoundedSemaphore(1), Event()
        gate.acquire()
        stop.set()
        before = time.monotonic()
        try:
            with self.assertRaises(ModelCancelled):
                with model_slot(gate, stop):
                    self.fail("Cancelled queue entry was dispatched")
            self.assertLess(time.monotonic() - before, 0.3)
        finally:
            gate.release()
        with model_slot(gate, None):
            pass

    def test_bounded_stdin_roundtrip(self):
        raw = b"x" * 100000
        result = run_command(
            [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
            cwd=self.root, timeout=3, input_bytes=raw)
        self.assertTrue(result.passed)
        self.assertEqual(result.output.strip(), "100000")
