"""Managed GGUF runner and opt-in LM Studio autoload regression coverage."""
from __future__ import annotations

from hashlib import sha256
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ironasu.core import AdaptiveEngine, Config, HTTPBackend, InferenceError
from ironasu.native_models import NativeModelManager
from ironasu.runtime import ManagedGGUFRuntime, RuntimeBackendMux, PREFIX
from ironasu.server import make_handler


class FakeProcess:
    def __init__(self):
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.terminated = True
        self.code = -9


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.native = NativeModelManager(root=Path(self.tmp.name) / "models")
        self.content = b"GGUF" + b"\x00" * 64
        self.file_id = "abcdef1234567890abcdefff"
        self.digest = sha256(self.content).hexdigest()
        model = self.native.root / "owner" / "repo" / ("a" * 40) / "small.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(self.content)
        self.native._save_index({
            self.file_id: {
                "repo": "owner/repo", "path": "small.gguf",
                "commit": "a" * 40, "size": len(self.content),
                "digest": self.digest,
            },
        })
        self.runtime = ManagedGGUFRuntime(self.native, startup_timeout_s=1.5)
        self.addCleanup(self.runtime.stop)

    def wait_state(self, state):
        until = time.monotonic() + 2.0
        while time.monotonic() < until:
            report = self.runtime.status()
            if report["state"] == state:
                return report
            time.sleep(0.015)
        self.fail("managed runtime did not reach " + state + ": " + repr(report))

    def test_verified_local_runtime_spawns_loopback_auth_and_routes_model(self):
        process = FakeProcess()
        commands = []
        alias = PREFIX + self.file_id
        with patch.object(self.runtime, "_binary", return_value=str(Path(__file__).resolve())), \
             patch("ironasu.runtime.subprocess.Popen",
                   side_effect=lambda argv, **opts: commands.append((argv, opts)) or process), \
             patch.object(HTTPBackend, "list_models", return_value=[alias]), \
             patch.object(HTTPBackend, "complete", return_value={"choices": [
                 {"message": {"role": "assistant", "content": "native response"}}
             ]}):
            result = self.runtime.start(self.file_id, context_length=4096, gpu_layers=24)
            self.assertEqual(result["state"], "starting")
            ready = self.wait_state("running")
            self.assertEqual(ready["model_id"], alias)
            argv, opts = commands[0]
            self.assertIn("--model", argv)
            self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
            self.assertEqual(argv[argv.index("--alias") + 1], alias)
            self.assertEqual(argv[argv.index("--n-gpu-layers") + 1], "24")
            self.assertGreater(len(argv[argv.index("--api-key") + 1]), 20)
            self.assertNotIn("shell", opts)
            self.assertIn("--no-webui", argv)
            self.assertIsNotNone(self.runtime.backend_for(alias))
            outer = type("OtherBackend", (), {
                "list_models": lambda self: ["remote-model"],
                "endpoint": lambda self: {"provider": "lmstudio", "url": "http://127.0.0.1:1234/v1"},
                "complete": lambda self, payload: {"source": "remote", "model": payload["model"]},
            })()
            mux = RuntimeBackendMux(outer, self.runtime)
            self.assertEqual(mux.list_models(), [alias, "remote-model"])
            self.assertEqual(mux.complete({"model": alias})["choices"][0]["message"]["content"],
                             "native response")
            self.assertEqual(mux.complete({"model": "remote-model"})["source"], "remote")
            self.assertEqual(mux.endpoint()["provider"], "managed-gguf")
            self.assertEqual(self.runtime.stop()["state"], "idle")
            self.assertTrue(process.terminated)
            with self.assertRaises(InferenceError):
                mux.complete({"model": alias})

    def test_digest_failure_rejects_start_before_process(self):
        model = Path(self.native.installed()[0]["path"])
        model.write_bytes(b"BAD!" + self.content[4:])
        with patch.object(self.runtime, "_binary", return_value=str(Path(__file__).resolve())), \
             patch("ironasu.runtime.subprocess.Popen") as popen:
            self.runtime.start(self.file_id)
            state = self.wait_state("failed")
            self.assertIn("SHA-256", state["detail"])
            popen.assert_not_called()

    def test_missing_executable_fails_without_mutating_state(self):
        with patch("ironasu.runtime.shutil.which", return_value=None), \
             patch.dict("os.environ", {"IRONASU_LLAMA_SERVER": ""}):
            with self.assertRaises(InferenceError) as error:
                self.runtime.start(self.file_id)
            self.assertEqual(error.exception.status, 503)
            self.assertEqual(self.runtime.status()["state"], "idle")

    def test_invalid_launch_limits(self):
        for settings in ({"context_length": 0}, {"context_length": 200000},
                         {"gpu_layers": -1}, {"gpu_layers": 1000},
                         {"gpu_layers": "24"}):
            with self.subTest(settings=settings), self.assertRaises(InferenceError):
                self.runtime.start(self.file_id, **settings)


class AutoLoadTests(unittest.TestCase):
    def test_opt_in_autoload_runs_before_completion_and_only_for_local_lmstudio(self):
        calls = []
        class Backend:
            def endpoint(self):
                return {"provider": "lmstudio", "url": "http://127.0.0.1:1234/v1"}
            def list_models(self):
                return ["my-model"]
            def complete(self, payload):
                calls.append(("complete", payload["model"]))
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        class Manager:
            def ensure_loaded(self, provider, model):
                calls.append(("load", provider, model))
                return {"status": "loaded"}

        engine = AdaptiveEngine(Config(), Backend())
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, models=Manager()))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_port)
        def post(body):
            request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
            return urlopen(request, timeout=4)
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "x_ironasu_model": "my-model", "x_ironasu_autoload": True,
        }
        with post(payload) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(calls, [("load", "lmstudio", "my-model"),
                                ("complete", "my-model")])
        calls.clear()
        with post({**payload, "x_ironasu_autoload": False}):
            pass
        self.assertEqual(calls, [("complete", "my-model")])
        with self.assertRaises(HTTPError) as exc:
            post({**payload, "x_ironasu_model": "ironasu-local-other"})
        self.assertEqual(exc.exception.code, 409)
        self.assertNotIn(("load", "lmstudio", "ironasu-local-other"), calls)


if __name__ == "__main__":
    unittest.main()
