"""Local model install actions and validation without external model downloads."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ironasu.core import AdaptiveEngine, Config, InferenceError
from ironasu.models import ModelManager, _model_name
from ironasu.server import make_handler


class FakeModelService(BaseHTTPRequestHandler):
    calls = []
    progress_reads = 0

    def log_message(self, *_args):
        pass

    def _send(self, code, value):
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/v1/models":
            self._send(200, {"models": [
                {"key": "ibm/granite-4-micro", "display_name": "Granite Micro",
                 "loaded_instances": [], "size_bytes": 100},
            ]})
        elif self.path == "/api/tags":
            self._send(200, {"models": [
                {"name": "qwen3:4b", "size": 500},
            ]})
        elif self.path == "/api/v1/models/download/status/job_test123":
            self.__class__.progress_reads += 1
            status = "completed" if self.__class__.progress_reads >= 2 else "downloading"
            self._send(200, {"status": status, "job_id": "job_test123",
                             "total_size_bytes": 1000,
                             "downloaded_bytes": 1000 if status == "completed" else 100})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        data = json.loads(content)
        self.__class__.calls.append((self.path, data))
        if self.path == "/api/v1/models/download":
            self._send(200, {"status": "downloading", "job_id": "job_test123"})
        elif self.path == "/api/v1/models/load":
            self._send(200, {"status": "loaded", "instance_id": data["model"]})
        elif self.path == "/api/pull":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for event in (
                {"status": "pulling manifest"},
                {"status": "pulling layer", "total": 300, "completed": 100},
                {"status": "success"},
            ):
                self.wfile.write(json.dumps(event).encode() + b"\n")
                self.wfile.flush()
        else:
            self._send(404, {"error": "not found"})


class TestModelManager(unittest.TestCase):
    def setUp(self):
        FakeModelService.calls = []
        FakeModelService.progress_reads = 0
        self.service = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelService)
        self.thread = threading.Thread(target=self.service.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.service.server_close)
        self.addCleanup(self.service.shutdown)
        origin = "http://127.0.0.1:" + str(self.service.server_port)
        self.config = Config()
        self.manager = ModelManager(
            self.config, origins={"lmstudio": origin, "ollama": origin}, poll_seconds=0.01,
        )

    def wait_done(self, ident):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            job = next(j for j in self.manager.list_jobs() if j["id"] == ident)
            if job["state"] in ("completed", "failed", "paused"):
                return job
            time.sleep(0.01)
        self.fail("download job did not finish")

    def test_lmstudio_download_polls_and_completes(self):
        job = self.manager.start("lmstudio", "ibm/granite-4-micro")
        result = self.wait_done(job["id"])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["total_bytes"], 1000)
        self.assertEqual(result["completed_bytes"], 1000)
        self.assertIn(("/api/v1/models/download", {"model": "ibm/granite-4-micro"}),
                      FakeModelService.calls)
        self.assertGreaterEqual(FakeModelService.progress_reads, 2)

    def test_ollama_download_progress_and_completion(self):
        job = self.manager.start("ollama", "qwen3:4b")
        result = self.wait_done(job["id"])
        self.assertEqual(result["state"], "completed")
        self.assertIn(("/api/pull", {"model": "qwen3:4b", "stream": True}),
                      FakeModelService.calls)

    def test_models_can_be_listed_and_explicitly_loaded(self):
        self.assertEqual(self.manager.installed("lmstudio")[0]["id"], "ibm/granite-4-micro")
        self.assertEqual(self.manager.installed("ollama")[0]["id"], "qwen3:4b")
        self.assertEqual(self.manager.load("lmstudio", "ibm/granite-4-micro")["status"], "loaded")

    def test_rejects_url_forgery_invalid_ids_and_unsupported_provider(self):
        invalid = [
            ("unsupported", "any"),
            ("ollama", "http://127.0.0.1/admin"),
            ("lmstudio", "https://127.0.0.1/secrets"),
            ("lmstudio", "../private"),
            ("ollama", ""),
            ("lmstudio", "model\nAttack"),
        ]
        for provider, model in invalid:
            with self.subTest(provider=provider, model=model):
                with self.assertRaises(InferenceError):
                    self.manager.start(provider, model)
        with self.assertRaises(InferenceError):
            self.manager.start("ollama", "qwen3:4b", "Q4_K_M")
        with self.assertRaises(InferenceError):
            self.manager.start("lmstudio", "ibm/granite-4-micro", "Q4_K_M")
        self.assertEqual(
            _model_name("lmstudio", "https://huggingface.co/owner/model-GGUF", "Q4_K_M"),
            ("https://huggingface.co/owner/model-GGUF", "Q4_K_M"),
        )

    def test_model_manager_routes_are_authenticated_and_same_origin(self):
        class Dummy:
            def complete(self, request):
                return {"choices": [{"message": {"content": "test"}}]}
            def list_models(self):
                return []
        engine = AdaptiveEngine(Config(server_key="local-token"), Dummy())
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, self.manager))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_port)

        def post(path, obj, origin=None, auth=True):
            headers = {"Content-Type": "application/json"}
            if origin: headers["Origin"] = origin
            if auth: headers["Authorization"] = "Bearer local-token"
            req = Request(base + path, data=json.dumps(obj).encode(),
                          method="POST", headers=headers)
            return urlopen(req, timeout=3)

        with self.assertRaises(HTTPError) as x:
            post("/ui/models/download", {"provider": "ollama", "model": "qwen3:4b"}, auth=False)
        self.assertEqual(x.exception.code, 401)
        with self.assertRaises(HTTPError) as x:
            post("/ui/models/download", {"provider": "ollama", "model": "qwen3:4b"},
                 origin="https://evil.example")
        self.assertEqual(x.exception.code, 403)
        with post("/ui/models/download", {"provider": "ollama", "model": "qwen3:4b"}) as r:
            self.assertEqual(r.status, 202)
            self.assertIn("job", json.load(r))
        req = Request(base + "/ui/models/downloads",
                      headers={"Authorization": "Bearer local-token"})
        with urlopen(req, timeout=3) as r:
            self.assertIn("jobs", json.load(r))
        req = Request(base + "/ui/models/installed?provider=lmstudio",
                      headers={"Authorization": "Bearer local-token"})
        with urlopen(req, timeout=3) as r:
            self.assertEqual(json.load(r)["models"][0]["id"], "ibm/granite-4-micro")
        with post("/ui/models/load", {"provider": "lmstudio", "model": "ibm/granite-4-micro"}) as r:
            self.assertEqual(json.load(r)["status"], "loaded")


if __name__ == "__main__":
    unittest.main()
