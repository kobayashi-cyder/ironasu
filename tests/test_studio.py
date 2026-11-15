"""Offline Studio packaging, HTTP integration, routing and SSE tests."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from importlib import resources
from io import BytesIO
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ironasu.core import AdaptiveEngine, Config, InferenceError
from ironasu.server import make_handler


class LocalBackend:
    def __init__(self):
        self.requests = []
        self.streams = []

    def complete(self, request):
        self.requests.append(request)
        return {
            "id": "local", "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "offline ok"}}],
        }

    def list_models(self):
        return ["local-model", "small-model"]

    @contextmanager
    def open_stream(self, request):
        self.streams.append(request)
        data = (
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" offline"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        with BytesIO(data) as stream:
            yield stream


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.backend = LocalBackend()
        self.config = Config()
        self.engine = AdaptiveEngine(self.config, self.backend)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def request(self, path, payload=None, key=None):
        headers = {}
        if key: headers["Authorization"] = "Bearer " + key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        with urlopen(Request(self.base + path, data=data, headers=headers), timeout=5) as res:
            return res.status, dict(res.headers), res.read()

    def test_ui_is_served_without_model_or_external_assets(self):
        for path, mime in [
            ("/ui", "text/html"), ("/ui/", "text/html"), ("/", "text/html"),
            ("/ui/ui.css", "text/css"), ("/ui/ui.js", "text/javascript"),
        ]:
            status, headers, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertTrue(headers["Content-Type"].startswith(mime))
            self.assertGreater(len(body), 100)
            self.assertNotIn(b"https://", body)
            self.assertNotIn(b"http://", body)
        self.assertIn(b"connect-src 'self'", self.request("/ui")[1]["Content-Security-Policy"].encode())
        self.assertIn("index.html", resources.files("ironasu.web").joinpath("index.html").name)
        self.assertIn(b"fetch(", self.request("/ui/ui.js")[2])

    def test_models_and_local_backend_status(self):
        status, _, raw = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in json.loads(raw)["data"]],
                         ["ironasu", "ironasu-fast", "ironasu-quality"])
        _, _, raw = self.request("/ui/status")
        status = json.loads(raw)
        self.assertEqual(status["upstream_models"], ["local-model", "small-model"])
        self.assertEqual(status["fast_model"], self.config.fast_model)
        self.assertTrue(status["upstream_available"])
        self.assertIn("active_requests", status["metrics"])

    def test_non_streaming_and_direct_model_selection(self):
        self.request("/v1/chat/completions", {
            "model": "ironasu-fast", "messages": [{"role": "user", "content": "Hello"}]
        })
        self.assertEqual(self.backend.requests[-1]["model"], self.config.fast_model)
        self.request("/v1/chat/completions", {
            "model": "ironasu", "x_ironasu_model": "local-model",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        self.assertEqual(self.backend.requests[-1]["model"], "local-model")
        self.assertNotIn("x_ironasu_model", self.backend.requests[-1])

    def test_streaming_passes_sse_and_respects_route(self):
        code, headers, body = self.request("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "x_ironasu_mode": "quality",
        })
        self.assertEqual(code, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        self.assertIn(b'"content":"hello"', body)
        self.assertIn(b'"content":" offline"', body)
        self.assertIn(b"data: [DONE]", body)
        self.assertEqual(self.backend.streams[-1]["model"], self.config.quality_model)
        self.assertEqual(self.engine.metrics.snapshot()["active_requests"], 0)

    def test_invalid_stream_and_invalid_model_return_json_400(self):
        for payload in [
            {"messages": [{"role": "user", "content": "Hi"}],
             "stream": True, "x_ironasu_files": []},
            {"messages": [{"role": "user", "content": "Hi"}],
             "x_ironasu_model": ""},
        ]:
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/chat/completions", payload)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("error", json.loads(caught.exception.read()))

    def test_key_protects_api_but_ui_assets_still_load(self):
        protected = AdaptiveEngine(Config(server_key="secret"), self.backend)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(protected))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_port)
        with urlopen(base + "/ui", timeout=5) as res:
            self.assertEqual(res.status, 200)
        with self.assertRaises(HTTPError) as caught:
            urlopen(base + "/ui/status", timeout=5)
        self.assertEqual(caught.exception.code, 401)
        req = Request(base + "/ui/status", headers={"Authorization": "Bearer secret"})
        with urlopen(req, timeout=5) as res:
            self.assertEqual(res.status, 200)


class EngineStreamCase(unittest.TestCase):
    def test_stream_unsupported_backend_fails_closed(self):
        class NoStream:
            def complete(self, request):
                return {"choices": [{"message": {"content": "ok"}}]}
        engine = AdaptiveEngine(Config(), NoStream())
        with self.assertRaisesRegex(InferenceError, "does not support streaming"):
            with engine.stream({"stream": True, "messages": [{"role": "user", "content": "x"}]}):
                pass
        self.assertEqual(engine.metrics.snapshot()["active_requests"], 0)

    def test_offline_catalog_unavailable_is_not_fake_connectivity(self):
        class OfflineBackend:
            def list_models(self):
                return None
            def complete(self, request):
                raise InferenceError("unavailable")
        engine = AdaptiveEngine(Config(), OfflineBackend())
        self.assertIsNone(engine.upstream_models())


if __name__ == "__main__":
    unittest.main()
