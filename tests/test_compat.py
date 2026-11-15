"""Cross-provider OpenAI compatibility and installable launcher regression tests."""
from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from ironasu.compat import AutoHTTPBackend, PRESETS, normalize_backend_url
from ironasu.core import AdaptiveEngine, Config, HTTPBackend, InferenceError
from ironasu.server import make_handler
from ironasu.studio import configure, main, parser


class MockOpenAIServer(BaseHTTPRequestHandler):
    models = ["offline-model"]

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        raw = json.dumps({"data": [{"id": model} for model in self.models]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        result = {
            "choices": [{"message": {"role": "assistant", "content": request["model"]}}],
        }
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


class CompatibilityTests(unittest.TestCase):
    def test_presets_are_loopback_only(self):
        for name, url in PRESETS.items():
            with self.subTest(name=name):
                self.assertTrue(url.startswith("http://127.0.0.1:"))
                self.assertTrue(url.endswith("/v1"))

    def test_openai_url_normalization(self):
        self.assertEqual(normalize_backend_url("http://localhost:1234"),
                         "http://localhost:1234/v1")
        self.assertEqual(normalize_backend_url("http://localhost:1234/v1/"),
                         "http://localhost:1234/v1")
        self.assertEqual(normalize_backend_url("http://localhost:1234/v1/chat/completions"),
                         "http://localhost:1234/v1")
        self.assertEqual(normalize_backend_url("http://localhost:1234/v1/models"),
                         "http://localhost:1234/v1")
        self.assertEqual(normalize_backend_url("https://example.net/api/v1"),
                         "https://example.net/api/v1")
        for invalid in ("", "/v1", "ftp://x/v1", "http://a:bad/v1",
                        "http://user:password@localhost/v1", "http://a/v1?k=x",
                        "http://a/v1#test", "http://\nlocalhost/v1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_backend_url(invalid)

    def test_launcher_respects_provider_and_settings(self):
        with patch.dict(os.environ, {
            "IRONASU_BACKEND_URL": "", "IRONASU_FAST_MODEL": "",
            "IRONASU_QUALITY_MODEL": "",
        }):
            args = parser().parse_args(["--provider", "ollama", "--no-browser", "--port", "8766"])
            config, backend = configure(args)
        self.assertIsInstance(backend, HTTPBackend)
        self.assertEqual(config.backend_url, PRESETS["ollama"])
        self.assertEqual(config.listen_port, 8766)
        self.assertTrue(config.auto_models)
        self.assertEqual(config.fast_model, "")
        self.assertEqual(config.quality_model, "")

    def test_auto_is_local_and_does_not_require_a_running_llm(self):
        with patch.dict(os.environ, {
            "IRONASU_BACKEND_URL": "", "IRONASU_FAST_MODEL": "",
            "IRONASU_QUALITY_MODEL": "",
        }):
            config, backend = configure(parser().parse_args(["--no-browser"]))
        self.assertIsInstance(backend, AutoHTTPBackend)
        self.assertIsNone(backend.endpoint()["provider"])
        self.assertFalse(backend.list_models())
        self.assertTrue(config.auto_models)

    def test_model_discovery_and_real_completion(self):
        mock_server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIServer)
        thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(mock_server.server_close)
        self.addCleanup(mock_server.shutdown)
        url = "http://127.0.0.1:" + str(mock_server.server_port) + "/v1"
        config = Config(backend_url=url, fast_model="", quality_model="", auto_models=True)
        backend = HTTPBackend(config)
        self.assertEqual(backend.list_models(), ["offline-model"])
        engine = AdaptiveEngine(config, backend)
        request = {"messages": [{"role": "user", "content": "Hello"}]}
        answer = engine.complete(request)
        self.assertEqual(answer["choices"][0]["message"]["content"], "offline-model")
        self.assertEqual(engine.effective_models(), ("offline-model", "offline-model"))
        answer = engine.complete({**request, "x_ironasu_mode": "fast"})
        self.assertEqual(answer["_ironasu"]["route"], "fast")
        gateway = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
        gw_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gw_thread.start()
        self.addCleanup(gw_thread.join, 2)
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        with urlopen("http://127.0.0.1:" + str(gateway.server_port) + "/ui/status") as response:
            status = json.load(response)
        self.assertEqual(status["quality_model"], "offline-model")
        self.assertEqual(status["backend_url"], url)

    def test_auto_reprobes_after_server_appears(self):
        local = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIServer)
        self.addCleanup(local.server_close)
        url = "http://127.0.0.1:" + str(local.server_port) + "/v1"
        config = Config(auto_models=True, fast_model="", quality_model="")
        auto = AutoHTTPBackend(config, ttl_s=0.02, probe_timeout_s=0.1)
        auto._backends = [("fake", HTTPBackend(replace(config, backend_url=url)))]
        self.assertIsNone(auto.list_models())
        thread = threading.Thread(target=local.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(local.shutdown)
        time.sleep(0.03)
        self.assertEqual(auto.list_models(), ["offline-model"])
        self.assertEqual(auto.endpoint()["provider"], "fake")
        self.assertEqual(
            AdaptiveEngine(config, auto).complete({
                "messages": [{"role": "user", "content": "hi"}]
            })["choices"][0]["message"]["content"], "offline-model")

    def test_empty_catalog_fails_closed_in_auto_mode(self):
        class Empty:
            def list_models(self):
                return []
            def complete(self, request):
                raise AssertionError("must not infer with an invented model")
        engine = AdaptiveEngine(
            Config(auto_models=True, fast_model="", quality_model=""), Empty(),
        )
        with self.assertRaises(InferenceError) as e:
            engine.complete({"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(e.exception.status, 503)


if __name__ == "__main__":
    unittest.main()
