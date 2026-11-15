"""Native Hugging Face GGUF installation tests. Entirely loopback; no model download."""
from __future__ import annotations

from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ironasu.core import AdaptiveEngine, Config, InferenceError
from ironasu.native_models import (
    HuggingFaceCatalog, NativeModelManager, validated_file, validated_repo,
)
from ironasu.server import make_handler


MODEL_DATA = b"GGUF" + bytes(range(256)) * 24
BAD_DATA = b"BAD!" + MODEL_DATA[4:]
COMMIT = "a" * 40
REPO = "demo/tiny-GGUF"
FILENAME = "quantized/q4_k_m.gguf"
DIGEST = sha256(MODEL_DATA).hexdigest()


class HubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/models/" + REPO:
            self._json({
                "sha": COMMIT,
                "siblings": [
                    {"rfilename": FILENAME, "size": len(MODEL_DATA),
                     "lfs": {"sha256": DIGEST, "size": len(MODEL_DATA)}},
                    {"rfilename": "unsafe/../x.gguf", "size": 4,
                     "lfs": {"sha256": DIGEST, "size": 4}},
                    {"rfilename": "unverified.gguf", "size": 30},
                    {"rfilename": "config.json", "size": 128},
                ],
            })
        elif path == "/api/models":
            self._json([{"id": REPO, "downloads": 42}])
        elif path == "/" + REPO + "/resolve/" + COMMIT + "/" + FILENAME:
            content = BAD_DATA if getattr(self.server, "corrupt", False) else MODEL_DATA
            self.server.requests.append(self.headers.get("Range"))
            start = 0
            if self.headers.get("Range"):
                start = int(self.headers["Range"][6:-1])
                if self.server.ignore_range:
                    start = 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range",
                                 f"bytes {start}-{len(content)-1}/{len(content)}")
            self.send_header("Content-Length", str(len(content) - start))
            self.end_headers()
            self.wfile.write(content[start:])
        elif path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/admin")
            self.end_headers()
        else:
            self.send_error(404)


class TestNativeGGUF(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hub = ThreadingHTTPServer(("127.0.0.1", 0), HubHandler)
        self.hub.requests = []
        self.hub.corrupt = False
        self.hub.ignore_range = False
        self.hub_thread = threading.Thread(target=self.hub.serve_forever, daemon=True)
        self.hub_thread.start()
        self.addCleanup(self.hub_thread.join, 2)
        self.addCleanup(self.hub.server_close)
        self.addCleanup(self.hub.shutdown)
        self.origin = "http://127.0.0.1:" + str(self.hub.server_port)
        self.root = Path(self.temp.name) / "models"
        self.native = NativeModelManager(
            root=self.root, catalog=HuggingFaceCatalog(origin=self.origin, timeout=5)
        )

    def done(self, ident):
        until = time.monotonic() + 4
        while time.monotonic() < until:
            job = next(j for j in self.native.jobs() if j["id"] == ident)
            if job["state"] in ("completed", "failed", "cancelled"):
                return job
            time.sleep(0.01)
        self.fail("Native GGUF job timeout")

    def test_catalog_lists_only_sha256_verified_gguf(self):
        data = self.native.files(REPO)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["sha256"], DIGEST)
        self.assertEqual(data[0]["path"], FILENAME)
        self.assertEqual(self.native.search("tiny"), [{"id": REPO, "downloads": 42}])

    def test_local_download_installs_and_removes_without_any_llm_server(self):
        self.assertFalse(self.native.installed())
        job = self.native.start(REPO, FILENAME)
        result = self.done(job["id"])
        self.assertEqual(result["state"], "completed", result)
        installed = self.native.installed()
        self.assertEqual(len(installed), 1)
        self.assertEqual(Path(installed[0]["path"]).read_bytes(), MODEL_DATA)
        self.assertTrue(str(self.root) in installed[0]["path"])
        self.assertEqual(installed[0]["sha256"], DIGEST)
        self.assertTrue(self.native.remove(installed[0]["id"])["removed"])
        self.assertFalse(self.native.installed())

    def test_resume_partial_uses_range_and_hash_verification(self):
        item = self.native.catalog.files(REPO)[0]
        target = self.native._target(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(str(target) + ".part").write_bytes(MODEL_DATA[:128])
        job = self.native.start(REPO, FILENAME)
        self.assertEqual(self.done(job["id"])["state"], "completed")
        self.assertIn("bytes=128-", self.hub.requests)
        self.assertEqual(target.read_bytes(), MODEL_DATA)

    def test_ignored_range_restarts_without_duplicate_bytes(self):
        self.hub.ignore_range = True
        item = self.native.catalog.files(REPO)[0]
        target = self.native._target(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(str(target) + ".part").write_bytes(MODEL_DATA[:512])
        job = self.native.start(REPO, FILENAME)
        self.assertEqual(self.done(job["id"])["state"], "completed")
        self.assertEqual(target.read_bytes(), MODEL_DATA)

    def test_corrupt_download_cannot_be_installed(self):
        self.hub.corrupt = True
        job = self.native.start(REPO, FILENAME)
        failed = self.done(job["id"])
        self.assertEqual(failed["state"], "failed", failed)
        self.assertIn("SHA-256", failed["detail"])
        self.assertFalse(self.native.installed())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_rejects_unsafe_repositories_and_paths(self):
        for repo in ("../secret", "user/../model", "http://127.0.0.1:1234/foo",
                     "https://evil.example/user/model", "user/model/extra", "user//model"):
            with self.subTest(repo=repo), self.assertRaises(InferenceError):
                validated_repo(repo)
        for path in ("../x.gguf", "x/../../m.gguf", "x.gguf.exe", "/tmp/x.gguf",
                     "x\\foo.gguf", "foo/.secret.gguf"):
            with self.subTest(path=path), self.assertRaises(InferenceError):
                validated_file(path)
        with self.assertRaises(InferenceError):
            self.native.start(REPO, "unverified.gguf")

    def test_safe_redirect_denies_localhost_request(self):
        with self.assertRaises(InferenceError):
            self.native.catalog._open("/redirect")

    def test_http_routes_auth_csrf_lookup_download_and_delete(self):
        class Backend:
            def complete(self, payload):
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            def list_models(self):
                return []
        engine = AdaptiveEngine(Config(server_key="secret"), Backend())
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, native=self.native))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_port)

        def request(method, path, payload=None, *, key=True, origin=None):
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = "Bearer secret"
            if origin:
                headers["Origin"] = origin
            data = json.dumps(payload).encode() if payload is not None else None
            return urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=8)

        with self.assertRaises(HTTPError) as err:
            request("POST", "/ui/models/download", {
                "provider": "native", "model": REPO, "file": FILENAME,
            }, key=False)
        self.assertEqual(err.exception.code, 401)
        with self.assertRaises(HTTPError) as err:
            request("POST", "/ui/models/download", {
                "provider": "native", "model": REPO, "file": FILENAME,
            }, origin="https://evil.example")
        self.assertEqual(err.exception.code, 403)
        with request("GET", "/ui/models/native/files?repo=demo%2Ftiny-GGUF") as response:
            self.assertEqual(json.load(response)["files"][0]["path"], FILENAME)
        with request("GET", "/ui/models/native/search?q=tiny") as response:
            self.assertEqual(json.load(response)["models"][0]["id"], REPO)
        with request("POST", "/ui/models/download",
                     {"provider": "native", "model": REPO, "file": FILENAME}) as response:
            job = json.load(response)["job"]
        self.assertEqual(self.done(job["id"])["state"], "completed")
        with request("GET", "/ui/models/installed?provider=native") as response:
            found = json.load(response)["models"]
        self.assertEqual(found[0]["repo"], REPO)
        with request("POST", "/ui/models/native/remove", {"id": found[0]["id"]}) as response:
            self.assertTrue(json.load(response)["removed"])


if __name__ == "__main__":
    unittest.main()
