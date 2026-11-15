"""OpenAI-compatible gateway with a same-origin, dependency-free Studio UI."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlsplit
import hmac
import json
import threading
import webbrowser

from .core import AdaptiveEngine, Config, InferenceError
from .models import ModelManager
from .native_models import NativeModelManager
from .runtime import ManagedGGUFRuntime, RuntimeBackendMux
from .core import HTTPBackend

MAX_BODY_BYTES = 2 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/": ("index.html", "text/html; charset=utf-8"),
    "/ui/ui.css": ("ui.css", "text/css; charset=utf-8"),
    "/ui/ui.js": ("ui.js", "text/javascript; charset=utf-8"),
}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'none'"
)


def make_handler(engine: AdaptiveEngine, models: ModelManager | None = None,
                 native: NativeModelManager | None = None,
                 managed: ManagedGGUFRuntime | None = None):
    """A handler factory so HTTP integration tests can use fake backends."""
    settings = engine.config
    models = models or ModelManager(settings)
    native = native or NativeModelManager()

    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(data)

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(code, data, "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            expected = settings.server_key
            if not expected:
                return True
            token = self.headers.get("Authorization", "")
            return hmac.compare_digest(token, "Bearer " + expected)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            if path in ASSETS:
                name, content_type = ASSETS[path]
                data = resources.files("ironasu.web").joinpath(name).read_bytes()
                self._send_bytes(200, data, content_type)
                return
            if not self._authorized():
                self._send(401, {"error": {"message": "unauthorized"}})
            elif path == "/health":
                self._send(200, {"status": "ok"})
            elif path == "/metrics":
                self._send(200, engine.diagnostic_snapshot())
            elif path == "/v1/models":
                self._send(200, {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "owned_by": "ironasu"}
                        for model in ("ironasu", "ironasu-fast", "ironasu-quality")
                    ],
                })
            elif path == "/ui/models/downloads":
                self._send(200, {"jobs": native.jobs() + models.list_jobs()})
            elif path == "/ui/runtime/status":
                self._send(200, managed.status() if managed is not None else
                           {"state": "unavailable", "detail": "Managed GGUF runner is unavailable",
                            "runtime_found": False, "model_id": None})
            elif path in ("/ui/models/installed", "/ui/models/native/files",
                          "/ui/models/native/search"):
                try:
                    qs = parse_qs(parsed.query)
                    if path == "/ui/models/native/files":
                        self._send(200, {"files": native.files(qs.get("repo", [""])[0])})
                    elif path == "/ui/models/native/search":
                        self._send(200, {"models": native.search(qs.get("q", [""])[0])})
                    else:
                        provider = qs.get("provider", [""])[0]
                        self._send(200, {"models": native.installed() if provider == "native"
                                         else models.installed(provider)})
                except InferenceError as exc:
                    self._send(exc.status, {"error": {"message": str(exc)}})
            elif path == "/ui/status":
                discovered = engine.upstream_models()
                fast, quality = engine.effective_models(discovered)
                endpoint = getattr(engine.backend, "endpoint", None)
                origin = endpoint() if callable(endpoint) else {
                    "provider": "custom", "url": settings.backend_url,
                }
                self._send(200, {
                    "upstream_available": discovered is not None,
                    "upstream_models": discovered or [],
                    "fast_model": fast,
                    "quality_model": quality,
                    "provider": origin["provider"],
                    "backend_url": origin["url"],
                    "metrics": engine.diagnostic_snapshot(),
                })
            else:
                self._send(404, {"error": {"message": "not found"}})

        def _read_json(self, *, cap: int = MAX_BODY_BYTES) -> dict:
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise InferenceError("invalid Content-Length", 400) from exc
            if size < 0 or size > cap:
                raise InferenceError("invalid or oversized request", 413)
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                raise InferenceError("Content-Type must be application/json", 415)
            try:
                data = json.loads(self.rfile.read(size))
            except (ValueError, UnicodeDecodeError) as exc:
                raise InferenceError("invalid JSON request", 400) from exc
            if not isinstance(data, dict):
                raise InferenceError("request must be an object", 400)
            return data

        def _same_origin(self) -> bool:
            if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
                return False
            origin = self.headers.get("Origin")
            if not origin:
                return True  # Local CLI callers generally omit Origin.
            parsed = urlsplit(origin)
            return (parsed.scheme in ("http", "https")
                    and parsed.netloc == self.headers.get("Host", "")
                    and parsed.path in ("", "/"))

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": {"message": "unauthorized"}})
                return
            path = urlsplit(self.path).path
            if path not in ("/v1/chat/completions", "/ui/models/download",
                             "/ui/models/load", "/ui/models/native/cancel",
                             "/ui/models/native/remove", "/ui/runtime/start", "/ui/runtime/stop"):
                self._send(404, {"error": {"message": "not found"}})
                return
            if not self._same_origin():
                self._send(403, {"error": {"message": "cross-site write denied"}})
                return
            try:
                request = self._read_json(
                    cap=8192 if path.startswith(("/ui/models/", "/ui/runtime/")) else MAX_BODY_BYTES
                )
                if path == "/ui/runtime/start":
                    if managed is None:
                        raise InferenceError("Managed runtime unavailable", 501)
                    result = managed.start(
                        request.get("id"),
                        context_length=request.get("context_length", 4096),
                        gpu_layers=request.get("gpu_layers"),
                    )
                    self._send(202, result)
                    return
                if path == "/ui/runtime/stop":
                    if managed is None:
                        raise InferenceError("Managed runtime unavailable", 501)
                    self._send(200, managed.stop())
                    return
                if path == "/ui/models/download":
                    if request.get("provider") == "native":
                        result = native.start(request.get("model"), request.get("file"))
                    else:
                        result = models.start(
                            request.get("provider", ""), request.get("model"),
                            request.get("quantization"),
                        )
                    self._send(202, {"job": result})
                    return
                if path == "/ui/models/native/cancel":
                    self._send(200, {"job": native.cancel(request.get("id", ""))})
                    return
                if path == "/ui/models/native/remove":
                    if managed is not None:
                        running = managed.status()
                        if (running["state"] in ("running", "starting")
                                and running["installed_id"] == request.get("id")):
                            raise InferenceError("Stop the managed model before deleting its GGUF", 409)
                    self._send(200, native.remove(request.get("id", "")))
                    return
                if path == "/ui/models/load":
                    result = models.load(request.get("provider", ""), request.get("model"))
                    self._send(200, result)
                    return
                if request.get("x_ironasu_autoload") is True:
                    name = request.get("x_ironasu_model")
                    if (not isinstance(name, str) or not name
                            or name.startswith("ironasu-local-")):
                        raise InferenceError(
                            "LM Studio autoload requires an explicit LM Studio model ID", 409,
                        )
                    upstream = getattr(engine.backend, "upstream", engine.backend)
                    origin_fn = getattr(upstream, "endpoint", None)
                    endpoint = origin_fn() if callable(origin_fn) else {
                        "url": settings.backend_url,
                    }
                    url = urlsplit(endpoint.get("url") or "")
                    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost")
                            or url.port != 1234):
                        raise InferenceError(
                            "Autoload requires the active local LM Studio server on port 1234", 409,
                        )
                    models.ensure_loaded("lmstudio", name)
                if request.get("stream") is True:
                    with engine.stream(request) as response:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache, no-transform")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        try:
                            for line in response:
                                self.wfile.write(line)
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            # The browser stopped generation; close upstream promptly.
                            return
                else:
                    self._send(200, engine.complete(request))
            except InferenceError as exc:
                self._send(exc.status, {"error": {"message": str(exc)}})
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"error": {"message": "invalid JSON request"}})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self._send(502, {"error": {"message": "unexpected backend failure"}})

    return Handler


def serve(config: Config | None = None, *, backend=None, open_browser: bool = False) -> None:
    settings = config or Config.from_env()
    native = NativeModelManager()
    managed = ManagedGGUFRuntime(native)
    upstream = backend if backend is not None else HTTPBackend(settings)
    engine = AdaptiveEngine(settings, RuntimeBackendMux(upstream, managed))
    server = ThreadingHTTPServer(
        (settings.listen_host, settings.listen_port),
        make_handler(engine, native=native, managed=managed),
    )
    url = f"http://{settings.listen_host}:{server.server_port}/ui"
    print(f"ironasu ready at {url}", flush=True)
    if open_browser:
        # Schedule only after the listening socket exists; no extra dependencies.
        timer = threading.Timer(0.4, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        managed.stop()
        server.server_close()
