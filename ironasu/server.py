"""OpenAI-compatible gateway with a same-origin, dependency-free Studio UI."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlsplit
import hmac
import json

from .core import AdaptiveEngine, Config, InferenceError

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


def make_handler(engine: AdaptiveEngine):
    """A handler factory so HTTP integration tests can use fake backends."""
    settings = engine.config

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
            path = urlsplit(self.path).path
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
            elif path == "/ui/status":
                discovered = engine.upstream_models()
                self._send(200, {
                    "upstream_available": discovered is not None,
                    "upstream_models": discovered or [],
                    "fast_model": settings.fast_model,
                    "quality_model": settings.quality_model,
                    "metrics": engine.diagnostic_snapshot(),
                })
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": {"message": "unauthorized"}})
                return
            if urlsplit(self.path).path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > MAX_BODY_BYTES:
                    raise InferenceError("invalid or oversized request", 413)
                request = json.loads(self.rfile.read(size))
                if isinstance(request, dict) and request.get("stream") is True:
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


def serve(config: Config | None = None) -> None:
    settings = config or Config.from_env()
    engine = AdaptiveEngine(settings)
    server = ThreadingHTTPServer((settings.listen_host, settings.listen_port), make_handler(engine))
    print(f"ironasu ready at http://{settings.listen_host}:{settings.listen_port}/ui", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
