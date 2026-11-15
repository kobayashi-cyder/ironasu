"""Minimal OpenAI-compatible chat proxy. All data is local unless you set a remote backend."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from .core import AdaptiveEngine, Config, InferenceError

MAX_BODY_BYTES = 2 * 1024 * 1024


def serve(config: Config | None = None) -> None:
    settings = config or Config.from_env()
    engine = AdaptiveEngine(settings)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            expected = settings.server_key
            if not expected:
                return True
            token = self.headers.get("Authorization", "")
            return hmac.compare_digest(token, "Bearer " + expected)

        def do_GET(self) -> None:
            if not self._authorized():
                self._send(401, {"error": {"message": "unauthorized"}})
            elif self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/metrics":
                self._send(200, engine.metrics.snapshot())
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": {"message": "unauthorized"}})
                return
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > MAX_BODY_BYTES:
                    raise InferenceError("invalid or oversized request", 413)
                request = json.loads(self.rfile.read(size))
                self._send(200, engine.complete(request))
            except InferenceError as exc:
                self._send(exc.status, {"error": {"message": str(exc)}})
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"error": {"message": "invalid JSON request"}})
            except Exception:
                self._send(502, {"error": {"message": "unexpected backend failure"}})

    server = ThreadingHTTPServer((settings.listen_host, settings.listen_port), Handler)
    print(f"ironasu ready on http://{settings.listen_host}:{settings.listen_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
