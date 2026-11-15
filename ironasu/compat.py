"""Local OpenAI-compatible backend presets and resilient loopback discovery.

No network lookup beyond the explicitly enumerated loopback addresses occurs in
auto mode. Explicit custom URLs are opt-in and may send private prompts remotely.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from .core import Config, HTTPBackend, InferenceError

# A shared 8080 port is intentionally described as a generic OpenAI backend.
PRESETS: dict[str, str] = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
    "localai": "http://127.0.0.1:8080/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "jan": "http://127.0.0.1:1337/v1",
}
DISCOVERY = ("lmstudio", "ollama", "llamacpp", "vllm", "jan")


def normalize_backend_url(value: str) -> str:
    """Accept an OpenAI API root, a server origin or a chat URL."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("backend URL must not be empty")
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or any(ord(c) < 32 for c in value)):
        raise ValueError("backend URL must be a plain http(s) URL without credentials/query/fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid backend port") from exc
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    elif path.endswith("/models"):
        path = path[:-len("/models")]
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class AutoHTTPBackend:
    """Periodically discover a local API; support starting it after the UI.

    Requests are never retried automatically to avoid replaying tool/LLM calls.
    This is local-only detection; the user must explicitly opt into remote URLs.
    """
    def __init__(self, config: Config, *, ttl_s: float = 5.0, probe_timeout_s: float = 0.6):
        self.config = config
        self.ttl_s = ttl_s
        self.probe_timeout_s = probe_timeout_s
        self._lock = threading.Lock()
        self._expires = 0.0
        self._active: tuple[str, HTTPBackend, list[str]] | None = None
        self._backends = [
            (name, HTTPBackend(replace(config, backend_url=PRESETS[name])))
            for name in DISCOVERY
        ]

    def _resolve(self) -> tuple[str, HTTPBackend, list[str]] | None:
        with self._lock:
            if time.monotonic() < self._expires:
                return self._active
            first_responding = None
            # Prefer a server that lists at least one usable model.
            for name, backend in self._backends:
                models = backend.list_models(timeout_s=self.probe_timeout_s)
                if models is None:
                    continue
                choice = (name, backend, models)
                if first_responding is None:
                    first_responding = choice
                if models:
                    first_responding = choice
                    break
            self._active = first_responding
            self._expires = time.monotonic() + self.ttl_s
            return self._active

    def list_models(self) -> list[str] | None:
        active = self._resolve()
        return list(active[2]) if active else None

    def endpoint(self) -> dict[str, str | None]:
        active = self._resolve()
        if active is None:
            return {"provider": None, "url": None}
        return {"provider": active[0], "url": active[1].config.backend_url}

    def _backend(self) -> HTTPBackend:
        active = self._resolve()
        if active is None:
            raise InferenceError(
                "No local LLM server detected. Start LM Studio, Ollama or another OpenAI-compatible server.",
                503,
            )
        return active[1]

    def complete(self, payload: dict):
        return self._backend().complete(payload)

    def open_stream(self, payload: dict):
        return self._backend().open_stream(payload)
