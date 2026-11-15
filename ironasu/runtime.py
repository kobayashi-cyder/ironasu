"""Managed local GGUF inference through an installed llama.cpp server binary.

No model downloads, shell commands, or arbitrary process paths are accepted
from the browser. The binary is explicitly installed by the user and found on
PATH or via IRONASU_LLAMA_SERVER. The process always binds loopback and
authenticates with a random ephemeral key.
"""
from __future__ import annotations

import atexit
from dataclasses import replace
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

from .core import Config, HTTPBackend, InferenceError
from .native_models import NativeModelManager

PREFIX = "ironasu-local-"


class ManagedGGUFRuntime:
    def __init__(self, native: NativeModelManager, *, startup_timeout_s: float = 120.0):
        self.native = native
        self.startup_timeout_s = startup_timeout_s
        self._lock = threading.RLock()
        self._state = "idle"
        self._detail = "No managed model loaded"
        self._model_id: str | None = None
        self._local_id: str | None = None
        self._port: int | None = None
        self._process: subprocess.Popen | None = None
        self._backend: HTTPBackend | None = None
        self._cancel = threading.Event()
        atexit.register(self.stop)

    @staticmethod
    def _binary() -> str:
        explicit = os.environ.get("IRONASU_LLAMA_SERVER")
        executable = explicit if explicit else (shutil.which("llama-server")
                                                or shutil.which("llama-server.exe"))
        if not executable:
            raise InferenceError(
                "llama-server was not found. Install a compatible llama.cpp binary "
                "and set IRONASU_LLAMA_SERVER to its full path.", 503,
            )
        path = Path(executable).expanduser()
        if not path.is_file():
            raise InferenceError("Configured llama-server executable does not exist", 400)
        return str(path.resolve())

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._process and self._process.poll() is not None and self._state == "running":
                self._state, self._detail = "failed", "Managed llama-server exited"
                self._backend = None
            return {
                "state": self._state, "detail": self._detail,
                "model_id": self._model_id if self._state == "running" else None,
                "installed_id": self._local_id,
                "port": self._port if self._state == "running" else None,
                "runtime_found": bool(os.environ.get("IRONASU_LLAMA_SERVER")
                                      or shutil.which("llama-server")
                                      or shutil.which("llama-server.exe")),
            }

    def model_id(self) -> str | None:
        return self.status()["model_id"]

    def _resolved_model(self, installed_id: str) -> tuple[Path, str]:
        if not isinstance(installed_id, str) or len(installed_id) > 64:
            raise InferenceError("Invalid native model ID", 400)
        installed = next((item for item in self.native.installed()
                          if item["id"] == installed_id), None)
        if installed is None:
            raise InferenceError("Model is not in ironasu's verified GGUF index", 404)
        file = Path(installed["path"])
        if not file.is_file() or file.is_symlink():
            raise InferenceError("Installed GGUF is not a regular file", 400)
        if self.native._hash(file) != installed["sha256"]:
            raise InferenceError("GGUF SHA-256 does not match the installed index", 409)
        return file, PREFIX + installed_id

    def start(self, installed_id: str, *, context_length: int = 4096,
              gpu_layers: int | None = None) -> dict[str, Any]:
        if type(context_length) is not int or not 512 <= context_length <= 131072:
            raise InferenceError("context_length must be 512-131072", 400)
        if gpu_layers is not None and (type(gpu_layers) is not int or not 0 <= gpu_layers <= 999):
            raise InferenceError("gpu_layers must be 0-999 or unset (auto)", 400)
        # Check executable before changing state; model integrity is rechecked
        # inside the background worker so large files don't block the UI.
        executable = self._binary()
        if not isinstance(installed_id, str) or not (8 <= len(installed_id) <= 64):
            raise InferenceError("Invalid installed model ID", 400)
        with self._lock:
            if self._state in ("starting", "running", "stopping"):
                raise InferenceError("Stop the current managed model before starting another", 409)
            self._cancel = threading.Event()
            self._state = "starting"
            self._detail = "Verifying GGUF SHA-256"
            self._model_id = None
            self._local_id = installed_id
            self._port = None
            self._process = None
            self._backend = None
            cancel = self._cancel
        worker = threading.Thread(
            target=self._start_worker,
            args=(installed_id, context_length, gpu_layers, executable, cancel),
            daemon=True, name="ironasu-managed-gguf",
        )
        worker.start()
        return self.status()

    def _start_worker(self, installed_id: str, context_length: int,
                      gpu_layers: int | None, executable: str,
                      cancel: threading.Event) -> None:
        proc: subprocess.Popen | None = None
        try:
            model_file, alias = self._resolved_model(installed_id)
            if cancel.is_set():
                return
            port = self._free_port()
            key = secrets.token_urlsafe(32)
            config = replace(Config(), backend_url=f"http://127.0.0.1:{port}/v1",
                             backend_key=key, request_timeout_s=600)
            backend = HTTPBackend(config)
            cmd = [executable, "--model", str(model_file), "--alias", alias,
                   "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(context_length), "--api-key", key,
                   "--no-webui"]
            if gpu_layers is not None:
                cmd.extend(["--n-gpu-layers", str(gpu_layers)])
            # No shell, no inherited stdin and no child WebUI. Tokens and
            # private paths should not be emitted in browser-visible logs.
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                if cancel.is_set():
                    return
                self._process, self._port = proc, port
                self._detail = "Waiting for local llama-server readiness"
            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline and not cancel.is_set():
                if proc.poll() is not None:
                    raise InferenceError("llama-server exited before readiness", 502)
                try:
                    ids = backend.list_models(timeout_s=0.5)
                    if ids and alias in ids:
                        with self._lock:
                            if cancel.is_set():
                                return
                            self._backend, self._model_id = backend, alias
                            self._state, self._detail = "running", "GGUF model is ready"
                        return
                except Exception:
                    pass
                time.sleep(0.25)
            if not cancel.is_set():
                raise InferenceError("llama-server startup timed out; check model and GPU support", 504)
        except Exception as exc:
            with self._lock:
                if not cancel.is_set():
                    self._state = "failed"
                    self._detail = str(exc)[:250] if isinstance(exc, InferenceError) else (
                        "Managed GGUF startup failed: " + type(exc).__name__)
        finally:
            with self._lock:
                should_kill = self._state != "running" or cancel.is_set()
                if should_kill and self._process is proc:
                    self._process, self._backend = None, None
            if proc and should_kill:
                self._terminate(proc)

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._cancel.set()
            proc = self._process
            self._process, self._backend = None, None
            self._model_id, self._local_id, self._port = None, None, None
            self._state, self._detail = "idle", "Managed runtime stopped"
        if proc is not None:
            self._terminate(proc)
        return self.status()

    def backend_for(self, model: str) -> HTTPBackend | None:
        with self._lock:
            if self._model_id == model and self._state == "running":
                if self._process is None or self._process.poll() is not None:
                    raise InferenceError("Managed llama-server exited", 503)
                return self._backend
            return None


class RuntimeBackendMux:
    """Route native model aliases to managed llama-server; other IDs are
    delegated to the preexisting LM Studio/Ollama/backend configuration.
    """
    def __init__(self, upstream: Any, managed: ManagedGGUFRuntime):
        self.upstream = upstream
        self.managed = managed

    def list_models(self) -> list[str] | None:
        native = self.managed.model_id()
        remote_fn = getattr(self.upstream, "list_models", None)
        remote = remote_fn() if callable(remote_fn) else None
        if native:
            return [native] + [x for x in (remote or []) if x != native]
        return remote

    def endpoint(self) -> dict[str, Any]:
        state = self.managed.status()
        if state["state"] == "running":
            return {"provider": "managed-gguf",
                    "url": f"http://127.0.0.1:{state['port']}/v1"}
        fn = getattr(self.upstream, "endpoint", None)
        return fn() if callable(fn) else {"provider": "custom", "url": None}

    def _route(self, payload: dict[str, Any]):
        name = payload.get("model")
        if isinstance(name, str) and name.startswith(PREFIX):
            chosen = self.managed.backend_for(name)
            if chosen is None:
                raise InferenceError("Native GGUF runtime is not ready for this model", 503)
            return chosen
        return self.upstream

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._route(payload).complete(payload)

    def open_stream(self, payload: dict[str, Any]):
        return self._route(payload).open_stream(payload)
