"""Local-provider model downloads for ironasu Studio.

Downloads are delegated to the installed LM Studio or Ollama server, never
through arbitrary URLs or into ironasu's inference engine. Jobs are in-memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .core import Config, InferenceError


# Only local model-management servers. Browser requests cannot modify these.
PROVIDER_ORIGINS = {
    "lmstudio": "http://127.0.0.1:1234",
    "ollama": "http://127.0.0.1:11434",
}
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$")
_HF = re.compile(r"^https://huggingface\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$")
_QUANT = re.compile(r"^[A-Za-z0-9_]{1,24}$")
_JOB = re.compile(r"^job_[A-Za-z0-9_-]{1,100}$")


def _model_name(provider: str, model: Any, quantization: Any = None) -> tuple[str, str | None]:
    if provider not in PROVIDER_ORIGINS:
        raise InferenceError("Only LM Studio and Ollama support direct installation", 400)
    if not isinstance(model, str) or not model or len(model) > 160:
        raise InferenceError("A model ID is required (max 160 characters)", 400)
    if provider == "lmstudio" and _HF.fullmatch(model):
        pass
    elif not _MODEL.fullmatch(model) or ".." in model or "//" in model:
        raise InferenceError("Invalid model name. Use a catalog ID or Hugging Face model URL.", 400)
    if quantization is not None:
        if provider != "lmstudio" or not _HF.fullmatch(model):
            raise InferenceError("Quantization is supported only for LM Studio Hugging Face URLs", 400)
        if not isinstance(quantization, str) or not _QUANT.fullmatch(quantization):
            raise InferenceError("Invalid quantization identifier", 400)
    return model, quantization


@dataclass
class DownloadJob:
    id: str
    provider: str
    model: str
    quantization: str | None
    state: str = "pending"
    detail: str = "Preparing download"
    completed: int | None = None
    total: int | None = None
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "provider": self.provider,
            "model": self.model, "quantization": self.quantization,
            "state": self.state, "detail": self.detail,
            "completed_bytes": self.completed, "total_bytes": self.total,
            "created_at": self.created_at,
        }


class ModelManager:
    """Max two concurrent downloads. The browser polls a safe local status API."""

    def __init__(self, config: Config, *, origins: dict[str, str] | None = None,
                 poll_seconds: float = 1.0, max_download_seconds: float = 4 * 3600):
        self.config = config
        self.origins = dict(origins) if origins is not None else dict(PROVIDER_ORIGINS)
        # Overrides are intended for local test servers only, never from HTTP.
        self.poll_seconds = poll_seconds
        self.max_download_seconds = max_download_seconds
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(2)

    def _headers(self, provider: str) -> dict[str, str]:
        result = {"Content-Type": "application/json", "Accept": "application/json"}
        # Forward the configured credential only to the same origin as the
        # configured inference backend (never to an unrelated localhost service).
        if self.config.backend_key:
            from urllib.parse import urlsplit
            upstream = urlsplit(self.config.backend_url)
            manager = urlsplit(self.origins[provider])
            if (upstream.scheme, upstream.hostname, upstream.port) == (
                manager.scheme, manager.hostname, manager.port
            ):
                result["Authorization"] = "Bearer " + self.config.backend_key
        return result

    def _open(self, provider: str, path: str, payload: dict | None = None, *, timeout: float = 20):
        if provider not in self.origins or not path.startswith("/") or "//" in path:
            raise InferenceError("invalid local model service", 400)
        request = Request(
            self.origins[provider].rstrip("/") + path,
            headers=self._headers(provider),
            data=json.dumps(payload).encode() if payload is not None else None,
            method="POST" if payload is not None else "GET",
        )
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as exc:
            # Never echo response bodies: some provider errors contain internal
            # paths or sensitive service details.
            raise InferenceError(f"{provider} model service returned HTTP {exc.code}", 502) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InferenceError(
                f"{provider} is unavailable; start its local server first", 503
            ) from exc

    def _json(self, provider: str, path: str, payload: dict | None = None) -> dict:
        with self._open(provider, path, payload) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise InferenceError("model service response too large", 502)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise InferenceError("model service returned invalid JSON", 502) from exc
        if not isinstance(value, dict):
            raise InferenceError("model service returned unexpected data", 502)
        return value

    def installed(self, provider: str) -> list[dict[str, Any]]:
        if provider not in PROVIDER_ORIGINS:
            raise InferenceError("unsupported model provider", 400)
        path = "/api/v1/models" if provider == "lmstudio" else "/api/tags"
        data = self._json(provider, path)
        entries = data.get("models", [])
        if not isinstance(entries, list):
            raise InferenceError("invalid installed model list", 502)
        result: list[dict[str, Any]] = []
        for model in entries[:200]:
            if not isinstance(model, dict):
                continue
            ident = model.get("key") if provider == "lmstudio" else model.get("name")
            if not isinstance(ident, str):
                continue
            size = model.get("size_bytes") if provider == "lmstudio" else model.get("size")
            result.append({
                "id": ident,
                "label": str(model.get("display_name") or ident)[:200],
                "size_bytes": size if type(size) is int and size >= 0 else None,
                "loaded": bool(model.get("loaded_instances")) if provider == "lmstudio" else None,
            })
        return result

    def load(self, provider: str, model: Any) -> dict[str, Any]:
        if provider != "lmstudio":
            raise InferenceError("Explicit model loading is available only for LM Studio", 400)
        name, _ = _model_name(provider, model)
        result = self._json(provider, "/api/v1/models/load", {"model": name})
        return {"model": name, "status": result.get("status", "unknown")}

    def ensure_loaded(self, provider: str, model: Any) -> dict[str, Any]:
        """Explicit opt-in autoload before a chat request.

        Never load arbitrary model strings: verify that the ID is installed
        and belongs to the locally running LM Studio instance.
        """
        if provider != "lmstudio":
            raise InferenceError("Autoload requires the local LM Studio API", 400)
        name, _ = _model_name(provider, model)
        with self._load_lock:
            matches = [item for item in self.installed(provider) if item["id"] == name]
            if not matches:
                raise InferenceError("Selected model is not installed in LM Studio", 404)
            if matches[0]["loaded"]:
                return {"model": name, "status": "loaded", "already_loaded": True}
            result = self.load(provider, name)
            if result.get("status") != "loaded":
                raise InferenceError("LM Studio did not confirm model loading", 502)
            return result

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.public() for job in reversed(list(self._jobs.values()))]

    def start(self, provider: str, model: Any, quantization: Any = None) -> dict[str, Any]:
        name, quant = _model_name(provider, model, quantization)
        if not self._slots.acquire(blocking=False):
            raise InferenceError("Two downloads are already running; retry after one finishes", 429)
        with self._lock:
            for job in self._jobs.values():
                if (job.provider, job.model, job.quantization) == (provider, name, quant) and (
                    job.state in ("pending", "downloading")
                ):
                    self._slots.release()
                    return job.public()
            # Bound in-memory job history.
            for key in list(self._jobs):
                if len(self._jobs) < 40:
                    break
                if self._jobs[key].state not in ("pending", "downloading"):
                    del self._jobs[key]
            job = DownloadJob(uuid4().hex, provider, name, quant)
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), name="ironasu-model-download", daemon=True).start()
        return job.public()

    def _update(self, job: DownloadJob, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)

    def _run(self, job: DownloadJob) -> None:
        try:
            self._update(job, state="downloading", detail="Connecting to " + job.provider)
            if job.provider == "lmstudio":
                self._lmstudio(job)
            else:
                self._ollama(job)
        except Exception as exc:
            if isinstance(exc, InferenceError):
                message = str(exc)
            else:
                message = f"Download failed: {type(exc).__name__}"
            self._update(job, state="failed", detail=message[:300])
        finally:
            self._slots.release()

    def _lmstudio(self, job: DownloadJob) -> None:
        body: dict[str, str] = {"model": job.model}
        if job.quantization:
            body["quantization"] = job.quantization
        result = self._json("lmstudio", "/api/v1/models/download", body)
        state = result.get("status")
        if state in ("already_downloaded", "completed"):
            self._update(job, state="completed", detail="Already downloaded" if
                         state == "already_downloaded" else "Download completed")
            return
        job_id = result.get("job_id")
        if state not in ("downloading", "paused") or not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
            raise InferenceError("LM Studio did not return a recognized download job", 502)
        deadline = time.monotonic() + self.max_download_seconds
        while time.monotonic() < deadline:
            data = self._json("lmstudio", "/api/v1/models/download/status/" + job_id)
            state = data.get("status")
            total, done = data.get("total_size_bytes"), data.get("downloaded_bytes")
            if state == "completed":
                self._update(job, state="completed", detail="Download completed",
                             total=total if type(total) is int else None,
                             completed=done if type(done) is int else None)
                return
            if state in ("failed", "paused"):
                self._update(job, state=state, detail="LM Studio download " + state)
                return
            if state != "downloading":
                raise InferenceError("Unrecognized LM Studio download status", 502)
            self._update(
                job, detail="LM Studio is downloading",
                total=total if type(total) is int and total >= 0 else None,
                completed=done if type(done) is int and done >= 0 else None,
            )
            time.sleep(self.poll_seconds)
        raise InferenceError("Download status polling timed out", 504)

    def _ollama(self, job: DownloadJob) -> None:
        succeeded = False
        # Ollama manages the destination, digest validation, partial files,
        # and available disk space. Never request insecure registry TLS.
        with self._open("ollama", "/api/pull", {"model": job.model, "stream": True}, timeout=120) as response:
            for raw in response:
                if len(raw) > 128 * 1024:
                    raise InferenceError("Ollama progress event too large", 502)
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise InferenceError("Invalid Ollama progress event", 502) from exc
                if not isinstance(event, dict):
                    raise InferenceError("Invalid Ollama progress event", 502)
                if event.get("error"):
                    raise InferenceError("Ollama download error: " + str(event["error"])[:200], 502)
                status = str(event.get("status", ""))[:160]
                total, done = event.get("total"), event.get("completed")
                self._update(
                    job, detail=status or "Downloading with Ollama",
                    total=total if type(total) is int and total >= 0 else None,
                    completed=done if type(done) is int and done >= 0 else None,
                )
                if status == "success":
                    succeeded = True
                    break
        if not succeeded:
            raise InferenceError("Ollama closed connection before confirming success", 502)
        self._update(job, state="completed", detail="Model installed in Ollama")
