"""Native public GGUF downloads from Hugging Face Hub.

Unlike the LM Studio/Ollama adapters, this feature downloads model bytes into
ironasu-owned storage. It uses the Python stdlib and is included in the frozen
Windows application. Downloading is NOT equivalent to executing GGUF inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from .core import InferenceError

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9_.+()/-]{1,220}$")
_MAX_JSON_BYTES = 3 * 1024 * 1024
_MAX_GGUF_BYTES = 200 * 1024**3
_CHUNK_SIZE = 1024 * 1024


def default_model_root() -> Path:
    """User-owned, persistent data; never the source checkout or Program Files."""
    override = os.environ.get("IRONASU_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "ironasu" / "models"
    return Path.home() / ".local" / "share" / "ironasu" / "models"


def validated_repo(repo: Any) -> str:
    if not isinstance(repo, str):
        raise InferenceError("Hugging Face repository ID is required", 400)
    repo = repo.strip()
    if repo.startswith("https://huggingface.co/"):
        repo = repo[len("https://huggingface.co/"):].strip("/")
    if not _REPO.fullmatch(repo) or any(segment in (".", "..") for segment in repo.split("/")):
        raise InferenceError("Use a public Hugging Face model ID such as owner/model-GGUF", 400)
    return repo


def validated_file(name: Any) -> str:
    if (not isinstance(name, str) or not name.lower().endswith(".gguf")
            or not _SAFE_FILE.fullmatch(name) or len(name) > 220
            or any(part in ("", ".", "..") or part.startswith(".") for part in name.split("/"))):
        raise InferenceError("Select a GGUF file from the verified repository file list", 400)
    return name


class SafeRedirects(HTTPRedirectHandler):
    """Forbid redirects to arbitrary hosts, plain HTTP and loopback addresses."""
    def __init__(self, test_origin: str | None = None):
        super().__init__()
        self.test_origin = test_origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if self.test_origin and newurl.startswith(self.test_origin + "/"):
            pass
        elif (parsed.scheme != "https" or parsed.username or parsed.password
              or not (parsed.hostname == "huggingface.co"
                      or parsed.hostname.endswith(".huggingface.co")
                      or parsed.hostname.endswith(".hf.co"))):
            raise InferenceError("Hugging Face redirect to an untrusted destination", 502)
        # Never forward authorization or cookies on redirects.
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected:
            redirected.headers.pop("Authorization", None)
            redirected.headers.pop("Cookie", None)
        return redirected


@dataclass(frozen=True)
class GGUFFile:
    repo: str
    path: str
    commit: str
    size: int
    digest: str

    def public(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size,
                "sha256": self.digest, "commit": self.commit}


class HuggingFaceCatalog:
    """Metadata and file streaming from fixed HF origin. No arbitrary URL input."""
    def __init__(self, *, origin: str = "https://huggingface.co", timeout: int = 25):
        # Custom origin is test injection; not exposed in HTTP or user settings.
        self.origin = origin.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(SafeRedirects(
            self.origin if self.origin.startswith("http://127.0.0.1:") else None
        ))

    def _open(self, path: str, *, headers: dict[str, str] | None = None):
        request = Request(self.origin + path, headers={
            "User-Agent": "ironasu-studio/1.4",
            "Accept-Encoding": "identity",
            **(headers or {}),
        })
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            code = exc.code
            if code in (401, 403, 404):
                raise InferenceError(
                    "Repository/file unavailable or gated. Native downloads support public GGUF only.",
                    404,
                ) from exc
            raise InferenceError("Hugging Face request failed (HTTP " + str(code) + ")", 502) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InferenceError("Hugging Face is unreachable; an Internet connection is required", 503) from exc

    def _json(self, path: str) -> Any:
        with self._open(path) as response:
            raw = response.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise InferenceError("Hugging Face catalog exceeds supported size", 502)
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise InferenceError("Invalid Hugging Face metadata", 502) from exc

    def search(self, term: str) -> list[dict[str, Any]]:
        if not isinstance(term, str) or not 2 <= len(term.strip()) <= 100:
            raise InferenceError("Search query must be 2-100 characters", 400)
        data = self._json("/api/models?" + urlencode({
            "search": term.strip(), "limit": 12, "sort": "downloads",
            "direction": -1,
        }))
        if not isinstance(data, list):
            raise InferenceError("Invalid model catalog response", 502)
        return [
            {"id": item["id"], "downloads": item.get("downloads", 0)}
            for item in data if isinstance(item, dict)
            and isinstance(item.get("id"), str) and _REPO.fullmatch(item["id"])
        ][:12]

    def files(self, repo: str) -> list[GGUFFile]:
        repo = validated_repo(repo)
        data = self._json("/api/models/" + quote(repo, safe="/") + "?blobs=true")
        if not isinstance(data, dict):
            raise InferenceError("Invalid repository metadata", 502)
        commit, siblings = data.get("sha"), data.get("siblings")
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise InferenceError("Repository revision cannot be pinned", 502)
        if not isinstance(siblings, list) or len(siblings) > 10000:
            raise InferenceError("Repository file list is invalid or too large", 502)
        results: list[GGUFFile] = []
        for item in siblings:
            if not isinstance(item, dict):
                continue
            path = item.get("rfilename")
            try:
                path = validated_file(path)
            except InferenceError:
                continue
            lfs = item.get("lfs") or {}
            size = lfs.get("size") if isinstance(lfs, dict) else None
            if type(size) is not int:
                size = item.get("size")
            digest = lfs.get("sha256") if isinstance(lfs, dict) else None
            # GGUF must be a known-sized, SHA256-addressed LFS artifact;
            # this is stronger than trusting a Content-Length on redirects.
            if (type(size) is int and 0 < size <= _MAX_GGUF_BYTES
                    and isinstance(digest, str) and _SHA.fullmatch(digest)):
                results.append(GGUFFile(repo, path, commit, size, digest))
        return results[:500]

    def download_stream(self, item: GGUFFile, offset: int):
        path = "/" + quote(item.repo, safe="/") + "/resolve/" + item.commit
        path += "/" + quote(item.path, safe="/") + "?download=true"
        headers = {"Range": "bytes=" + str(offset) + "-"} if offset else {}
        return self._open(path, headers=headers)


@dataclass
class NativeJob:
    id: str
    item: GGUFFile
    state: str = "pending"
    detail: str = "Preparing"
    completed: int = 0
    created_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "provider": "native",
            "model": self.item.repo + "/" + self.item.path,
            "repository": self.item.repo,
            "file": self.item.path,
            "state": self.state, "detail": self.detail,
            "completed_bytes": self.completed, "total_bytes": self.item.size,
            "created_at": self.created_at,
        }


class NativeModelManager:
    def __init__(self, *, root: Path | None = None, catalog: HuggingFaceCatalog | None = None):
        self.root = Path(root or default_model_root()).expanduser().resolve()
        self.catalog = catalog or HuggingFaceCatalog()
        self._slots = threading.BoundedSemaphore(2)
        self._lock = threading.RLock()
        self._jobs: dict[str, NativeJob] = {}
        self._running: set[str] = set()

    def search(self, term: str) -> list[dict[str, Any]]:
        return self.catalog.search(term)

    def files(self, repo: str) -> list[dict[str, Any]]:
        return [f.public() for f in self.catalog.files(repo)]

    def _target(self, item: GGUFFile) -> Path:
        # All parts come from narrowly validated canonical repo metadata.
        pieces = [*item.repo.split("/"), item.commit, *item.path.split("/")]
        target = self.root.joinpath(*pieces)
        if any(part in (".", "..") for part in pieces):
            raise InferenceError("Unsafe model path", 400)
        # Reject symlink redirects even if a local process placed one in the
        # storage tree between requests.
        node = self.root
        if self.root.exists() and self.root.is_symlink():
            raise InferenceError("Model storage root must not be a symlink", 400)
        for part in pieces:
            node = node / part
            if node.is_symlink():
                raise InferenceError("Model storage path contains a symlink", 400)
        if not target.resolve().is_relative_to(self.root):
            raise InferenceError("Model path escapes native storage", 400)
        return target

    def _index_path(self) -> Path:
        return self.root / "installed.json"

    def _read_index(self) -> dict[str, dict]:
        path = self._index_path()
        if not path.exists():
            return {}
        if path.is_symlink():
            raise InferenceError("Unsafe model index path", 400)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise InferenceError("Native model index is unreadable", 500) from exc
        if not isinstance(data, dict):
            raise InferenceError("Native model index is invalid", 500)
        return data

    def _save_index(self, index: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.root / ("index-" + uuid4().hex + ".tmp")
        try:
            with temp.open("x", encoding="utf-8") as out:
                json.dump(index, out, ensure_ascii=False, indent=2)
            os.replace(temp, self._index_path())
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _id(item: GGUFFile) -> str:
        return sha256((item.repo + "\n" + item.commit + "\n" + item.path).encode()).hexdigest()[:24]

    def installed(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for ident, info in self._read_index().items():
                if not isinstance(info, dict):
                    continue
                try:
                    item = GGUFFile(**{k: info[k] for k in ("repo", "path", "commit", "size", "digest")})
                    target = self._target(item)
                    if target.is_file() and target.stat().st_size == item.size:
                        result.append({
                            "id": ident, "label": item.path, "repo": item.repo,
                            "size_bytes": item.size, "path": str(target),
                            "sha256": item.digest, "loaded": None,
                        })
                except (KeyError, TypeError, ValueError, OSError, InferenceError):
                    continue
            return result

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.public() for job in reversed(list(self._jobs.values()))]

    def start(self, repo: str, filename: str) -> dict[str, Any]:
        repo, filename = validated_repo(repo), validated_file(filename)
        if not self._slots.acquire(blocking=False):
            raise InferenceError("Two native downloads are already running", 429)
        try:
            # Model information is resolved first and the SHA-256 determines
            # whether the selected file can be downloaded safely.
            available = self.catalog.files(repo)
            matches = [entry for entry in available if entry.path == filename]
            if not matches:
                raise InferenceError("GGUF file not found or lacks verifiable SHA-256 metadata", 400)
            item = matches[0]
            with self._lock:
                identity = self._id(item)
                if identity in self._running:
                    raise InferenceError("This GGUF file is already being downloaded", 409)
                self._running.add(identity)
                job = NativeJob("native_" + uuid4().hex, item)
                self._jobs[job.id] = job
                while len(self._jobs) > 40:
                    old = next((x for x, j in self._jobs.items() if j.state not in (
                        "pending", "downloading", "verifying")), None)
                    if old is None:
                        break
                    del self._jobs[old]
            thread = threading.Thread(target=self._run, args=(job, identity),
                                      daemon=True, name="ironasu-native-model")
            thread.start()
            return job.public()
        except Exception:
            self._slots.release()
            raise

    def cancel(self, ident: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(ident)
            if job is None:
                raise InferenceError("Unknown download job", 404)
            if job.state in ("pending", "downloading"):
                job.cancel.set()
            return job.public()

    def remove(self, ident: str) -> dict[str, Any]:
        with self._lock:
            index = self._read_index()
            info = index.get(ident)
            if not isinstance(info, dict):
                raise InferenceError("Installed model not found", 404)
            item = GGUFFile(**{k: info[k] for k in ("repo", "path", "commit", "size", "digest")})
            if self._id(item) in self._running:
                raise InferenceError("Cannot remove a downloading model", 409)
            target = self._target(item)
            if target.exists():
                target.unlink()  # File only; never recursively delete directories.
            del index[ident]
            self._save_index(index)
            return {"removed": True, "id": ident}

    def _run(self, job: NativeJob, identity: str) -> None:
        try:
            self._download(job)
        except Exception as exc:
            with self._lock:
                job.state = "cancelled" if job.cancel.is_set() else "failed"
                job.detail = "Cancelled; partial download kept for resume" if job.cancel.is_set() else (
                    str(exc)[:240] if isinstance(exc, InferenceError)
                    else "Download failed: " + type(exc).__name__
                )
        finally:
            with self._lock:
                self._running.discard(identity)
            self._slots.release()

    def _download(self, job: NativeJob) -> None:
        item = job.item
        target = self._target(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(target) + ".part")
        if partial.is_symlink():
            raise InferenceError("Unsafe partial download path", 400)
        if job.cancel.is_set():
            raise InferenceError("Cancelled", 499)
        if target.exists():
            if target.stat().st_size == item.size and self._hash(target) == item.digest:
                self._finish(job, target)
                return
            raise InferenceError("Different model file already exists at this path", 409)
        current = partial.stat().st_size if partial.exists() else 0
        if current > item.size:
            partial.unlink()
            current = 0
        try:
            free = shutil.disk_usage(target.parent).free
        except OSError as exc:
            raise InferenceError("Cannot measure free disk space", 507) from exc
        if free < item.size - current + (32 * 1024**2):
            raise InferenceError("Not enough free disk space for the model", 507)
        with self._lock:
            job.state, job.completed, job.detail = "downloading", current, "Downloading GGUF from Hugging Face"
        with self.catalog.download_stream(item, current) as source:
            status = source.getcode()
            if current:
                content_range = source.headers.get("Content-Range", "")
                expected = "bytes " + str(current) + "-"
                if status == 206 and content_range.startswith(expected) and content_range.endswith("/" + str(item.size)):
                    mode = "ab"
                elif status == 200:
                    # Server ignored Range; restart, never append duplicated bytes.
                    current, mode = 0, "wb"
                else:
                    raise InferenceError("Remote server returned an invalid resume response", 502)
            elif status == 200:
                mode = "wb"
            else:
                raise InferenceError("Unexpected download response", 502)
            with partial.open(mode) as destination:
                downloaded = current
                while True:
                    if job.cancel.is_set():
                        raise InferenceError("Cancelled", 499)
                    chunk = source.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > item.size:
                        raise InferenceError("Download larger than catalog size", 502)
                    destination.write(chunk)
                    with self._lock:
                        job.completed = downloaded
        if downloaded != item.size:
            raise InferenceError("Incomplete download; retry to resume", 502)
        with self._lock:
            job.state, job.detail = "verifying", "Verifying SHA-256"
        if self._hash(partial) != item.digest:
            partial.unlink(missing_ok=True)
            raise InferenceError("SHA-256 verification failed; invalid download removed", 502)
        if job.cancel.is_set():
            raise InferenceError("Cancelled", 499)
        os.replace(partial, target)
        self._finish(job, target)

    @staticmethod
    def _hash(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file:
            while block := file.read(4 * 1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def _finish(self, job: NativeJob, target: Path) -> None:
        with self._lock:
            index = self._read_index()
            item = job.item
            index[self._id(item)] = {
                "repo": item.repo, "path": item.path,
                "commit": item.commit, "size": item.size, "digest": item.digest,
            }
            self._save_index(index)
            job.state, job.detail = "completed", "SHA-256 verified; saved to " + str(target)
            job.completed = item.size
