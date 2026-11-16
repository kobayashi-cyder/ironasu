"""Adaptive compute-saving proxy for OpenAI-compatible local inference backends."""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import threading
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .context import select_code
from .budget import RequestBudget


class InferenceError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Config:
    backend_url: str = "http://127.0.0.1:8080/v1"
    backend_key: str = ""
    fast_model: str = "qwen2.5-coder-1.5b"
    quality_model: str = "qwen2.5-coder-7b"
    auto_models: bool = False
    max_concurrency: int = 2
    max_waiting_requests: int = 32
    queue_timeout_s: float = 10.0
    request_timeout_s: float = 600.0
    slow_request_s: float = 20.0
    cache_entries: int = 128
    cache_ttl_s: float = 300.0
    server_key: str = ""
    listen_host: str = "127.0.0.1"
    listen_port: int = 8765

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ
        return cls(
            backend_url=e.get("IRONASU_BACKEND_URL", cls.backend_url),
            backend_key=e.get("IRONASU_BACKEND_KEY", ""),
            fast_model=e.get("IRONASU_FAST_MODEL", cls.fast_model),
            quality_model=e.get("IRONASU_QUALITY_MODEL", cls.quality_model),
            auto_models=e.get("IRONASU_AUTO_MODELS", "").lower() in ("1", "true", "yes"),
            max_concurrency=int(e.get("IRONASU_MAX_CONCURRENCY", 2)),
            max_waiting_requests=int(e.get("IRONASU_MAX_WAITING_REQUESTS", 32)),
            queue_timeout_s=float(e.get("IRONASU_QUEUE_TIMEOUT_S", 10)),
            request_timeout_s=float(e.get("IRONASU_REQUEST_TIMEOUT_S", 600)),
            slow_request_s=float(e.get("IRONASU_SLOW_REQUEST_S", 20)),
            cache_entries=int(e.get("IRONASU_CACHE_ENTRIES", 128)),
            cache_ttl_s=float(e.get("IRONASU_CACHE_TTL_S", 300)),
            server_key=e.get("IRONASU_SERVER_KEY", ""),
            listen_host=e.get("IRONASU_HOST", "127.0.0.1"),
            listen_port=int(e.get("IRONASU_PORT", 8765)),
        )

    def validate(self) -> None:
        if not (self.backend_url.startswith("http://") or self.backend_url.startswith("https://")):
            raise ValueError("backend URL must be http(s)")
        if (type(self.max_waiting_requests) is not int or
                not 0 <= self.max_waiting_requests <= 1024):
            raise ValueError("max_waiting_requests must be in [0, 1024]")
        if min(self.max_concurrency, self.cache_entries) < 1 or min(
            self.queue_timeout_s, self.request_timeout_s, self.slow_request_s, self.cache_ttl_s
        ) <= 0:
            raise ValueError("invalid concurrency, timeout, or cache settings")
        if not 1 <= self.listen_port <= 65535:
            raise ValueError("invalid listen port")
        if self.listen_host not in ("127.0.0.1", "::1", "localhost") and not self.server_key:
            raise ValueError("IRONASU_SERVER_KEY is mandatory when binding beyond localhost")


class Backend(Protocol):
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class HTTPBackend:
    def __init__(self, config: Config):
        self.config = config

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": accept}
        if self.config.backend_key:
            headers["Authorization"] = f"Bearer {self.config.backend_key}"
        return headers

    def open_stream(self, payload: dict[str, Any]):
        """Open upstream SSE before sending HTTP 200 to the caller."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            self.config.backend_url.rstrip("/") + "/chat/completions",
            data=data, headers=self._headers("text/event-stream"), method="POST",
        )
        try:
            response = urlopen(req, timeout=self.config.request_timeout_s)
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                response.close()
                raise InferenceError("upstream does not provide SSE streaming", 502)
            return response
        except HTTPError as exc:
            raise InferenceError(f"upstream returned HTTP {exc.code}", 502) from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise InferenceError("upstream streaming unavailable", 502) from exc

    def list_models(self, *, timeout_s: float = 3.0) -> list[str] | None:
        """Best-effort discovery; never send upstream credentials to the browser."""
        req = Request(self.config.backend_url.rstrip("/") + "/models",
                      headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=min(timeout_s, self.config.request_timeout_s)) as response:
                raw = response.read(262145)
            if len(raw) > 262144:
                return None
            catalog = json.loads(raw)
            entries = catalog.get("data", []) if isinstance(catalog, dict) else []
            if not isinstance(entries, list):
                return None
            return list(dict.fromkeys(
                entry["id"] for entry in entries[:200]
                if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                and 0 < len(entry["id"]) <= 256
            ))
        except (HTTPError, URLError, OSError, TimeoutError, ValueError):
            return None

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._headers()
        req = Request(
            self.config.backend_url.rstrip("/") + "/chat/completions",
            data=data, headers=headers, method="POST",
        )
        try:
            with urlopen(req, timeout=self.config.request_timeout_s) as resp:
                output = json.load(resp)
        except HTTPError as exc:
            raise InferenceError(f"upstream returned HTTP {exc.code}", status=502) from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise InferenceError("upstream unavailable or invalid JSON") from exc
        if not isinstance(output, dict) or "choices" not in output:
            raise InferenceError("upstream returned an invalid chat completion")
        return output


class ExactCache:
    """TTL-bound in-memory cache: never store data on disk."""

    def __init__(self, max_entries: int, ttl_s: float):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if time.monotonic() > item[0]:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return dict(item[1])

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_s, dict(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


class Metrics:
    def __init__(self, slow_request_s: float):
        self.slow_request_s = slow_request_s
        self._lock = threading.Lock()
        self.calls = defaultdict(int)
        self.cache_hits = 0
        self.singleflight_hits = 0
        self.overloads = 0
        self.errors = 0
        self.policy_requests = 0
        self.policy_quality_guards = 0
        self.policy_queue_limits = 0
        self.policy_cache_bypasses = 0
        self.reused_reported_output_tokens = 0
        self.reuse_usage_unknown = 0
        self.active_requests = 0
        self.waiting_requests = 0
        self.peak_active_requests = 0
        self.peak_waiting_requests = 0
        self.latency = defaultdict(lambda: deque(maxlen=32))
        self.queue_wait = deque(maxlen=32)
        self.pressure = False
        self._recovery = 0

    def record(self, route: str, latency_s: float, queue_s: float, failed: bool) -> None:
        with self._lock:
            self.calls[route] += 1
            self.errors += int(failed)
            self.latency[route].append(round(latency_s, 4))
            self.queue_wait.append(round(queue_s, 4))
            # The latency threshold is a signal, not an assertion about FLOPs.
            if latency_s > self.slow_request_s or queue_s > self.slow_request_s / 4:
                self.pressure = True
                self._recovery = 0
            elif self.pressure:
                self._recovery += 1
                if self._recovery >= 3:
                    self.pressure = False
                    self._recovery = 0

    def adjust_admission(self, *, active_delta: int = 0, waiting_delta: int = 0) -> None:
        """Track bounded active/queued requests without storing request content."""
        with self._lock:
            self.active_requests += active_delta
            self.waiting_requests += waiting_delta
            if self.active_requests < 0 or self.waiting_requests < 0:
                raise RuntimeError("admission accounting underflow")
            self.peak_active_requests = max(self.peak_active_requests, self.active_requests)
            self.peak_waiting_requests = max(self.peak_waiting_requests, self.waiting_requests)

    def record_policy(self, budget: RequestBudget) -> None:
        if not budget.explicit:
            return
        with self._lock:
            self.policy_requests += 1
            self.policy_quality_guards += int(budget.preserve_auto_quality)
            self.policy_queue_limits += int(budget.max_queue_ms is not None)
            self.policy_cache_bypasses += int(not budget.allow_cache)

    def record_reuse(self, source: str, result: dict[str, Any]) -> None:
        """Record only successful cache / Singleflight response reuse."""
        if source not in ("cache", "singleflight"):
            raise ValueError("invalid reuse source")
        usage = result.get("usage") if isinstance(result, dict) else None
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        with self._lock:
            if source == "cache":
                self.cache_hits += 1
            else:
                self.singleflight_hits += 1
            if type(tokens) is int and tokens >= 0:
                self.reused_reported_output_tokens += tokens
            else:
                self.reuse_usage_unknown += 1

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def is_pressured(self) -> bool:
        with self._lock:
            return self.pressure

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            upstream_attempts = sum(self.calls.values())
            upstream_successes = max(0, upstream_attempts - self.errors)
            avoided = self.cache_hits + self.singleflight_hits
            served = upstream_successes + avoided
            return {
                "backend_calls": dict(self.calls),
                "cache_hits": self.cache_hits,
                "singleflight_hits": self.singleflight_hits,
                "overloads": self.overloads,
                "active_requests": self.active_requests,
                "waiting_requests": self.waiting_requests,
                "peak_active_requests": self.peak_active_requests,
                "peak_waiting_requests": self.peak_waiting_requests,
                "errors": self.errors,
                "pressure": self.pressure,
                "latency_seconds": {k: list(v) for k, v in self.latency.items()},
                "queue_wait_seconds": list(self.queue_wait),
                "execution_budget": {
                    "explicit_plans": self.policy_requests,
                    "quality_guard_requested": self.policy_quality_guards,
                    "queue_limit_requested": self.policy_queue_limits,
                    "cache_bypass_requested": self.policy_cache_bypasses,
                },
                "efficiency": {
                    "observed": {
                        "avoided_upstream_requests": avoided,
                        "cache_responses": self.cache_hits,
                        "singleflight_responses": self.singleflight_hits,
                        "successful_upstream_responses": upstream_successes,
                        "reused_reported_output_tokens": (
                            self.reused_reported_output_tokens if self.reuse_usage_unknown == 0 else None
                        ),
                        "reuse_missing_token_usage": self.reuse_usage_unknown,
                    },
                    "derived": {
                        "response_reuse_fraction": avoided / served if served else None,
                    },
                    "unmeasured": [
                        "GPU compute avoided", "GPU memory saved", "FLOPs saved",
                        "energy saved", "counterfactual latency saved",
                    ],
                },
            }


# Only these narrow, short operations are automatically eligible for smaller models.
_SIMPLE_MARKERS = ("explain", "docstring", "rename variable", "format this", "コメント", "説明して", "整形して", "変数名", "命名")
_COMPLEX_MARKERS = ("security", "auth", "architecture", "migration", "multi-file", "repository", "脆弱", "設計", "全ファイル", "認証", "データ移行")


def _route(messages: list[dict[str, Any]], mode: str, pressure: bool) -> str:
    if mode in ("fast", "quality"):
        return mode
    text = "\n".join(str(m.get("content", "")) for m in messages).lower()
    if len(text) > 900 or any(x in text for x in _COMPLEX_MARKERS):
        return "quality"
    if any(x in text for x in _SIMPLE_MARKERS):
        return "fast"
    # Under slow-backend pressure, ambiguous short requests can use the small model.
    if pressure and len(text) <= 300:
        return "fast"
    return "quality"


class AdaptiveEngine:
    def __init__(self, config: Config, backend: Backend | None = None):
        config.validate()
        self.config = config
        self.backend = backend or HTTPBackend(config)
        self.cache = ExactCache(config.cache_entries, config.cache_ttl_s)
        self.metrics = Metrics(config.slow_request_s)
        self._slots = threading.BoundedSemaphore(config.max_concurrency)
        # Includes both queued upstream calls and singleflight followers.
        # ThreadingHTTPServer has unbounded threads; reject waiting work early.
        self._waiters = threading.BoundedSemaphore(config.max_waiting_requests)
        self._flight_lock = threading.Lock()
        self._inflight: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._context_lock = threading.Lock()
        self._context_stats = {"requests": 0, "original_source_chars": 0, "selected_source_chars": 0}

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise InferenceError("request must be a JSON object", 400)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(x, dict) for x in messages):
            raise InferenceError("messages must be a nonempty list of objects", 400)
        if payload.get("stream"):
            raise InferenceError("streaming is not yet supported", 400)
        try:
            budget = RequestBudget.from_request(payload)
        except ValueError as exc:
            raise InferenceError(str(exc), 400) from exc
        route, model = self._select_model(payload, messages, budget)
        self.metrics.record_policy(budget)
        request = {k: v for k, v in payload.items() if not k.startswith("x_ironasu_")}
        request["model"] = model
        request.pop("stream", None)
        context_report = None
        if "x_ironasu_files" in payload:
            user_positions = [idx for idx, item in enumerate(messages)
                              if item.get("role") == "user"]
            if not user_positions or not isinstance(messages[user_positions[-1]].get("content"), str):
                raise InferenceError("code retrieval requires a text user message", 400)
            index = user_positions[-1]
            try:
                selected = select_code(
                    payload["x_ironasu_files"], messages[index]["content"],
                    payload.get("x_ironasu_context_chars", 12000),
                )
            except ValueError as exc:
                raise InferenceError(str(exc), 400) from exc
            request["messages"] = [dict(item) for item in messages]
            request["messages"][index]["content"] += "\n\n" + selected.text
            context_report = selected.report()
            with self._context_lock:
                self._context_stats["requests"] += 1
                self._context_stats["original_source_chars"] += selected.original_chars
                self._context_stats["selected_source_chars"] += selected.selected_chars
        # No caching for tools, nondeterministic requests, or explicit opt-out.
        cacheable = (
            budget.allow_cache
            and payload.get("x_ironasu_cache", True) is True
            and payload.get("temperature", 1) == 0
            and not payload.get("tools")
            and not payload.get("tool_choice")
        )
        key = sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest() if cacheable else ""
        if key:
            cached = self.cache.get(key)
            if cached is not None:
                self.metrics.record_reuse("cache", cached)
                return self._annotate(cached, route, "cache", context_report, budget)
            with self._flight_lock:
                if key in self._inflight:
                    event, holder = self._inflight[key]
                    follower = True
                else:
                    event, holder = threading.Event(), {}
                    self._inflight[key] = (event, holder)
                    follower = False
            if follower:
                # Followers also occupy a waiting thread; enforce the same cap
                # used for upstream queueing to prevent unbounded singleflight waiters.
                wait_budget = self.config.request_timeout_s + self.config.queue_timeout_s
                if budget.max_queue_ms is not None:
                    wait_budget = min(wait_budget, budget.max_queue_ms / 1000.0)
                if wait_budget == 0 and not event.is_set():
                    self.metrics.bump("overloads")
                    raise InferenceError("singleflight wait budget exhausted", 503)
                self._reserve_waiter()
                try:
                    if not event.wait(wait_budget):
                        raise InferenceError("identical upstream request timed out", 504)
                    if "error" in holder:
                        raise holder["error"]
                    result = holder["result"]
                    self.metrics.record_reuse("singleflight", result)
                    return self._annotate(result, route, "singleflight", context_report, budget)
                finally:
                    self._release_waiter()
        try:
            result = self._invoke(request, route, queue_timeout_s=budget.queue_timeout_s(self.config.queue_timeout_s))
            if key:
                self.cache.put(key, result)
                holder["result"] = result
            return self._annotate(result, route, "upstream", context_report, budget)
        except Exception as exc:
            if key:
                holder["error"] = exc
            raise
        finally:
            if key:
                with self._flight_lock:
                    self._inflight.pop(key, None)
                    event.set()

    def _select_model(
        self, payload: dict[str, Any], messages: list[dict[str, Any]],
        budget: RequestBudget | None = None,
    ) -> tuple[str, str]:
        mode = payload.get("x_ironasu_mode", "auto")
        if mode not in ("auto", "fast", "quality"):
            raise InferenceError("x_ironasu_mode must be auto, fast, or quality", 400)
        direct = payload.get("x_ironasu_model")
        if direct is not None:
            if (not isinstance(direct, str) or not 0 < len(direct) <= 256
                    or any(ord(c) < 32 for c in direct)):
                raise InferenceError("invalid x_ironasu_model", 400)
            return "direct", direct
        if mode == "auto":
            if payload.get("model") == "ironasu-fast":
                mode = "fast"
            elif payload.get("model") == "ironasu-quality":
                mode = "quality"
        pressure = self.metrics.is_pressured()
        route = _route(messages, mode,
                       budget.pressure_for_routing(pressure) if budget else pressure)
        chosen = self.config.fast_model if route == "fast" else self.config.quality_model
        if self.config.auto_models and not chosen:
            catalog = self.upstream_models()
            if not catalog:
                raise InferenceError(
                    "No model available. Start a local model server and load a model, then retry.",
                    503,
                )
            chosen = catalog[0]
        return route, chosen

    @contextmanager
    def stream(self, payload: dict[str, Any]):
        """Bounded, uncached SSE passthrough. No tool execution or disk state."""
        if not isinstance(payload, dict):
            raise InferenceError("request must be a JSON object", 400)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(item, dict) for item in messages
        ):
            raise InferenceError("messages must be a nonempty list of objects", 400)
        if payload.get("stream") is not True:
            raise InferenceError("stream must be true", 400)
        if "x_ironasu_files" in payload:
            raise InferenceError("streaming with code-context retrieval is not supported", 400)
        try:
            budget = RequestBudget.from_request(payload)
        except ValueError as exc:
            raise InferenceError(str(exc), 400) from exc
        route, model = self._select_model(payload, messages, budget)
        self.metrics.record_policy(budget)
        request = {k: v for k, v in payload.items() if not k.startswith("x_ironasu_")}
        request["model"] = model
        request["stream"] = True
        if not hasattr(self.backend, "open_stream"):
            raise InferenceError("this backend does not support streaming", 501)
        queued = time.perf_counter()
        acquired = self._slots.acquire(blocking=False)
        if not acquired:
            wait_budget = budget.queue_timeout_s(self.config.queue_timeout_s)
            if wait_budget == 0:
                self.metrics.bump("overloads")
                raise InferenceError("queue wait budget exhausted", 503)
            self._reserve_waiter()
            try:
                acquired = self._slots.acquire(timeout=wait_budget)
            finally:
                self._release_waiter()
        wait_s = time.perf_counter() - queued
        if not acquired:
            self.metrics.bump("overloads")
            raise InferenceError("queue is saturated; retry later", 503)
        self.metrics.adjust_admission(active_delta=1)
        started = time.perf_counter()
        failed = False
        try:
            with self.backend.open_stream(request) as response:
                yield response
        except Exception:
            failed = True
            raise
        finally:
            self.metrics.record(route, time.perf_counter() - started, wait_s, failed)
            self.metrics.adjust_admission(active_delta=-1)
            self._slots.release()

    def effective_models(self, discovered: list[str] | None = None) -> tuple[str | None, str | None]:
        if discovered is None:
            discovered = self.upstream_models()
        first = discovered[0] if discovered else None
        return (
            self.config.fast_model or (first if self.config.auto_models else None),
            self.config.quality_model or (first if self.config.auto_models else None),
        )

    def upstream_models(self) -> list[str] | None:
        discover = getattr(self.backend, "list_models", None)
        return discover() if callable(discover) else None

    def _reserve_waiter(self) -> None:
        if not self._waiters.acquire(blocking=False):
            self.metrics.bump("overloads")
            raise InferenceError("queue is saturated; retry later", 503)
        self.metrics.adjust_admission(waiting_delta=1)

    def _release_waiter(self) -> None:
        self.metrics.adjust_admission(waiting_delta=-1)
        self._waiters.release()

    def _invoke(self, request: dict[str, Any], route: str, *,
                queue_timeout_s: float | None = None) -> dict[str, Any]:
        queued = time.perf_counter()
        timeout = self.config.queue_timeout_s if queue_timeout_s is None else queue_timeout_s
        # Do not consume queue capacity when an execution slot is already free.
        acquired = self._slots.acquire(blocking=False)
        if not acquired:
            if timeout == 0:
                self.metrics.bump("overloads")
                raise InferenceError("queue wait budget exhausted", 503)
            self._reserve_waiter()
            try:
                acquired = self._slots.acquire(timeout=timeout)
            finally:
                self._release_waiter()
        wait_s = time.perf_counter() - queued
        if not acquired:
            self.metrics.bump("overloads")
            raise InferenceError("queue is saturated; retry later", 503)
        self.metrics.adjust_admission(active_delta=1)
        started = time.perf_counter()
        failed = False
        try:
            return self.backend.complete(request)
        except Exception:
            failed = True
            raise
        finally:
            self.metrics.record(route, time.perf_counter() - started, wait_s, failed)
            self.metrics.adjust_admission(active_delta=-1)
            self._slots.release()

    @staticmethod
    def _annotate(result: dict[str, Any], route: str, source: str,
                  context: dict | None = None, budget: RequestBudget | None = None) -> dict[str, Any]:
        metadata = {"route": route, "source": source}
        if context is not None:
            metadata["context_retrieval"] = context
        if budget and budget.explicit:
            metadata["execution_budget"] = budget.diagnostic(
                selected_route=route, observed_pressure=False)
            # Pressure is intentionally omitted here: the current pressure
            # may differ from the observation at admission time.
        return {**result, "_ironasu": metadata}

    def diagnostic_snapshot(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        with self._context_lock:
            snapshot["context_retrieval"] = dict(self._context_stats)
        return snapshot

