"""Bounded, read-only agent team jobs exposed to the authenticated mobile UI.

This does not accept workspace paths, shell commands, write permissions or tests
from the client. The host process administrator controls the workspace.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import RLock
import os
import time
from uuid import uuid4

from .core import Config, InferenceError
from .team import TeamTask, run_harness_team


class MobileJobManager:
    """One active bounded team run with a small bounded in-memory history."""

    def __init__(self, config: Config, *, runner=run_harness_team, workspace: Path | None = None):
        self.config = config
        self._runner = runner
        self.workspace = Path(workspace or os.environ.get("IRONASU_MOBILE_WORKSPACE") or Path.cwd()).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("mobile workspace must be a directory")
        self._lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ironasu-mobile-team")
        self._jobs: dict[str, dict] = {}
        self._active: str | None = None

    @staticmethod
    def _task(item: object) -> TeamTask:
        if not isinstance(item, dict):
            raise InferenceError("each agent task must be an object", 400)
        deps = item.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise InferenceError("depends_on must be a list of task IDs", 400)
        try:
            return TeamTask(task_id=item.get("task_id"), prompt=item.get("prompt"),
                            depends_on=tuple(deps), priority=item.get("priority", 0))
        except (ValueError, TypeError) as exc:
            raise InferenceError(str(exc), 400) from exc

    def start(self, payload: dict) -> dict:
        raw = payload.get("tasks")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
            raise InferenceError("tasks must contain between 1 and 16 entries", 400)
        tasks = tuple(self._task(item) for item in raw)
        context = payload.get("shared_context", "")
        workers = payload.get("workers", min(4, len(tasks)))
        slots = payload.get("model_slots", 2)
        if not isinstance(context, str) or len(context) > 32_000:
            raise InferenceError("shared_context exceeds 32000 characters", 400)
        if type(workers) is not int or not 1 <= workers <= 16:
            raise InferenceError("workers must be 1..16", 400)
        if type(slots) is not int or not 1 <= slots <= 16:
            raise InferenceError("model_slots must be 1..16", 400)
        # Validate all dependencies/cycles before consuming an execution slot.
        from .team import TeamScheduler
        try:
            TeamScheduler._validate(tasks)
        except ValueError as exc:
            raise InferenceError(str(exc), 400) from exc
        # Never exceed the gateway's configured LLM admission budget.
        effective_slots = min(slots, workers, self.config.max_concurrency)
        identifier = uuid4().hex[:12]
        item = {"id": identifier, "state": "queued", "workers": workers,
                "model_slots": effective_slots, "created_at": time.time(),
                "finished_at": None, "results": [], "error": ""}
        with self._lock:
            if self._active is not None:
                raise InferenceError("another mobile team job is still active", 409)
            self._active = identifier
            # Keep only a few jobs in memory, never unbounded history.
            while len(self._jobs) >= 8:
                self._jobs.pop(next(iter(self._jobs)))
            self._jobs[identifier] = item
            self._pool.submit(self._execute, identifier, tasks, context, workers, effective_slots)
        return self.get(identifier)

    def _execute(self, identifier, tasks, context, workers, slots):
        with self._lock:
            self._jobs[identifier]["state"] = "running"
        try:
            results = self._runner(
                tasks, workspace=self.workspace,
                endpoint=f"http://127.0.0.1:{self.config.listen_port}/v1",
                model="ironasu", api_key=self.config.server_key,
                shared_context=context, workers=workers, model_slots=slots,
                max_steps=6, run_id=identifier,
            )
            with self._lock:
                item = self._jobs[identifier]
                item["results"] = [asdict(x) for x in results]
                item["state"] = ("completed" if all(x.status == "succeeded" for x in results)
                                 else "partial_failure")
        except Exception as exc:
            with self._lock:
                self._jobs[identifier]["state"] = "failed"
                self._jobs[identifier]["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            with self._lock:
                self._jobs[identifier]["finished_at"] = time.time()
                self._active = None

    def get(self, identifier: str) -> dict:
        with self._lock:
            value = self._jobs.get(identifier)
            if value is None:
                raise InferenceError("agent job not found", 404)
            return {**value, "results": [dict(x) for x in value["results"]]}

    def list_jobs(self) -> dict:
        with self._lock:
            return {"jobs": [self.get(i) for i in reversed(self._jobs)]}

    def close(self) -> None:
        # Does not interrupt a possibly non-idempotent harness tool call.
        self._pool.shutdown(wait=False, cancel_futures=True)