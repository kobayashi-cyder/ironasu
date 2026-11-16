"""Authenticated, bounded Android job bridge for existing corpus agents and harness.

Workers are logical sessions; model slots are separately bounded by the gateway.
No corpus content is returned by GET. Jobs are process-local and expire on restart.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4
import os
import time
from typing import Any

from .core import AdaptiveEngine, InferenceError
from .corpus_agents import CorpusAgentTeam, MAX_CORPUS_CHARS


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"
    completed: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    stop: Event = field(default_factory=Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "state": self.state,
            "completed": self.completed, "total": self.total,
            "created_at": self.created_at, "result": self.result,
            "error": self.error,
        }


class MobileJobs:
    """Maximum two active background jobs, 20 retained finished results."""

    def __init__(self, engine: AdaptiveEngine, *, max_active: int = 2):
        self.engine = engine
        self.max_active = max_active
        self._lock = RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()

    def _register(self, kind: str) -> Job:
        with self._lock:
            active = sum(j.state in ("queued", "running") for j in self._jobs.values())
            if active >= self.max_active:
                raise InferenceError("Background jobs are busy; retry later", 429)
            for ident, existing in list(self._jobs.items()):
                if len(self._jobs) < 22:
                    break
                if existing.state not in ("queued", "running"):
                    del self._jobs[ident]
            job = Job(id=uuid4().hex, kind=kind)
            self._jobs[job.id] = job
            return job

    def get(self, ident: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(ident)
            if job is None:
                raise InferenceError("Unknown job", 404)
            return job.public()

    def cancel(self, ident: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(ident)
            if job is None:
                raise InferenceError("Unknown job", 404)
            job.stop.set()
            if job.state == "queued":
                job.state = "stopped"
            # Cancellation is cooperative: in-flight model calls can still finish.
            return job.public()

    def _thread(self, job: Job, work) -> dict[str, Any]:
        Thread(target=self._execute, args=(job, work),
               name="ironasu-mobile-" + job.kind, daemon=True).start()
        return job.public()

    def _execute(self, job: Job, work) -> None:
        with self._lock:
            if job.stop.is_set():
                job.state = "stopped"
                return
            job.state = "running"
        try:
            result = work()
            with self._lock:
                job.result = result
                job.state = "stopped" if job.stop.is_set() else (
                    "completed" if result.get("complete", True) else "incomplete")
        except Exception as exc:
            with self._lock:
                job.error = f"{type(exc).__name__}: {str(exc)[:240]}"
                job.state = "stopped" if job.stop.is_set() else "failed"

    def start_corpus(self, request: dict[str, Any]) -> dict[str, Any]:
        corpus, objective = request.get("corpus"), request.get("objective")
        parts, workers = request.get("partitions", 4), request.get("workers", 4)
        slots, model = request.get("model_slots", 2), request.get("model", "auto")
        if not isinstance(corpus, str) or not parts or not 4 <= len(corpus) <= MAX_CORPUS_CHARS:
            raise InferenceError("Corpus must contain 4..1,000,000 characters", 400)
        if not isinstance(objective, str) or not 1 <= len(objective.strip()) <= 2000:
            raise InferenceError("Objective must contain 1..2,000 characters", 400)
        if type(parts) is not int or parts not in (4, 16) or len(corpus) < parts:
            raise InferenceError("Partitions must be 4 or 16, with enough corpus text", 400)
        if type(workers) is not int or not 1 <= workers <= 16:
            raise InferenceError("Workers must be 1..16", 400)
        if type(slots) is not int or not 1 <= slots <= 16:
            raise InferenceError("Model slots must be 1..16", 400)
        if not isinstance(model, str) or not 1 <= len(model) <= 256:
            raise InferenceError("Invalid model", 400)
        effective_slots = min(slots, self.engine.config.max_concurrency)
        job = self._register("corpus")

        def progress(_partition: int, _part_done: int, _part_total: int, done: int) -> None:
            with self._lock:
                job.completed = done

        def work() -> dict[str, Any]:
            report = CorpusAgentTeam(self.engine).run(
                corpus, objective, partitions=parts, workers=workers,
                model_slots=effective_slots, model=model,
                cancel=job.stop, progress=progress)
            with self._lock:
                job.total = report.total_segments
                job.completed = min(job.completed, report.total_segments)
            # Cap the response payload while preserving exact source indices.
            return {
                "complete": report.complete,
                "partitions": report.partitions,
                "workers": report.workers,
                "model_slots": report.model_slots,
                "corpus_chars": report.corpus_chars,
                "total_segments": report.total_segments,
                "unique_segments": report.unique_segments,
                "cache_hits": report.cache_hits,
                "model_calls": report.model_calls,
                "synthesis": report.synthesis,
                "parts": [asdict(p) for p in report.parts],
                "evidence_count": len(report.evidence),
                "evidence": [asdict(e) for e in report.evidence[:128]],
                "evidence_truncated": len(report.evidence) > 128,
            }

        return self._thread(job, work)

    def start_harness(self, request: dict[str, Any]) -> dict[str, Any]:
        """Opt-in, read-only harness. Never exposes arbitrary workspace paths."""
        workspace = os.environ.get("IRONASU_MOBILE_HARNESS_WORKSPACE", "")
        if os.environ.get("IRONASU_MOBILE_HARNESS_ENABLED") != "1" or not workspace:
            raise InferenceError("Mobile harness is disabled on the PC", 403)
        task = request.get("task")
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 4000:
            raise InferenceError("Task must contain 1..4,000 characters", 400)
        job = self._register("harness")

        def work() -> dict[str, Any]:
            from .harness import Harness, Settings
            config = self.engine.config
            settings = Settings(
                workspace=Path(workspace),
                endpoint=f"http://127.0.0.1:{config.listen_port}/v1",
                model="ironasu",
                api_key=config.server_key,
                allow_writes=False, allow_tests=False,
                max_steps=8,
            )
            # Each task has an isolated session. Cancellation is cooperative:
            # it cannot interrupt an already-running harness tool call.
            if job.stop.is_set():
                return {"complete": False, "output": "Cancelled before start"}
            output = Harness(settings).run(job.id, task)
            return {"complete": not job.stop.is_set(), "output": output[:24000]}

        return self._thread(job, work)
