"""Durable bounded development missions for local OpenAI-compatible models.

Persisted task DAG -> permission-gated harness -> independent verification ->
bounded repair. Does not provide a shell or general OS computer automation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from threading import BoundedSemaphore, Event
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

from .harness import Harness, OpenAIBackend, Settings, _clip
from .team import TeamScheduler, TeamTask

_ID = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
_CHECKS = {"unittest", "pytest", "compileall", "git_diff_check"}
_MAX_GOAL = 8000
_MAX_TASKS = 16


@contextmanager
def _exclusive(workspace: Path, ident: str):
    """Process-level lock released by the OS even if Python crashes."""
    directory = workspace / ".ironasu"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / ("mission_" + ident + ".lock")).open("a+b") as fp:
        fp.seek(0, os.SEEK_END)
        if not fp.tell():
            fp.write(b"0")
            fp.flush()
        fp.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Mission already active") from exc
            try:
                yield
            finally:
                fp.seek(0)
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Mission already active") from exc
            try:
                yield
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class MissionJournal:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve(strict=True)
        root = self.workspace / ".ironasu"
        root.mkdir(mode=0o700, exist_ok=True)
        self.path = root / "missions.sqlite"
        with self.db() as con:
            con.executescript("""CREATE TABLE IF NOT EXISTS missions (
                id TEXT PRIMARY KEY, goal TEXT NOT NULL, workspace TEXT NOT NULL,
                endpoint TEXT NOT NULL, model TEXT NOT NULL, plan TEXT NOT NULL,
                checks TEXT NOT NULL, artifacts TEXT NOT NULL,
                state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                verdict TEXT NOT NULL DEFAULT '{}', updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS mission_tasks (
                mission_id TEXT NOT NULL, task_id TEXT NOT NULL,
                state TEXT NOT NULL, output TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(mission_id,task_id));
                CREATE TABLE IF NOT EXISTS mission_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT NOT NULL, at REAL NOT NULL,
                kind TEXT NOT NULL, detail TEXT NOT NULL);""")

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def create(self, ident, goal, endpoint, model, tasks, checks, artifacts):
        with self.db() as con:
            con.execute(
                "INSERT INTO missions(id,goal,workspace,endpoint,model,plan,checks,"
                "artifacts,state,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ident, goal, str(self.workspace), endpoint, model,
                 json.dumps([asdict(t) for t in tasks]),
                 json.dumps(checks), json.dumps(artifacts), "planned", time.time()))
            con.executemany(
                "INSERT INTO mission_tasks(mission_id,task_id,state) VALUES(?,?,?)",
                [(ident, t.task_id, "pending") for t in tasks])

    def mission(self, ident):
        with self.db() as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM missions WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise ValueError("Mission not found")
        return dict(row)

    def tasks(self, ident):
        with self.db() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM mission_tasks WHERE mission_id=? ORDER BY rowid",
                (ident,)).fetchall()
        return [dict(x) for x in rows]

    def recent(self, limit: int = 30):
        with self.db() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id,goal,state,updated FROM missions "
                "ORDER BY updated DESC LIMIT ?",
                (min(100, max(1, limit)),)).fetchall()
        return [dict(x) for x in rows]

    def task(self, ident, task_id):
        result = next((t for t in self.tasks(ident) if t["task_id"] == task_id), None)
        if result is None:
            raise ValueError("Task not found")
        return result

    def set_task(self, ident, task_id, state, output="", error=""):
        with self.db() as con:
            con.execute(
                "UPDATE mission_tasks SET state=?,output=?,error=? "
                "WHERE mission_id=? AND task_id=?",
                (state, output[:8000], error[:3000], ident, task_id))

    def set_state(self, ident, state, attempts=None, verdict=None):
        fields = ["state=?", "updated=?"]
        values = [state, time.time()]
        if attempts is not None:
            fields.append("attempts=?")
            values.append(attempts)
        if verdict is not None:
            fields.append("verdict=?")
            values.append(json.dumps(verdict, ensure_ascii=False))
        with self.db() as con:
            con.execute("UPDATE missions SET " + ",".join(fields) + " WHERE id=?",
                        (*values, ident))

    def event(self, ident, kind, detail):
        with self.db() as con:
            con.execute("INSERT INTO mission_events(mission_id,at,kind,detail)"
                        " VALUES(?,?,?,?)",
                        (ident, time.time(), kind[:80], detail[:3000]))

    def events(self, ident, limit=80):
        with self.db() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT seq,at,kind,detail FROM mission_events WHERE mission_id=? "
                "ORDER BY seq DESC LIMIT ?", (ident, min(500, max(1, limit)))).fetchall()
        return [dict(x) for x in reversed(rows)]


def parse_plan(value: Any) -> tuple[TeamTask, ...]:
    if isinstance(value, dict):
        value = value.get("tasks")
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= _MAX_TASKS:
        raise ValueError("Plan must contain 1..16 tasks")
    items = []
    for obj in value:
        if not isinstance(obj, dict) or not isinstance(obj.get("depends_on", []), (list, tuple)):
            raise ValueError("Task must contain a valid dependency list")
        items.append(TeamTask(obj["task_id"], obj["prompt"],
                              tuple(obj.get("depends_on", ())), obj.get("priority", 0)))
    TeamScheduler._validate(tuple(items))
    return tuple(items)


@dataclass(frozen=True)
class MissionReport:
    mission_id: str
    state: str
    goal: str
    tasks: tuple[dict[str, Any], ...]
    plan: tuple[dict[str, Any], ...]
    checks: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    repairs: int
    verified: bool
    reason: str

    def as_dict(self):
        return asdict(self)


def check_acceptance(workspace: Path, kinds: tuple[str, ...],
                     artifact_paths: tuple[str, ...], *, allowed: bool,
                     stop: Event) -> dict[str, Any]:
    artifacts = []
    for name in artifact_paths:
        p = (workspace / name).resolve()
        if (not p.is_relative_to(workspace)
                or any(part in (".git", ".ironasu", ".ssh")
                       for part in Path(name).parts)
                or any(part in (".git", ".ironasu", ".ssh")
                       for part in p.relative_to(workspace).parts)):
            raise ValueError("Private/escaping artifact path")
        present = p.is_file()
        digest = (sha256(p.read_bytes()).hexdigest()
                  if present and p.stat().st_size <= 16 * 1024 * 1024 else None)
        artifacts.append({"path": name, "exists": present, "sha256": digest})
    results = []
    if kinds and not allowed:
        return {"passed": False, "reason": "Tests require explicit permission",
                "checks": results, "artifacts": artifacts}
    commands = {
        "unittest": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "compileall": [sys.executable, "-m", "compileall", "-q", "."],
        "git_diff_check": ["git", "diff", "--check"],
    }
    for name in kinds:
        if stop.is_set():
            return {"passed": False, "reason": "cancelled",
                    "checks": results, "artifacts": artifacts}
        environment = os.environ.copy()
        for secret in ("IRONASU_BACKEND_KEY", "OPENAI_API_KEY"):
            environment.pop(secret, None)
        try:
            result = subprocess.run(commands[name], cwd=workspace,
                                    capture_output=True, text=True, shell=False,
                                    timeout=120, env=environment)
            results.append({
                "kind": name, "passed": result.returncode == 0,
                "exit_code": result.returncode,
                "output": _clip(result.stdout + "\n" + result.stderr, 3000),
            })
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append({"kind": name, "passed": False, "exit_code": None,
                            "output": f"{type(exc).__name__}: {exc}"[:3000]})
    passed = bool(kinds or artifact_paths) and all(
        a["exists"] for a in artifacts) and all(c["passed"] for c in results)
    return {"passed": passed, "checks": results, "artifacts": artifacts,
            "reason": ("Requested checks/artifacts verified" if passed else
                       "No acceptance criteria or verification failed")}


class MissionController:
    def __init__(self, workspace: Path, *, endpoint: str, model: str,
                 api_key: str = "",
                 backend_factory: Callable[[Settings], Any] = OpenAIBackend):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        Settings(workspace=self.workspace, endpoint=endpoint, model=model)
        self.endpoint, self.model, self.api_key = endpoint, model, api_key
        self.backend_factory = backend_factory
        self.journal = MissionJournal(self.workspace)

    def _agent(self, gate, *, writes=False, tests=False, steps=12):
        settings = Settings(
            workspace=self.workspace, endpoint=self.endpoint, model=self.model,
            api_key=self.api_key, allow_writes=writes, allow_tests=tests,
            max_steps=steps)
        agent = Harness(settings)
        original = self.backend_factory(settings)
        class GatedBackend:
            def complete(self, messages, tools):
                with gate:
                    return original.complete(messages, tools)
        agent.backend = GatedBackend()
        return agent

    def propose(self, goal: str) -> tuple[TeamTask, ...]:
        settings = Settings(workspace=self.workspace, endpoint=self.endpoint,
                            model=self.model, api_key=self.api_key)
        instructions = (
            "Return only JSON: an object with tasks: an array of 1..8 objects "
            "with task_id, prompt, depends_on (string array), priority (integer). "
            "Plan read-only inspection before edits and verification last. "
            "Task IDs must be short ASCII. Do not propose dangerous actions. "
            "This plan grants no permissions; tools are separately restricted.")
        try:
            answer = self.backend_factory(settings).complete(
                [{"role": "system", "content": instructions},
                 {"role": "user", "content": goal}], [])
            raw = str(answer.get("content") or "").strip()
            if raw.startswith(chr(96)*3):
                raw = raw[raw.find("\n")+1:].rsplit(chr(96)*3, 1)[0].strip()
            return parse_plan(json.loads(raw))
        except (ValueError, KeyError, TypeError, RuntimeError):
            return (TeamTask("work", goal),)

    def report(self, ident: str) -> MissionReport:
        row = self.journal.mission(ident)
        verdict = json.loads(row["verdict"])
        return MissionReport(
            ident, row["state"], row["goal"], tuple(self.journal.tasks(ident)),
            tuple(json.loads(row["plan"])), tuple(verdict.get("checks", [])),
            tuple(verdict.get("artifacts", [])), row["attempts"],
            row["state"] == "verified", verdict.get("reason", ""))

    def run(self, *, goal: str | None = None, mission_id: str | None = None,
            plan: Any = None, checks: Iterable[str] = ("compileall", "git_diff_check"),
            accept_files: Iterable[str] = (), workers: int = 4,
            model_slots: int = 2, max_steps: int = 12, max_repairs: int = 2,
            allow_writes: bool = False, allow_tests: bool = False,
            cancel: Event | None = None, progress: Callable[[dict], None] | None = None,
            resume: bool = False, auto_plan: bool = True) -> MissionReport:
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("workers must be 1..16")
        if type(model_slots) is not int or not 1 <= model_slots <= 16:
            raise ValueError("model slots must be 1..16")
        if type(max_steps) is not int or not 1 <= max_steps <= 100:
            raise ValueError("max_steps must be 1..100")
        if type(max_repairs) is not int or not 0 <= max_repairs <= 5:
            raise ValueError("repairs must be 0..5")
        kinds, paths = tuple(checks), tuple(accept_files)
        if len(kinds) > 4 or any(k not in _CHECKS for k in kinds):
            raise ValueError("Unsupported verification check")
        if len(paths) > 16 or any(not isinstance(p, str) or not p or len(p) > 256
                                 or Path(p).is_absolute() or ".." in Path(p).parts
                                 for p in paths):
            raise ValueError("Artifacts must be workspace-relative paths")
        ident = mission_id or uuid4().hex[:16]
        if not _ID.fullmatch(ident):
            raise ValueError("Invalid mission ID")
        stop = cancel or Event()
        def event(kind, detail):
            self.journal.event(ident, kind, detail)
            if progress:
                progress({"mission_id": ident, "kind": kind,
                          "detail": detail[:3000]})

        with _exclusive(self.workspace, ident):
            if not resume:
                if not isinstance(goal, str) or not goal.strip() or len(goal) > _MAX_GOAL:
                    raise ValueError("New mission requires 1..8000 goal characters")
                tasks = (parse_plan(plan) if plan is not None
                         else self.propose(goal) if auto_plan
                         else (TeamTask("work", goal),))
                self.journal.create(ident, goal, self.endpoint, self.model,
                                    tasks, kinds, paths)
                event("created", f"{len(tasks)} bounded tasks")
            row = self.journal.mission(ident)
            if (row["workspace"] != str(self.workspace)
                    or row["model"] != self.model or row["endpoint"] != self.endpoint):
                raise ValueError("Mission workspace/model/endpoint mismatch")
            if row["state"] == "verified":
                return self.report(ident)
            previous_repair = row["state"] == "repairing"
            tasks = parse_plan(json.loads(row["plan"]))
            kinds = tuple(json.loads(row["checks"]))
            paths = tuple(json.loads(row["artifacts"]))
            gate = BoundedSemaphore(model_slots)
            # A single shared workspace writer avoids file-edit races.
            parallel = 1 if allow_writes else workers

            def worker(task, assigned):
                if stop.is_set():
                    return "PAUSED: cancelled"
                record = self.journal.task(ident, task.task_id)
                if record["state"] == "succeeded":
                    return record["output"]
                if record["state"] == "uncertain":
                    return "UNCERTAIN: manual resolution required"
                session = f"m_{ident}_{task.task_id}"
                agent = self._agent(gate, writes=allow_writes,
                                    tests=allow_tests, steps=max_steps)
                try:
                    self.journal.set_task(ident, task.task_id, "running")
                    event("task_started", task.task_id)
                    saved = agent.store.load(session)
                    answer = (agent.run(session, None, resume=True) if saved else
                              agent.run(session, assigned + "\n\nMission goal:\n" + row["goal"]))
                    if answer.startswith("UNCERTAIN"):
                        self.journal.set_task(ident, task.task_id, "uncertain", error=answer)
                        event("needs_review", f"{task.task_id}: uncertain tool call")
                    elif answer.startswith("PAUSED:"):
                        self.journal.set_task(ident, task.task_id, "paused", error=answer)
                        event("paused", task.task_id)
                    else:
                        self.journal.set_task(ident, task.task_id, "succeeded", output=answer)
                        event("task_succeeded", task.task_id)
                    return answer
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {str(exc)[:180]}"
                    self.journal.set_task(ident, task.task_id, "failed", error=detail)
                    event("task_failed", task.task_id + ": " + detail)
                    raise
                finally:
                    agent.store.conn.close()

            if stop.is_set():
                self.journal.set_state(ident, "cancelled")
                event("cancelled", "Stop requested")
                return self.report(ident)
            self.journal.set_state(ident, "executing")
            event("executing", f"{len(tasks)} tasks, {parallel} worker(s), "
                  f"{model_slots} model slot(s)")
            outcomes = TeamScheduler(parallel).run(tasks, worker)
            if stop.is_set():
                self.journal.set_state(ident, "cancelled")
                event("cancelled", "Stopped after active model calls")
                return self.report(ident)
            if any(o.status != "succeeded" for o in outcomes):
                self.journal.set_state(ident, "needs_review")
                event("needs_review", "Blocked, failed or uncertain agent task")
                return self.report(ident)
            self.journal.set_state(ident, "verifying")
            verdict = check_acceptance(self.workspace, kinds, paths,
                                       allowed=allow_tests, stop=stop)
            self.journal.set_state(ident, "verifying", verdict=verdict)
            event("verification", verdict["reason"])
            if verdict["passed"]:
                self.journal.set_state(ident, "verified", verdict=verdict)
                event("verified", "Observable acceptance checks passed")
                return self.report(ident)
            if not kinds and not paths:
                self.journal.set_state(ident, "unverified", verdict=verdict)
                event("unverified", "No external acceptance criteria")
                return self.report(ident)

            while not stop.is_set():
                saved = self.journal.mission(ident)
                attempts = int(saved["attempts"])
                if not (allow_writes and allow_tests):
                    self.journal.set_state(ident, "needs_permission", verdict=verdict)
                    event("needs_permission", "Repair requires explicit writes AND tests")
                    break
                if not previous_repair and attempts >= max_repairs:
                    self.journal.set_state(ident, "unverified", verdict=verdict)
                    event("unverified", "Repair budget exhausted")
                    break
                if previous_repair:
                    previous_repair = False
                else:
                    attempts += 1
                    self.journal.set_state(ident, "repairing", attempts=attempts,
                                           verdict=verdict)
                    event("repair_started", f"Repair {attempts}/{max_repairs}")
                session = f"m_{ident}_repair_{attempts}"
                agent = self._agent(gate, writes=True, tests=True, steps=max_steps)
                try:
                    saved_session = agent.store.load(session)
                    task_text = (
                        "A requested acceptance check failed. Diagnose, modify only "
                        "authorized workspace files, and retry permitted checks. "
                        "Do not assert success without evidence.\nMission:\n"
                        + row["goal"] + "\nVerification:\n"
                        + json.dumps(verdict, ensure_ascii=False)[:8000])
                    answer = (agent.run(session, None, resume=True) if saved_session
                              else agent.run(session, task_text))
                    if answer.startswith(("UNCERTAIN", "PAUSED:")):
                        self.journal.set_state(ident, "needs_review", verdict=verdict)
                        event("needs_review", "Repair has an uncertain or paused tool call")
                        break
                except Exception as exc:
                    self.journal.set_state(ident, "needs_review", verdict=verdict)
                    event("repair_failed", f"{type(exc).__name__}: {exc}"[:200])
                    break
                finally:
                    agent.store.conn.close()
                verdict = check_acceptance(self.workspace, kinds, paths,
                                           allowed=True, stop=stop)
                self.journal.set_state(ident, "verifying", verdict=verdict)
                event("verification", verdict["reason"])
                if verdict["passed"]:
                    self.journal.set_state(ident, "verified", verdict=verdict)
                    event("verified", "Requested criteria passed after repair")
                    break
            if stop.is_set():
                self.journal.set_state(ident, "cancelled", verdict=verdict)
                event("cancelled", "Stopped during repair")
            return self.report(ident)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Durable ironasu Mission Controller")
    p.add_argument("goal", nargs="?")
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--mission-id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--plan-file", type=Path)
    p.add_argument("--endpoint", default=os.getenv(
        "IRONASU_BACKEND_URL", "http://127.0.0.1:1234/v1"))
    p.add_argument("--model", default=os.getenv(
        "IRONASU_QUALITY_MODEL", "qwen3.5-4b-weight-space-programming"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model-slots", type=int, default=2)
    p.add_argument("--verify", action="append", choices=sorted(_CHECKS), default=[])
    p.add_argument("--accept-file", action="append", default=[])
    p.add_argument("--allow-writes", action="store_true")
    p.add_argument("--allow-tests", action="store_true")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--no-auto-plan", action="store_true")
    a = p.parse_args(argv)
    try:
        controller = MissionController(
            a.workspace, endpoint=a.endpoint, model=a.model,
            api_key=os.getenv("IRONASU_BACKEND_KEY", ""))
        if a.inspect:
            if not a.mission_id:
                p.error("--inspect requires --mission-id")
            print(json.dumps({"report": controller.report(a.mission_id).as_dict(),
                              "events": controller.journal.events(a.mission_id)},
                             ensure_ascii=False, indent=2))
            return 0
        if a.resume and not a.mission_id:
            p.error("--resume requires --mission-id")
        if a.resume and a.goal:
            p.error("Do not specify a new goal when resuming")
        plan = json.loads(a.plan_file.read_text(encoding="utf-8")) if a.plan_file else None
        report = controller.run(
            goal=a.goal, mission_id=a.mission_id, resume=a.resume,
            plan=plan, workers=a.workers, model_slots=a.model_slots,
            checks=tuple(a.verify) if a.verify else ("compileall", "git_diff_check"),
            accept_files=a.accept_file, max_steps=a.max_steps,
            max_repairs=a.max_repairs, allow_writes=a.allow_writes,
            allow_tests=a.allow_tests, auto_plan=not a.no_auto_plan,
            progress=lambda e: print(f"[{e['kind']}] {e['detail']}", file=sys.stderr))
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.verified else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ironasu mission: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
