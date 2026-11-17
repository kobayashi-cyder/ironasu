"""Opt-in durable, parallel Git worktree missions.

Git worktrees isolate *tracked file changes*, NOT arbitrary process effects.
Only use for trusted repositories. Nothing writes to the user's checked-out
branch without a separate explicit approval action.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
import subprocess
from threading import BoundedSemaphore, Event
from typing import Any

from .mission import MissionController, MissionReport, _exclusive, check_acceptance, parse_plan
from .team import TeamScheduler, TeamTask

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
_TASK = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_PRIVATE = frozenset((".git", ".ironasu", ".ssh", "node_modules", "__pycache__"))


class GitConflict(RuntimeError):
    """A merge conflict requires human review; never auto-resolve it."""


class GitWorktrees:
    def __init__(self, root: Path, ident: str):
        if not _SAFE.fullmatch(ident):
            raise ValueError("Invalid mission ID")
        self.root = root.resolve(strict=True)
        self.ident = ident
        self.folder = self.root / ".ironasu" / "worktrees" / ident
        self.folder.mkdir(parents=True, exist_ok=True)
        self.no_hooks = self.root / ".ironasu" / "no-git-hooks"
        self.no_hooks.mkdir(parents=True, exist_ok=True)
        result = self.git(self.root, "rev-parse", "--show-toplevel")
        if Path(result).resolve() != self.root:
            raise ValueError("Isolated missions require the Git working-tree root")

    def git(self, cwd: Path, *args: str, check: bool = True) -> str:
        env = os.environ.copy()
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true",
                    "GIT_AUTHOR_NAME": "ironasu mission",
                    "GIT_AUTHOR_EMAIL": "ironasu@localhost.invalid",
                    "GIT_COMMITTER_NAME": "ironasu mission",
                    "GIT_COMMITTER_EMAIL": "ironasu@localhost.invalid"})
        cmd = ["git", "-c", "core.hooksPath=" + str(self.no_hooks),
               "-c", "commit.gpgsign=false", "-C", str(cwd), *args]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=90, env=env, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Git operation failed: " + type(exc).__name__) from exc
        if result.returncode and check:
            raise GitConflict("Git " + args[0] + " failed: " +
                              (result.stderr or result.stdout).strip()[:600])
        return result.stdout.rstrip("\r\n")

    def clean_main(self) -> None:
        raw = self.git(self.root, "status", "--porcelain", "--untracked-files=all")
        # The mission's own journals and worktrees are never source content.
        unexpected = [line for line in raw.splitlines()
                      if not line[3:].replace("\\", "/").startswith(".ironasu/")]
        if unexpected:
            raise ValueError("Refusing to modify a dirty repository. Commit/stash existing "
                             "work first: " + ", ".join(unexpected[:4]))
        if self.git(self.root, "rev-parse", "--is-shallow-repository") == "true":
            raise ValueError("Shallow Git repositories are not yet supported")

    def base(self) -> str:
        return self.git(self.root, "rev-parse", "HEAD")

    def path(self, name: str) -> Path:
        if name not in ("candidate", "baseline") and not _TASK.fullmatch(name):
            raise ValueError("Invalid task ID")
        return self.folder / name

    def checkout(self, name: str, commit: str) -> Path:
        target = self.path(name)
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("Invalid Git commit ID")
        if target.exists():
            if not target.is_dir() or Path(
                    self.git(target, "rev-parse", "--show-toplevel")).resolve() != target.resolve():
                raise RuntimeError("Existing task directory is not its expected worktree")
        else:
            self.git(self.root, "worktree", "add", "--detach", str(target), commit)
        return target

    def head(self, cwd: Path) -> str:
        return self.git(cwd, "rev-parse", "HEAD")

    def merge(self, cwd: Path, commit: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("Invalid Git commit ID")
        common = self.git(cwd, "merge-base", "HEAD", commit)
        if common == commit:
            return
        if self.git(cwd, "status", "--porcelain", "--untracked-files=all"):
            raise GitConflict("Task worktree has uncommitted changes before merge")
        self.git(cwd, "merge", "--no-edit", "--no-ff", commit)

    def commit(self, cwd: Path, *, label: str) -> str:
        # The harness journal is always local to the worktree and excluded.
        # Avoid passing the private journal directory as a pathspec:
        # Git errors if .ironasu is gitignored, and could stage private data
        # if it is not. Enumerate and screen source paths instead.
        tracked = self.git(cwd, "diff", "--name-only", "-z").split("\x00")
        untracked = self.git(cwd, "ls-files", "--others", "--exclude-standard",
                             "-z").split("\x00")
        pre_staged = self.git(cwd, "diff", "--cached", "--name-only", "-z").split("\x00")
        paths = list(dict.fromkeys(
            name for name in (*tracked, *untracked, *pre_staged)
            if name and Path(name).parts[0] != ".ironasu"))
        if len(paths) > 100:
            raise ValueError("Too many changed files in one task")
        total = 0
        for name in paths:
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts
                    or any(part in _PRIVATE or part.startswith(".env") for part in relative.parts)
                    or any(relative.name.lower().endswith(ext)
                           for ext in (".pem", ".key"))):
                raise ValueError("Protected Git path in task changes: " + name)
            path = (cwd / relative)
            if not path.is_file() or path.is_symlink():
                raise ValueError("Deletion, symlink, directory or unsupported mode: " + name)
            size = path.stat().st_size
            total += size
            if size > 2_000_000 or total > 10_000_000:
                raise ValueError("Task changes exceed the bounded staging budget")
        if paths:
            self.git(cwd, "add", "--all", "--", *paths)
        self.git(cwd, "diff", "--cached", "--check")
        if not paths:
            return self.head(cwd)
        self.git(cwd, "commit", "-m", "ironasu isolated task: " + label)
        return self.head(cwd)

    def delta(self, base: str, tip: str) -> dict[str, Any]:
        changed = self.git(self.root, "diff", "--name-only", base, tip).splitlines()
        return {"base": base, "candidate": tip, "changed_files": changed[:100],
                "file_count": len(changed),
                "diff_sha256": sha256(self.git(self.root, "diff", "--binary", base, tip)
                                      .encode("utf-8")).hexdigest()}


class IsolatedMission:
    def __init__(self, controller: MissionController):
        self.controller = controller
        self.workspace = controller.workspace
        self.journal = controller.journal
        with self.journal.db() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS mission_isolation (
                mission_id TEXT PRIMARY KEY, base TEXT NOT NULL, candidate TEXT,
                state TEXT NOT NULL, baseline TEXT NOT NULL DEFAULT '{}',
                last_verdict TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS mission_tips (
                mission_id TEXT NOT NULL, task_id TEXT NOT NULL, tip TEXT NOT NULL,
                PRIMARY KEY (mission_id, task_id));""")

    def meta(self, ident: str) -> dict[str, Any] | None:
        with self.journal.db() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM mission_isolation WHERE mission_id=?", (ident,)).fetchone()
            return dict(row) if row else None

    def save(self, ident: str, *, state: str | None = None,
             candidate: str | None = None, baseline: Any = None, last: Any = None) -> None:
        fields = []
        args: list[Any] = []
        for field, val in (("state", state), ("candidate", candidate),
                           ("baseline", json.dumps(baseline) if baseline is not None else None),
                           ("last_verdict", json.dumps(last) if last is not None else None)):
            if val is not None:
                fields.append(field + "=?")
                args.append(val)
        if not fields:
            return
        with self.journal.db() as db:
            db.execute("UPDATE mission_isolation SET " + ",".join(fields) +
                       " WHERE mission_id=?", (*args, ident))

    def tip(self, ident: str, task: str) -> str | None:
        with self.journal.db() as db:
            row = db.execute("SELECT tip FROM mission_tips WHERE mission_id=? AND task_id=?",
                             (ident, task)).fetchone()
            return row[0] if row else None

    def store_tip(self, ident: str, task: str, commit: str) -> None:
        with self.journal.db() as db:
            db.execute("INSERT OR REPLACE INTO mission_tips VALUES(?,?,?)",
                       (ident, task, commit))

    def report(self, ident: str) -> dict[str, Any]:
        data = self.controller.report(ident).as_dict()
        meta = self.meta(ident)
        if meta:
            data["isolation"] = {
                "base": meta["base"], "candidate": meta["candidate"],
                "state": meta["state"], "baseline": json.loads(meta["baseline"]),
                "verification": json.loads(meta["last_verdict"]),
                "worktrees": str((self.workspace / ".ironasu" / "worktrees" / ident)),
            }
        return data

    def run(self, *, goal: str | None, mission_id: str | None,
            plan: Any, checks: tuple[str, ...], accept_files: tuple[str, ...],
            workers: int, model_slots: int, max_steps: int, max_repairs: int,
            allow_writes: bool, allow_tests: bool,
            cancel: Event | None, progress: Any, resume: bool, auto_plan: bool
            ) -> MissionReport:
        if not (allow_writes and allow_tests):
            raise ValueError("Isolated editing requires explicit write AND test permissions")
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("workers must be 1..16")
        if type(model_slots) is not int or not 1 <= model_slots <= 16:
            raise ValueError("model_slots must be 1..16")
        if type(max_steps) is not int or not 1 <= max_steps <= 100:
            raise ValueError("max_steps must be 1..100")
        if type(max_repairs) is not int or not 0 <= max_repairs <= 5:
            raise ValueError("max_repairs must be 0..5")
        if not checks or any(x not in ("unittest", "pytest", "compileall", "git_diff_check")
                             for x in checks):
            raise ValueError("Isolated mode requires supported verification checks")
        if len(accept_files) > 16 or any(
                not isinstance(x, str) or not x or Path(x).is_absolute()
                or ".." in Path(x).parts
                or any(part in _PRIVATE for part in Path(x).parts)
                for x in accept_files):
            raise ValueError("Invalid output path")
        ident = mission_id or uuid4().hex[:16]
        if not _SAFE.fullmatch(ident):
            raise ValueError("Invalid mission ID")
        stop = cancel or Event()
        git = GitWorktrees(self.workspace, ident)

        def event(kind: str, value: str) -> None:
            self.journal.event(ident, kind, value)
            if progress:
                progress({"mission_id": ident, "kind": kind, "detail": value[:3000]})

        with _exclusive(self.workspace, ident):
            if not resume:
                git.clean_main()
                if not isinstance(goal, str) or not goal.strip() or len(goal) > 8000:
                    raise ValueError("Mission goal must be 1..8000 characters")
                tasks = (parse_plan(plan) if plan is not None else
                         self.controller.propose(goal) if auto_plan else
                         (TeamTask("work", goal),))
                if any(t.task_id in ("candidate", "baseline") for t in tasks):
                    raise ValueError("Reserved task name in isolated worktree plan")
                base = git.base()
                self.journal.create(ident, goal, self.controller.endpoint,
                                    self.controller.model, tasks, checks, accept_files)
                with self.journal.db() as db:
                    db.execute("INSERT INTO mission_isolation(mission_id,base,state) "
                               "VALUES(?,?,?)", (ident, base, "planned"))
                event("isolated_created", "Git HEAD " + base[:12])
            row = self.journal.mission(ident)
            meta = self.meta(ident)
            if meta is None:
                raise RuntimeError("Isolated journal missing; do not resume without inspection")
            if (row["workspace"] != str(self.workspace) or
                    row["endpoint"] != self.controller.endpoint or
                    row["model"] != self.controller.model):
                raise ValueError("Workspace/model/endpoint differs from stored mission")
            if meta["state"] in ("integrating", "integration_pending", "integration_needs_review"):
                raise RuntimeError("An interrupted Git integration needs manual review")
            if meta["state"] in ("verified_isolated", "integrated"):
                return self.controller.report(ident)
            tasks = parse_plan(json.loads(row["plan"]))
            checks = tuple(json.loads(row["checks"]))
            accept_files = tuple(json.loads(row["artifacts"]))
            base = meta["base"]
            gate = BoundedSemaphore(model_slots)
            if stop.is_set():
                self.journal.set_state(ident, "cancelled")
                event("cancelled", "Stopped before execution")
                return self.controller.report(ident)
            if not json.loads(meta["baseline"]):
                baseline_dir = git.checkout("baseline", base)
                baseline = check_acceptance(
                    baseline_dir, checks, accept_files, allowed=True, stop=stop)
                self.save(ident, baseline=baseline)
                event("baseline", "Baseline checks passed: " + str(baseline["passed"]))
            else:
                baseline = json.loads(meta["baseline"])
            if stop.is_set():
                self.journal.set_state(ident, "cancelled")
                return self.controller.report(ident)

            def worker(task: TeamTask, assigned: str) -> str:
                if stop.is_set():
                    return "PAUSED: stopped"
                record = self.journal.task(ident, task.task_id)
                if record["state"] == "succeeded":
                    if not self.tip(ident, task.task_id):
                        raise RuntimeError("Missing committed task tip; needs human review")
                    return record["output"]
                task_dir = git.checkout(task.task_id, base)
                # Dependency results are included as real Git commits rather
                # than only textual summaries. Concurrent branches never edit
                # the same working directory.
                if record["state"] == "pending":
                    for dep in task.depends_on:
                        parent_tip = self.tip(ident, dep)
                        if not parent_tip:
                            raise RuntimeError("Missing prerequisite commit " + dep)
                        git.merge(task_dir, parent_tip)
                else:
                    # An interrupted merge can leave unresolved index state.
                    if git.git(task_dir, "ls-files", "--unmerged"):
                        raise GitConflict("Interrupted or conflicting task merge")
                agent = self.controller._agent(gate, writes=True, tests=True,
                                               steps=max_steps, workspace=task_dir)
                sid = f"m_{ident}_{task.task_id}"
                try:
                    saved = agent.store.load(sid)
                    if record["state"] == "uncertain" and saved is None:
                        return "UNCERTAIN: task journal missing"
                    self.journal.set_task(ident, task.task_id, "running")
                    event("task_started", task.task_id)
                    answer = (agent.run(sid, None, resume=True, cancel=stop) if saved else
                              agent.run(sid, assigned + "\nMission goal:\n" + row["goal"], cancel=stop))
                    if answer.startswith("UNCERTAIN"):
                        self.journal.set_task(ident, task.task_id, "uncertain", error=answer)
                        return answer
                    if answer.startswith("PAUSED:"):
                        self.journal.set_task(ident, task.task_id, "paused", error=answer)
                        return answer
                    tip = git.commit(task_dir, label=task.task_id)
                    self.store_tip(ident, task.task_id, tip)
                    self.journal.set_task(ident, task.task_id, "succeeded", output=answer)
                    event("task_committed", task.task_id + ": " + tip[:12])
                    return answer
                except Exception as exc:
                    self.journal.set_task(ident, task.task_id, "failed",
                                          error=f"{type(exc).__name__}: {exc}"[:180])
                    raise
                finally:
                    agent.store.conn.close()

            self.journal.set_state(ident, "executing_isolated")
            outcomes = TeamScheduler(workers).run(tasks, worker)
            if stop.is_set():
                self.journal.set_state(ident, "cancelled")
                event("cancelled", "After active requests completed")
                return self.controller.report(ident)
            if any(result.status != "succeeded" for result in outcomes):
                self.journal.set_state(ident, "needs_review")
                event("needs_review", "Agent task failed; no changes applied to main")
                return self.controller.report(ident)

            # Stage branch merging and verification are distinct from the
            # checked-out user branch. No implicit push, commit or merge there.
            if meta["state"] not in ("candidate_ready", "repairing_stage", "repair_paused_stage"):
                self.save(ident, state="integrating")
                stage = git.checkout("candidate", base)
                parents = {dep for task in tasks for dep in task.depends_on}
                terminals = [task for task in tasks if task.task_id not in parents]
                try:
                    for task in terminals:
                        tip = self.tip(ident, task.task_id)
                        if not tip:
                            raise RuntimeError("Missing terminal task tip")
                        git.merge(stage, tip)
                except GitConflict as exc:
                    self.journal.set_state(ident, "merge_conflict")
                    self.save(ident, state="integration_needs_review")
                    event("merge_conflict", str(exc))
                    return self.controller.report(ident)
                self.save(ident, state="candidate_ready", candidate=git.head(stage))
            else:
                stage = git.checkout("candidate", base)
            current = git.head(stage)
            resume_repair = meta["state"] in (
                "repairing_stage", "repair_paused_stage", "repair_uncertain_stage")
            if resume_repair:
                # The repair worktree may contain incomplete/uncommitted edits.
                # Never validate dirty files and label an earlier commit as
                # passing. First resume the exact harness journal session.
                verdict = json.loads(meta["last_verdict"])
                if not verdict:
                    raise RuntimeError("Interrupted repair has no stored verifier evidence")
            else:
                # Only committed content is eligible for verification.
                dirty = [line for line in git.git(
                    stage, "status", "--porcelain", "--untracked-files=all").splitlines()
                    if not line[3:].replace("\\", "/").startswith(".ironasu/")]
                if dirty:
                    self.journal.set_state(ident, "needs_review")
                    event("needs_review", "Candidate contains uncommitted changes")
                    return self.controller.report(ident)
                delta = git.delta(base, current)
                if not delta["file_count"]:
                    self.journal.set_state(ident, "unverified")
                    event("unverified", "No source change to review")
                    return self.controller.report(ident)
                # Check the actual committed patch relative to the original
                # baseline, not the clean candidate index.
                try:
                    git.git(stage, "diff", "--check", base, current)
                except GitConflict as exc:
                    self.journal.set_state(ident, "unverified")
                    event("patch_failed", str(exc)[:400])
                    return self.controller.report(ident)
                candidate = check_acceptance(stage, checks, accept_files,
                                             allowed=True, stop=stop)
                verdict = {"passed": candidate["passed"], "reason": candidate["reason"],
                           "checks": candidate["checks"], "artifacts": candidate["artifacts"],
                           "baseline": baseline, "delta": delta,
                           "candidate": candidate}
                self.save(ident, candidate=current, last=verdict)
                self.journal.set_state(ident, "verifying", verdict=verdict)
                if candidate["passed"]:
                    self.save(ident, state="verified_isolated")
                    self.journal.set_state(ident, "verified_isolated", verdict=verdict)
                    event("verified_isolated", "Passing candidate: " + current[:12] +
                          "; explicit approval required to integrate")
                    return self.controller.report(ident)

            # Repair only the integration candidate. Failed or uncertain
            # model tools are not replayed automatically by the harness.
            while not stop.is_set():
                record = self.journal.mission(ident)
                n = int(record["attempts"])
                resuming_repair = resume_repair or (record["state"] in
                    ("repairing_stage", "repair_paused_stage", "repair_uncertain_stage"))
                resume_repair = False
                if n >= max_repairs and not resuming_repair:
                    break
                if not resuming_repair:
                    n += 1
                    self.journal.set_state(ident, "repairing_stage", attempts=n,
                                           verdict=verdict)
                    self.save(ident, state="repairing_stage")
                task_id = f"m_{ident}_repair_{n}"
                agent = self.controller._agent(gate, writes=True, tests=True,
                                               steps=max_steps, workspace=stage)
                try:
                    saved = agent.store.load(task_id)
                    if resuming_repair and saved is None:
                        self.journal.set_state(ident, "repair_uncertain_stage", verdict=verdict)
                        self.save(ident, state="repair_uncertain_stage")
                        break
                    content = ("Repair failed candidate verification in this isolated "
                               "Git worktree. Keep changes minimal. The original Git "
                               "working directory must remain untouched.\nMission:\n"
                               + row["goal"] + "\nEvidence:\n" +
                               json.dumps(verdict, ensure_ascii=False)[:6000])
                    answer = (agent.run(task_id, None, resume=True, cancel=stop) if saved else
                              agent.run(task_id, content, cancel=stop))
                    if answer.startswith("UNCERTAIN"):
                        self.journal.set_state(ident, "repair_uncertain_stage", verdict=verdict)
                        self.save(ident, state="repair_uncertain_stage")
                        break
                    if answer.startswith("PAUSED:"):
                        self.journal.set_state(ident, "repair_paused_stage", verdict=verdict)
                        self.save(ident, state="repair_paused_stage")
                        break
                    current = git.commit(stage, label="repair-" + str(n))
                finally:
                    agent.store.conn.close()
                if not git.delta(base, current)["file_count"]:
                    break
                try:
                    git.git(stage, "diff", "--check", base, current)
                except GitConflict as exc:
                    self.journal.set_state(ident, "needs_review", verdict=verdict)
                    event("patch_failed", str(exc)[:300])
                    break
                candidate = check_acceptance(stage, checks, accept_files,
                                             allowed=True, stop=stop)
                verdict = {"passed": candidate["passed"], "reason": candidate["reason"],
                           "checks": candidate["checks"], "artifacts": candidate["artifacts"],
                           "baseline": baseline, "delta": git.delta(base, current),
                           "candidate": candidate}
                self.save(ident, candidate=current, last=verdict,
                          state="candidate_ready")
                self.journal.set_state(ident, "verifying", verdict=verdict)
                if candidate["passed"]:
                    self.save(ident, state="verified_isolated")
                    self.journal.set_state(ident, "verified_isolated", verdict=verdict)
                    event("verified_isolated", "Repaired candidate ready for review")
                    return self.controller.report(ident)
            if stop.is_set():
                self.journal.set_state(ident, "cancelled", verdict=verdict)
            elif self.journal.mission(ident)["state"] not in (
                    "repair_uncertain_stage", "repair_paused_stage"):
                self.journal.set_state(ident, "unverified", verdict=verdict)
            return self.controller.report(ident)

    def approve(self, ident: str) -> dict[str, Any]:
        """Separate, explicit, at-most-once integration into checked-out HEAD."""
        if not _SAFE.fullmatch(ident):
            raise ValueError("Invalid mission ID")
        with _exclusive(self.workspace, "integration"):
            with _exclusive(self.workspace, ident):
                meta = self.meta(ident)
                row = self.journal.mission(ident)
                if meta is None or row["state"] != "verified_isolated" or (
                        meta["state"] != "verified_isolated"):
                    raise ValueError("Only independently verified candidates can be approved")
                git = GitWorktrees(self.workspace, ident)
                git.clean_main()
                if git.base() != meta["base"]:
                    raise RuntimeError("Main HEAD changed after the candidate was created")
                candidate = meta["candidate"]
                if not candidate or git.git(self.workspace, "cat-file", "-t", candidate) != "commit":
                    raise RuntimeError("Candidate commit is not available")
                if git.git(self.workspace, "symbolic-ref", "--quiet", "--short", "HEAD") == "":
                    raise RuntimeError("Refusing to integrate on detached HEAD")
                stage = git.checkout("candidate", meta["base"])
                if git.head(stage) != candidate:
                    raise RuntimeError("Candidate worktree HEAD differs from verified commit")
                # Persist risk state first. A crash cannot lead to a silent
                # second merge. Operator must inspect after interruption.
                self.save(ident, state="integration_pending")
                self.journal.set_state(ident, "integration_pending")
                try:
                    git.git(self.workspace, "merge", "--ff-only", candidate)
                except Exception:
                    self.save(ident, state="integration_needs_review")
                    self.journal.set_state(ident, "integration_needs_review")
                    raise
                self.save(ident, state="integrated")
                self.journal.set_state(ident, "integrated")
                self.journal.event(ident, "integrated", candidate)
                return self.report(ident)
