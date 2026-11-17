"""Fail-closed Docker execution with cgroup quotas and a private project snapshot."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from uuid import uuid4

from .execution import CommandResult, exclusive_lock, run_command


@dataclass(frozen=True)
class SandboxPolicy:
    image: str
    memory_mb: int = 512
    cpus: float = 1.0
    disk_mb: int = 256
    pids: int = 64
    snapshot_mb: int = 64

    def __post_init__(self):
        if not isinstance(self.image, str) or not re.fullmatch(
                r"(?:sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64})", self.image):
            raise ValueError("Select an installed Docker image by immutable sha256 ID/digest")
        for name, low, high in (("memory_mb", 128, 65536), ("disk_mb", 16, 65536),
                                ("snapshot_mb", 1, 1024), ("pids", 8, 4096)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid sandbox " + name)
        if not math.isfinite(self.cpus) or not 0.1 <= self.cpus <= 64:
            raise ValueError("Invalid sandbox CPU budget")
        if self.snapshot_mb >= self.disk_mb:
            raise ValueError("Snapshot budget must leave writable scratch space")


class Snapshot:
    def __init__(self, limit, stop=None, deadline=None):
        self.remaining = limit
        self.count = 0
        self.scanned = 0
        self.stop = stop
        self.deadline = deadline

    def check(self):
        if self.stop is not None and self.stop.is_set():
            raise InterruptedError("Snapshot cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Snapshot exceeded the execution deadline")

    def copy(self, source: Path, target: Path, *, private=True, depth=0, boundary=None):
        self.check()
        if depth > 64:
            raise ValueError("Snapshot exceeds directory depth budget")
        boundary = source.absolute() if boundary is None else boundary
        if not source.resolve().is_relative_to(boundary):
            raise ValueError("Snapshot directory changed or escaped its boundary")
        self.count += 1
        if self.count > 10000:
            raise ValueError("Snapshot exceeds entry count budget")
        info = source.lstat()
        if source.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Snapshot refuses symlinks/reparse points")
        if stat.S_ISDIR(info.st_mode):
            target.mkdir()
            with os.scandir(source) as entries:
                for raw_entry in entries:
                    self.check()
                    self.scanned += 1
                    if self.scanned > 10000:
                        raise ValueError("Snapshot exceeds scanned entry budget")
                    entry = Path(raw_entry.path)
                    name = entry.name.casefold()
                    if private and (name in {".git", ".ironasu", ".ssh", "__pycache__", "node_modules",
                                             ".aws", ".azure", ".docker", ".kube", ".netrc", ".npmrc", ".pypirc", ".gitconfig", ".git-credentials"}
                                    or name.startswith(".env") or name in {"credentials.json", "id_rsa", "id_ed25519"}
                                    or name.endswith((".key", ".pem"))):
                        continue
                    if not private and name in {"alternates", "http-alternates"}:
                        raise ValueError("External Git object stores are not allowed")
                    self.copy(entry, target / entry.name, private=private, depth=depth+1, boundary=boundary)
        elif stat.S_ISREG(info.st_mode):
            if private and info.st_nlink > 1:
                raise ValueError("Snapshot refuses hard-linked project files")
            if info.st_size > self.remaining:
                raise ValueError("Snapshot exceeds input budget")
            with source.open("rb") as reader:
                opened = os.fstat(reader.fileno())
                if not os.path.samestat(info, opened) or opened.st_size != info.st_size:
                    raise ValueError("Snapshot source changed while opening")
                with target.open("xb") as writer:
                    total = 0
                    while True:
                        self.check()
                        chunk = reader.read(min(1024 * 1024, self.remaining + 1))
                        if not chunk:
                            break
                        total += len(chunk)
                        self.remaining -= len(chunk)
                        if self.remaining < 0 or total > info.st_size:
                            raise ValueError("Snapshot source grew beyond budget")
                        writer.write(chunk)
                after = os.fstat(reader.fileno())
                if (after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns
                        or total != info.st_size):
                    raise ValueError("Snapshot source changed during copy")
            target.chmod(0o644 | (info.st_mode & 0o111))
        else:
            raise ValueError("Snapshot refuses device/socket/pipe entries")


def _git_snapshot(workspace: Path, target: Path, copier: Snapshot):
    marker = workspace / ".git"
    if not marker.exists():
        return
    if marker.is_symlink():
        raise ValueError("Git marker must not be a symlink")
    if marker.is_dir():
        state = common = marker.resolve()
    else:
        with marker.open("rb") as reader:
            text = reader.read(4097)
        if len(text) > 4096 or not text.startswith(b"gitdir: "):
            raise ValueError("Invalid Git worktree marker")
        state = (workspace / text[8:].decode("utf-8").strip()).resolve()
        common_marker = state / "commondir"
        if common_marker.is_file():
            with common_marker.open("rb") as reader:
                value = reader.read(4097)
            if len(value) > 4096:
                raise ValueError("Git common-directory marker exceeds budget")
            common = (state / value.decode("utf-8").strip()).resolve()
        else:
            common = state
    allowed = {(p / ".git").resolve() for p in (workspace, *workspace.parents)}
    if common not in allowed or not state.is_relative_to(common):
        raise ValueError("Git metadata is outside the project ancestry")
    target.mkdir()
    for name in ("objects", "refs", "packed-refs", "shallow"):
        source = common / name
        if source.exists():
            copier.copy(source, target / name, private=False)
    for name in ("HEAD", "index"):
        source = state / name
        if source.exists():
            copier.copy(source, target / name, private=False)
    (target / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfsmonitor = false\n",
                                   encoding="utf-8")


def _docker(args, *, cwd, timeout=15, stop=None):
    result = run_command(["docker", *args], cwd=cwd, timeout=timeout,
                         max_output_bytes=32768, stop=stop)
    if result.reason == "cancelled":
        raise InterruptedError("Docker control cancelled")
    if not result.passed:
        raise RuntimeError("Docker control failed: " + result.output[:1000] + " " + result.reason)
    return result.output.strip()


def _run_sandbox(argv, *, cwd: Path, timeout: float, policy: SandboxPolicy,
                stop=None, max_output_bytes=128 * 1024):
    """Only the sanitised snapshot is mounted. Never fall back to host execution."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Sandbox timeout must be finite and positive")
    if not argv or len(argv) > 256 or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("Invalid sandbox argv")
    if sum(len(arg) for arg in argv) > 20000:
        raise ValueError("Sandbox command exceeds argument budget")
    deadline = time.monotonic() + timeout
    if stop is not None and stop.is_set():
        return CommandResult(None, "", "cancelled", 0)
    if not shutil.which("docker"):
        raise RuntimeError("Sandbox requires a local Docker engine with Linux cgroup v2")
    host = os.getenv("DOCKER_HOST", "")
    if not host:
        host = _docker(["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                       cwd=cwd, timeout=min(15, timeout), stop=stop)
    if not (host.startswith("unix://") or host.startswith("npipe://")):
        raise RuntimeError("Sandbox refuses a remote Docker daemon")
    config = json.loads(_docker(["image", "inspect", "--format", "{{json .}}", policy.image], cwd=cwd, timeout=min(15, timeout), stop=stop))
    if config.get("Os") != "linux" or config.get("Config", {}).get("Volumes"):
        raise RuntimeError("Sandbox needs a Linux image without declared writable volumes")
    image = config.get("Id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("Docker did not return an immutable image ID")
    owner = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
    managed = Path(tempfile.gettempdir()) / ("ironasu-sandbox-" + owner)
    if managed.is_symlink():
        raise RuntimeError("Unsafe sandbox staging root")
    managed.mkdir(mode=0o700, exist_ok=True)
    label = "ironasu.owner=" + owner
    # One active sandbox per user also bounds aggregate container/staging budgets.
    with exclusive_lock(managed / "budget.lock", stop=stop,
                        timeout=max(0.001, deadline-time.monotonic())):
        stale = _docker(["ps", "-aq", "--filter", "label=" + label], cwd=cwd).splitlines()
        for ident in stale:
            if not re.fullmatch(r"[0-9a-f]{12,64}", ident):
                raise RuntimeError("Unexpected managed container ID")
            _docker(["rm", "-f", "-v", ident], cwd=cwd)
        for folder in managed.iterdir():
            if re.fullmatch(r"snapshot_[0-9a-f]{32}", folder.name):
                if folder.is_symlink() or not folder.resolve().is_relative_to(managed.resolve()):
                    raise RuntimeError("Unsafe stale snapshot")
                shutil.rmtree(folder)
        name = "ironasu-" + uuid4().hex
        stage = managed / ("snapshot_" + uuid4().hex)
        stage.mkdir(mode=0o700)
        created = False
        try:
            project = stage / "project"
            copier = Snapshot(policy.snapshot_mb * 1024 * 1024, stop=stop, deadline=deadline)
            copier.copy(cwd.resolve(), project)
            _git_snapshot(cwd.resolve(), project / ".git", copier)
            runner = Path(__file__).with_name("sandbox_worker.py")
            shutil.copyfile(runner, stage / "runner.py")
            stage.chmod(0o755)
            if "," in str(stage):
                raise ValueError("Docker mount paths must not contain commas")
            if stop is not None and stop.is_set():
                return CommandResult(None, "", "cancelled", 0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return CommandResult(None, "", "timeout", 0)
            payload = {"argv": ["python3" if arg == sys.executable and i == 0 else arg
                                 for i, arg in enumerate(argv)],
                       "timeout": remaining, "policy": asdict(policy)}
            args = ["create", "--name", name, "--label", label, "--pull", "never",
                    "--network", "none", "--read-only", "--user", "0:0",
                    "--no-healthcheck", "--workdir", "/work",
                    "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
                    "--cap-add", "KILL", "--security-opt", "no-new-privileges=true",
                    "--memory", str(policy.memory_mb) + "m", "--memory-swap", str(policy.memory_mb) + "m",
                    "--cpus", str(policy.cpus), "--pids-limit", str(policy.pids),
                    "--shm-size", "8m", "--log-driver", "none",
                    "--tmpfs", "/work:rw,nosuid,nodev,mode=1777,size=" + str(policy.disk_mb) + "m",
                    "--mount", "type=bind,src=" + str(stage) + ",dst=/input,readonly",
                    "--env", "OPENAI_API_KEY=", "--env", "IRONASU_BACKEND_KEY=",
                    "--entrypoint", "python3", image, "-I", "/input/runner.py", json.dumps(payload)]
            # Set before create: a timeout can leave creation outcome uncertain.
            created = True
            _docker(args, cwd=cwd, timeout=min(15, remaining), stop=stop)
            remaining = max(0.1, deadline-time.monotonic())
            result = run_command(["docker", "start", "--attach", name], cwd=cwd,
                                 timeout=remaining+5, stop=stop, max_output_bytes=max_output_bytes)
            status = json.loads(_docker(["inspect", "--format", "{{json .State}}", name], cwd=cwd))
            if result.reason == "exited":
                if status.get("Running"):
                    raise RuntimeError("Docker attach ended before container exit")
                result = CommandResult(status["ExitCode"], result.output, result.reason, result.output_bytes)
            if status.get("OOMKilled") or result.returncode == 86:
                return CommandResult(result.returncode, result.output, "memory_limit", result.output_bytes)
            if result.reason == "exited" and result.returncode == 124:
                return CommandResult(124, result.output, "timeout", result.output_bytes)
            return result
        finally:
            try:
                if created:
                    _docker(["rm", "-f", "-v", name], cwd=cwd)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)


def run_sandbox(argv, *, cwd: Path, timeout: float, policy: SandboxPolicy,
                stop=None, max_output_bytes=128 * 1024):
    try:
        return _run_sandbox(argv, cwd=cwd, timeout=timeout, policy=policy,
                            stop=stop, max_output_bytes=max_output_bytes)
    except InterruptedError:
        return CommandResult(None, "", "cancelled", 0)
    except TimeoutError:
        return CommandResult(None, "", "timeout", 0)


def add_sandbox_arguments(parser):
    parser.add_argument("--sandbox-image", default=os.getenv("IRONASU_SANDBOX_IMAGE"),
                        help="Installed Linux Docker image; enforce isolation and quotas, no host fallback")
    parser.add_argument("--sandbox-memory-mb", type=int, default=512)
    parser.add_argument("--sandbox-cpus", type=float, default=1.0)
    parser.add_argument("--sandbox-disk-mb", type=int, default=256)
    parser.add_argument("--sandbox-pids", type=int, default=64)


def policy_from_args(args):
    if not args.sandbox_image:
        return None
    return SandboxPolicy(args.sandbox_image, memory_mb=args.sandbox_memory_mb,
                         cpus=args.sandbox_cpus, disk_mb=args.sandbox_disk_mb,
                         pids=args.sandbox_pids,
                         snapshot_mb=min(64, max(1, args.sandbox_disk_mb // 2)))


def policy_from_environment():
    image = os.getenv("IRONASU_SANDBOX_IMAGE")
    return SandboxPolicy(image) if image else None
