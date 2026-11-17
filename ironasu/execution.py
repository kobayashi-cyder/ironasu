"""Bounded command lifecycle and OS-released locks for trusted local projects.

These controls are not an OS sandbox or a total CPU/RAM/disk quota.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
from threading import Event, Thread
import time
from typing import Mapping, Sequence


@contextmanager
def exclusive_lock(path: Path, *, timeout: float = 0):
    """Never unlink: removing a held lock can split concurrent ownership."""
    if os.name == "nt":
        raw = str(path.absolute())
        prefix = "\\\\?\\"
        if not raw.startswith(prefix):
            raw = (prefix + "UNC\\" + raw[2:]
                   if raw.startswith("\\\\") else prefix + raw)
        path = Path(raw)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+b") as fp:
        fp.seek(0, os.SEEK_END)
        if not fp.tell():
            fp.write(b"0")
            fp.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    fp.seek(0)
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Operation already active: " + path.name) from exc
                time.sleep(0.02)
        try:
            yield
        finally:
            if os.name == "nt":
                fp.seek(0)
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class _WindowsJob:
    """A kill-on-close Job Object owns ordinary descendant processes."""
    def __init__(self):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD),
                        ("scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes",
                         "write_bytes", "other_bytes")]

        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO),
                        ("process_mem", ctypes.c_size_t), ("job_mem", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int,
                                                        ctypes.c_void_p, w.DWORD]
        self.kernel.SetInformationJobObject.restype = w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            code = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(code)

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    output: str
    reason: str  # exited, timeout, cancelled, output_limit
    output_bytes: int  # Retained bytes, not total bytes emitted.

    @property
    def passed(self):
        return self.reason == "exited" and self.returncode == 0


def run_command(argv: Sequence[str], *, cwd: Path, timeout: float,
                stop: Event | None = None, max_output_bytes: int = 128 * 1024,
                env: Mapping[str, str] | None = None) -> CommandResult:
    """Drain combined output with bounded retention; stop the owned process tree."""
    if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes < 1:
        raise ValueError("Invalid command timeout/output budget")
    if stop is not None and stop.is_set():
        return CommandResult(None, "", "cancelled", 0)
    environment = dict(os.environ if env is None else env)
    for secret in ("IRONASU_BACKEND_KEY", "OPENAI_API_KEY"):
        environment.pop(secret, None)
    job = _WindowsJob() if os.name == "nt" else None
    process = None
    reader = None
    captured = bytearray()
    exceeded = Event()
    reason = "exited"

    def terminate_tree():
        if job is not None:
            job.close()
        elif process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                   if os.name == "nt" else {"start_new_session": True})
        process = subprocess.Popen(
            list(argv), cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=False, env=environment, **options)
        if job is not None:
            job.assign(process)

        def drain():
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    return
                remaining = max_output_bytes - len(captured)
                captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    exceeded.set()

        reader = Thread(target=drain, name="ironasu-check-output", daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        while True:
            if stop is not None and stop.is_set():
                reason = "cancelled"
                break
            if exceeded.is_set():
                reason = "output_limit"
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            if process.poll() is not None:
                break
            time.sleep(0.02)
    finally:
        terminate_tree()
        if process is not None:
            if process.poll() is None:
                process.kill()  # Also handles failed job assignment.
            process.wait(timeout=5)
            if reader is not None:
                reader.join(timeout=5)
            if reader is not None and reader.is_alive():
                raise RuntimeError("Command output pipe did not close after tree termination")
            process.stdout.close()
    if reason == "exited" and exceeded.is_set():
        reason = "output_limit"
    output = captured.decode("utf-8", "replace")
    if reason == "output_limit":
        output += "\n[output limit exceeded]"
    return CommandResult(process.returncode, output, reason, len(captured))
