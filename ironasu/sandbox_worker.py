"""Trusted PID-1 quota verifier/watchdog. Project code runs as UID/GID 65534."""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def main():
    data = json.loads(sys.argv[1])
    policy = data["policy"]
    # Kernel timeout remains active even if the host client disappears.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(max(1, math.ceil(data["timeout"])))
    root = Path("/sys/fs/cgroup")
    memory = (root / "memory.max").read_text().strip()
    swap = (root / "memory.swap.max").read_text().strip()
    pids = (root / "pids.max").read_text().strip()
    cpu = (root / "cpu.max").read_text().split()
    if (memory == "max" or int(memory) > policy["memory_mb"] * 1024 * 1024
            or swap != "0" or pids == "max" or int(pids) > policy["pids"]
            or cpu[0] == "max" or int(cpu[0]) / int(cpu[1]) > policy["cpus"] + 0.00001):
        raise RuntimeError("Requested cgroup v2 limits are not enforced")
    # The image root is read-only; this tmpfs is the only general writable area.
    work = Path("/work")
    available = os.statvfs(work).f_blocks * os.statvfs(work).f_frsize
    if available > policy["disk_mb"] * 1024 * 1024:
        raise RuntimeError("Writable workspace exceeds the disk budget")
    target = work / "project"
    shutil.copytree("/input/project", target)
    for base, dirs, files in os.walk(target):
        os.chmod(base, 0o777)
        for name in files:
            path = Path(base) / name
            os.chmod(path, 0o666 | (path.stat().st_mode & 0o111))
    (work / "tmp").mkdir(mode=0o777)
    os.chmod(work / "tmp", 0o777)
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
           "HOME": "/work/tmp", "TMPDIR": "/work/tmp", "LANG": "C.UTF-8",
           "PYTHONUNBUFFERED": "1", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
    print("[ironasu sandbox: verified cgroup limits; isolated snapshot]", flush=True)
    command = data["argv"]
    if command[0] == "git":
        command = ["git", "-c", "safe.directory=/work/project", "-c", "core.fsmonitor=false",
                   "-c", "core.hooksPath=/nonexistent", *command[1:]]
    process = subprocess.Popen(command, cwd=target, env=env, stdin=subprocess.DEVNULL,
                               user=65534, group=65534, extra_groups=[])
    status = process.wait()
    events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
    if int(events.get("oom_kill", "0")):
        return 86
    # Exiting container PID 1 kills every remaining descendant, including setsid.
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("SANDBOX_ERROR " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(125)
