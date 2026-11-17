# Execution isolation, aggregate quotas and cancellation

Use a Docker execution profile when project checks require OS isolation. The
profile mounts only a filtered, bounded copy of the selected project, disables
network access, makes the image root read-only and runs project code as UID/GID
65534. No host workspace directory, Docker socket, backend key, host environment,
Git hooks or Git configuration is exposed to project code.

Docker must be a local Linux engine (including Linux containers in Docker Desktop)
with enforced cgroup v2 CPU, memory, swap and PID controllers. An immutable,
installed image ID/digest is required; no image is pulled automatically. Images
declaring writable volumes and remote daemons are rejected. If isolation or any
required quota cannot be established, execution fails; it never falls back to
running the same check on the host.

## Budgets

| Budget | Default | Enforcement |
| --- | --- | --- |
| Combined container memory | 512 MiB | cgroup memory.max, including descendants and tmpfs pages |
| Swap | 0 | memory.swap.max |
| CPU bandwidth | 1 CPU | cgroup cpu.max across the container |
| Process/thread count | 64 | cgroup pids.max |
| General writable workspace | 256 MiB | Size-limited tmpfs; image and input snapshot are read-only |
| Shared memory | 8 MiB | Docker shm-size |
| Host input snapshot | 64 MiB / 10,000 entries / depth 64 | Bounded copy, including sanitized Git metadata |
| Retained command output | 128 KiB | Streaming bounded capture; excess fails the check |
| Check deadline | 60 s harness / 120 s mission | Host deadline plus independent container PID-1 watchdog |
| Concurrent sandbox containers per user | 1 | OS-released shared budget lock; waiting is cancellable |
| Model HTTP worker memory | 256 MiB | Windows aggregate Job Object / POSIX worker RLIMIT_AS |
| Model response / worker input | 1 MiB / 2 MiB | Bounded reads and serialization |

CPU, memory, PID and writable capacity apply to the whole active container,
including descendants. Staging and container admission are serialized per user
so concurrent agents do not multiply the configured active-container budget.
Read-only image storage, existing model files, the trusted Docker daemon and an
external model server are outside this execution budget.

The trusted root PID-1 verifies actual cgroup limits before project code starts.
Project code has no capabilities. The watchdog keeps only SETUID/SETGID/KILL so
it can drop project identity and enforce the deadline; project code cannot stop
the root watchdog. Normal exit, cancellation and overflow remove the container,
including detached descendants. After a host-client crash the watchdog still
expires; the next admitted run reaps old managed containers and staging files.
Cleanup/control operations have their own bounded timeout and an explicit error
if daemon cleanup cannot be confirmed.

## Configure

Build an operator-reviewed image containing Python 3 and any required project
dependencies. Git is needed for git_diff_check. The CI fixture is an example:

~~~sh
docker build -t ironasu-checks -f tests/sandbox.Dockerfile .
docker image inspect --format '{{.Id}}' ironasu-checks
~~~

Pass the returned immutable sha256 ID:

~~~sh
ironasu-harness "Inspect and test this project" --workspace . --session isolated \
  --allow-tests --sandbox-image sha256:YOUR_64_HEX_IMAGE_ID \
  --sandbox-memory-mb 512 --sandbox-cpus 1 --sandbox-disk-mb 256 --sandbox-pids 64

ironasu-mission "Implement the requested change" --workspace . \
  --allow-writes --allow-tests --sandbox-image sha256:YOUR_64_HEX_IMAGE_ID
~~~

Set IRONASU_SANDBOX_IMAGE to the immutable ID to apply the default profile to
native desktop missions too. Python callers can pass SandboxPolicy to Settings
or MissionController. Session and mission journals bind the execution policy:
resuming with changed quotas or without the original profile is refused.
Reading an already completed result does not execute anything.

Without a profile, the existing explicitly permitted trusted-local checks remain
available. They retain bounded output, timeout, cancellation and ordinary
descendant cleanup; choose the Docker profile for OS isolation and aggregate
resource enforcement. Snapshots deliberately reject symlinks/reparse points and
hard-linked project files, exclude common private/generated paths, and copy only
sanitized Git objects/refs/HEAD/index for Git checks, including linked worktrees.
Checks run on a disposable copy: their writes do not become host deliverables.

## In-flight model cancellation

OpenAI-compatible HTTP calls run in a private worker process. Cancellation closes
the connection and terminates the worker even while response headers/body stall;
waiting for a model slot is cancellable too. The harness pauses without creating
a partial assistant reply or dispatching pending tool effects, and can resume
the same session. CLI/desktop frozen EXEs use the same private worker protocol
without opening another window.

This interrupts ironasu's request and releases its local resources. An external
server's internal GPU computation is governed by that server's disconnect/abort
behavior; a generic chat/completions client cannot promise a server-side abort
acknowledgment that the server API does not provide.

## Evidence and security boundary

CI runs real container tests for identity, filesystem/network isolation, memory
OOM, CPU throttling, PID exhaustion, full writable capacity, output floods,
detached children, cancellation, aggregate admission and recovery after killing
the host client. Both HTTP and windowless EXE tests cancel an actual stalled
HTTP response. Missing Docker integration prerequisites cause an explicit skip
in the general stdlib suite; the dedicated sandbox CI supplies the image and
requires every real-container test.

The Docker engine, host OS, installed immutable image and snapshot source on the
host are trusted. Containers are an OS isolation boundary, not protection against
kernel vulnerabilities or an already compromised host. Test results remain
evidence for the tested program, not proof that arbitrary project code is correct.

Docker reference: https://docs.docker.com/reference/cli/docker/container/run/
