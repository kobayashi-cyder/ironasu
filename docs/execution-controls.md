# Harness execution controls

The harness serializes each session with an OS-released advisory lock. A second
runner or manual uncertain-call resolver fails immediately while that session is
active. Different sessions remain concurrent. SQLite journal initialization is
serialized across processes to avoid first-open WAL contention. Locks are never
unlinked; the operating system releases ownership after a crash.

Started writes and checks still require operator reconciliation after interruption.
Read-only replay retains its explicit recovery annotation. Cancellation between
harness steps preserves pending calls for the same session's later resume.

## Checks

Harness checks retain their 60-second timeout; mission acceptance checks retain
120 seconds. Both use the same bounded runner:

- Combined stdout/stderr retains at most 128 KiB of raw bytes. Exceeding the limit
  stops the command and fails verification, even if its exit code is zero.
- Cancellation is polled during checks, rather than only between commands.
- Ordinary descendants are terminated on timeout, cancellation, output overflow,
  and after the parent exits. POSIX uses a new process group; Windows uses a
  kill-on-close Job Object. Failure to assign the Job Object fails closed.
- Commands have no stdin, use no shell and omit the two existing backend-key
  environment variables. Other inherited environment values remain available.
- Verification records the exit code and reason: exited, timeout, cancelled or
  output_limit. Only exited with code zero passes.

A reader thread drains the pipe without retaining unlimited output. Cleanup waits
up to five seconds each for the process and output reader. Windows checks use no
new console window. Invalid UTF-8 is decoded with replacement.

## Model and file boundaries

HTTP completion response bodies are capped at 1 MiB. The harness validates reply
shape, content type, call count (32), ID length (256), and session-wide call-ID
reuse before saving new pending calls. An unshrinkable task exceeding the context
budget fails before contacting the model. Timeout settings must be finite and
positive. A workspace alias is checked against the private-path policy after
resolution, and file reads retain only the configured file budget plus one byte.

## Limits

These are lifecycle controls for trusted projects, not a sandbox. They do not
provide total CPU, memory or disk quotas, rollback, global same-file coordination
between different sessions, or isolation against hostile code. POSIX children
that deliberately escape the process group and the Windows spawn-to-job-assignment
race require stronger OS isolation. A currently running model request is not
preempted by cancellation: the configured backend timeout/retry behavior still
applies, and cancellation is observed before the next harness step. Network
filesystems may not provide the required advisory-lock semantics.

## Regression checks

Run the standard unittest suite. The execution tests use actual subprocesses for
output flooding, timeout/cancellation, descendant cleanup, process-level lock
exclusion and crash recovery, plus concurrent journal initialization and malformed
model replies. CI also runs the relevant harness/mission suites on Windows.
