# ironasu v0.3: bounded multi-agent orchestrator

This is an **experimental local coding-agent orchestration layer**, not a hosted service or Grokbot-compatible replacement.

## Capabilities

- Up to **16 logical workers by default** (1–128 configured at first workspace initialization). Each worker is a distinct Harness session executing one DAG task at a time.
- Separate, **process-local** model request gate (`--inference 2`) and tool execution gate (`--tools 4`). A worker waiting for a tool does not hold a model permit; tasks can overlap without forcing 16 simultaneous LLM inference requests.
- SQLite WAL-backed task dependencies, atomic claims, priority aging (1 point/minute), submission idempotency keys, per-task lease tokens, heartbeats, and task results.
- Durable append-only shared event log (`topic`, <=400-character summary, optional original body + SHA-256), worker-specific incremental cursors, operator-readable event details, and short operator-maintained global brief.
- Completion and failure/uncertainty events automatically published. Event cursors are advanced **only** after a successful task which was actually started with those events in its prompt. For a resumed v0.2 Harness session, no fresh-event acknowledgement is claimed.
- Explicit **operator** recovery for uncertain/paused/failed tasks. Crashed workers are marked `uncertain`; the scheduler never blindly replays a task with possibly completed side effects.
- Tasks requiring a failed upstream task become blocked; operator recovery can unblock dependent DAG nodes.

## Install and use

Requires Python 3.10+; standard library only. Start an OpenAI-compatible model server separately (for example LocalAI or llama.cpp server). The orchestrator does not load large-model weights or accelerate GPU kernels.

```bash
python -m unittest discover -s tests -v
python -m compileall -q ironasu tests

# Submit a whole dependency graph atomically:
python -m ironasu.orchestrator --workspace . submit examples/orchestration_plan.json

# Inspect the task DAG, pending tasks, agent event lag:
python -m ironasu.orchestrator --workspace . status

# Run eligible tasks with default limits 16 workers / 2 simultaneous LLM requests / 4 tools:
IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1 \
  IRONASU_QUALITY_MODEL=qwen2.5-coder-7b \
  python -m ironasu.orchestrator --workspace . --allow-tests run

# Mutating source code is denied until separately opted in:
python -m ironasu.orchestrator --workspace . --allow-writes --allow-tests run
```

**Security:** `--allow-tests` executes project code, and `--allow-writes` edits project files. The v0.2 Harness implements workspace checks and permission gates but is **not an OS sandbox**. Use untrusted repositories only in an isolated VM/container with separately enforced file/network permissions. Never store secrets in shared events. Operators can review/edit the proposed plan before submission. Do not put credentials into plan files or environment inherited by executed checks.

A JSON plan is a list of tasks:

```json
[
  {"id":"inspect", "role":"investigate", "prompt":"Inspect the scheduler, report concerns", "priority":5,
   "idempotency_key":"demo-inspect-v1"},
  {"id":"fix", "role":"code", "prompt":"Fix the scheduler issues identified by inspect", "depends_on":["inspect"]},
  {"id":"verify", "role":"review", "prompt":"Run allowed checks and review the patch", "depends_on":["fix"]}
]
```

## Shared memory is delivery, not understanding

Every agent has an independent `cursor` into the same event journal. Event bodies are persisted **once**; workers are prompted with summaries instead of re-reading full bodies. The operator can publish an event or update a short global brief:

```bash
python -m ironasu.orchestrator publish code.changed "Scheduler API updated" --critical --body "Full change details here"
python -m ironasu.orchestrator brief "Project goal: keep RAM bounded; do not claim FLOP reductions"
python -m ironasu.orchestrator updates agent-01
python -m ironasu.orchestrator detail 1
python -m ironasu.orchestrator critical-status
```

When a human has checked events received by `updates`, they may manually acknowledge the delivered range. `ack` cannot jump beyond an event previously returned by the retrieval API, but an acknowledgement is **not** proof an LLM understood or remembered the event:

```bash
python -m ironasu.orchestrator ack agent-01 1
```

Shared context hierarchy:

- **L0** global operator brief, at most 4,000 characters, stored separately.
- **L1** unread event summaries + SHA-256 digests, size bounded by the event context budget.
- **L2** original event body retrieved explicitly with `detail` (not automatically injected into every LLM call).

If backlog cannot be delivered in the configured L1 budget, that task becomes `paused` and is *not* executed; cursors do not silently skip events. Operator should use `updates`/`detail`, compact essential facts into the brief, manually acknowledge verified events, then retry. Summary delivery is not lossless comprehension; there is no embedding search, LLM summarizer, automatic relevance selector, or guaranteed all-agent barrier yet. `critical-status` shows how many agents have not acknowledged each important event.

## Failure and recovery

A claimed task gets a random lease token, a renewable lease, and a unique Harness session (`orch-<task-id>`). A crashed process cannot prove whether model or tools already acted. After lease expiry, the task becomes `uncertain` instead of being automatically rerun. A stale/late worker cannot commit its old lease result.

```bash
python -m ironasu.orchestrator status
# First inspect changed files / tool session and verify effects.
python -m ironasu.orchestrator resolve TASK_ID retry
# Or declare verified completion without re-executing:
python -m ironasu.orchestrator resolve TASK_ID complete
# Or mark permanently failed:
python -m ironasu.orchestrator resolve TASK_ID fail
```

If the interrupted Harness session contains an uncertain tool call, use the existing `python -m ironasu --session orch-TASK_ID --resolve-call ... --resolve-result ...` workflow before retrying the orchestrator task. Retrying a completed Harness session resumes it; it does not repeat the task. Never assume exactly-once file/external side effects merely from an idempotency key. A late worker may still have applied changes before failing to commit. Persistent task claims and heartbeats are advisory coordination, **not** a distributed transaction over files, tests and HTTP.

## Operational boundaries

- **One orchestrator `run` process per model backend.** SQLite prevents duplicate task claims across processes, but the inference and tool semaphores are **process-local**; multiple independent runners can exceed the configured aggregate inference/tool limits. No cross-process GPU scheduler yet.
- This is a **run-until-idle CLI**, not a long-lived daemon, cron service, browser automation, MCP server, or sandbox.
- Up to 16 workers means 16 task/session slots, not 16 always-active GPU streams. Peak memory and throughput depend on the upstream server and hardware and must be benchmarked.
- SQLite WAL allows concurrent readers but one writer at a time. Task state and logs may grow indefinitely; no rotation, snapshotting, vacuum policy, or retention enforcement yet.
- The common brief and event summaries still incur prompt tokens **per worker**. Prefix KV-cache reuse depends on external inference-server support and identical tokenization/prefix structure; this repository does not implement KV-sharing.
- The shared event summary is limited; details require deliberate retrieval. Automatic summarization might lose essential facts, so original content is retained.
- Each task inherits the v0.2 Harness step budget and runs until it returns a terminal message or pauses. Inference error and uncertain results are not silently retried.

## Verification

`python -m unittest discover -s tests -v` includes: DAG ordering, cyclic/unknown dependency rejection, atomic/idempotent submit, failure blocking and unblocking, durable restart, lease-expiry fencing, operator recovery, event/ack correctness, backlog pause, two-semaphore enforcement with 16 fake-backed real Harness workers, and the v0.2 Harness security tests. No benchmark of a 70B/2.8T model or 16 independent live LLM endpoints is claimed.
