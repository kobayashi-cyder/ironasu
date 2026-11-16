# ironasu v1.9 Mission Controller

The fourth Qt6 tab, **ミッション**, is a durable, bounded local coding
agent workflow. It does not embed a browser or run a general computer agent.

## v1.9: isolated parallel editing (opt-in)

A new **Git worktree isolation** option lets 1–16 agents edit independent
worktrees in parallel while a separate candidate worktree is verified against
the baseline. The checked-out branch remains unchanged until explicit
fast-forward approval. Both write and test permissions must be enabled,
and trusted Git repository roots are required.

This is **Git working-tree isolation, not an operating-system sandbox**.
Details, CLI flags, recovery limits and approval conditions are in
[Isolated Git missions](isolated-missions.md).

## Workflow

Goal → optional model-generated JSON DAG (1–8 tasks, maximum 16) →
dependency-aware agent sessions → acceptance checks and artifact hashes →
bounded repair attempts → evidence report.

Invalid model plans fall back to one task. A user-supplied plan file can
specify an array of task_id, prompt, depends_on and priority entries.

Up to 16 read-only workers may execute concurrently, with a separate
model-call concurrency cap (default 2). If file writes are explicitly
authorized, the system serializes workspace-writing workers to avoid races.
The system runs only known tools from the existing ironasu coding harness.

Two permissions are separate and OFF by default:
- File writes: creating or editing workspace text, using safe relative paths.
- Test execution: running unittest, pytest, compileall or git diff --check.
  These run project code, so only grant permission to trusted workspaces.

A model reporting "completed" does not imply success. Mission status becomes
verified only when all requested checks pass and expected artifacts exist.
A successful git diff --check detects formatting problems but is not proof
that requested functionality is correct. Test results and SHA-256 hashes for
artifacts <=16 MiB are included in the report.

Failed verification is followed by at most 2 repair passes (configurable
0–5), and only with both explicit write and test permissions. Unresolved
failure or uncertain side-effect status is surfaced, not automatically
reported as success.

## Desktop usage

1. Open the new **ミッション** tab in the Qt6 Studio window.
2. Select your repository folder and state the goal.
3. Choose worker count (1–16), model slots (1–16) and repair limit.
4. Authorize writes and tests separately, understanding that tests execute
   project code. The GUI asks for confirmation before either is enabled.
5. Start the mission and review the event log, status and checks.
6. Save the JSON report. To resume after restarting the app, select the
   same workspace and mission ID in saved missions.

The tab does not create state files simply by opening. Starting a mission
stores SQLite journals under workspace/.ironasu/missions.sqlite and
workspace/.ironasu/harness.sqlite. Each mission is protected by an OS
file lock, preventing concurrent duplicate runs under the same ID.

## PowerShell usage

The CLI is named ironasu-mission. Example using a currently running local
LM Studio model:

    cd C:\Users\hazim\ironasu
    .\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
    .\.venv\Scripts\ironasu-mission.exe "Fix failing tests" --workspace . --model qwen3.5-4b-weight-space-programming --workers 4 --model-slots 2 --allow-writes --allow-tests --verify unittest --max-repairs 2

By default, verification requires permission and may return needs_permission.
Reapprove permissions at each resume; they are not silently inherited.
To resume a saved mission, specify:

    ironasu-mission --mission-id YOUR_ID --workspace . --resume --model qwen3.5-4b-weight-space-programming --allow-writes --allow-tests

Inspect the report and audit events without running the mission:

    ironasu-mission --mission-id YOUR_ID --workspace . --inspect

## Resumability and safety

The stored DAG, task status and model messages are reused. If a tool call
began a side effect before a crash, the harness does NOT replay it
automatically. Review the workspace and resolve uncertain operations
manually through the existing harness CLI.

The model cannot issue arbitrary shell commands, browser actions, remote
deployment, account changes or Git pushes through this mission tool.
The GUI accepts only loopback local inference endpoints. The CLI can use
an explicit remote endpoint, so do not point it at an untrusted provider.

A stop request is cooperative; an in-flight model request may finish or
time out before the mission can end.

## Limits

This release does not prove parity with Grok Bot or provide an always-on
computer. A local notebook must remain on, and the model must support
sensible planning and tool calling. No automated browser/OS interaction,
isolated Git worktrees or semantic correctness oracle are supplied.

Model-generated task outputs are untrusted and are never treated as
independent proof of correctness. Compare measured success rates, manual
interventions, failed/uncertain recovery and wall time on the same tasks
before claiming superiority.

## Tests

Run Python standard-library tests:

    python -m unittest tests.test_mission -v
    python -m unittest tests.test_qt_desktop -v
