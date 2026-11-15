# ironasu harness v0.2 (local coding-agent foundation)

This is an **independent, small implementation** of a local coding harness. It is *not* Grokbot, not a clone of its proprietary implementation, and not yet a Grokbot-class always-on service.

## Supported

- OpenAI-compatible `/v1/chat/completions` tool-calling models (e.g. a compatible llama.cpp or LocalAI endpoint)
- Agent-model-tool feedback loop with a bounded step count and basic HTTP retries
- Persistent SQLite sessions, saved tool results and notes
- At-most-once **automatic** tool execution across restarts: an operation marked `started` is reported `UNCERTAIN` rather than repeated after a crash
- Read/list/search files, create/edit files (requires `--allow-writes`), and run fixed checks (requires `--allow-tests`)
- Workspace path confinement, private path exclusions, atomic writes, and optimistic SHA256 precondition on file replacement
- Context window trimming that preserves intact tool-call/result groups; older context is *omitted*, not accurately summarized

## Quick start

Python 3.10+; harness itself uses only Python standard library.

```bash
python -m unittest discover -s tests -v
export IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1
export IRONASU_QUALITY_MODEL=qwen2.5-coder-7b
python -m ironasu "Explain this repository" --workspace . --session demo
python -m ironasu "Fix the test failures" --workspace . --session demo --allow-writes --allow-tests
python -m ironasu --workspace . --session demo --resume --allow-writes --allow-tests
```

Models must support OpenAI-style structured tool calls. No model is downloaded or started by ironasu. The optional upstream key is `IRONASU_BACKEND_KEY`.

**Security:** This is *not an OS-level sandbox*. Running project tests executes project Python code with the current user's permissions. Only use `--allow-tests` for trusted projects; execute in Docker/VM as an additional boundary for unknown code. `--allow-writes` allows the agent to modify files under the configured workspace; back up / use Git. Path checks are defense-in-depth, not a complete adversarial filesystem sandbox. Do not expose the harness to untrusted users as a service.

## Crash recovery

After a failure, run `--resume` for the same workspace and session. If a tool call is marked `UNCERTAIN`, inspect the filesystem and decide its actual outcome. Then explicitly record the verified result and resume:

```bash
python -m ironasu --workspace . --session demo --resolve-call CALL_ID \
  --resolve-result "Verified: the file was written and checked" --allow-writes
```

The harness does **not** automatically replay started tools; operator resolution is required. This minimizes duplicate side effects, but does not guarantee exactly-once execution across operating-system crashes.

## Missing versus full Grokbot-class system

- No persistent background worker, scheduler, webhooks, wake-on-event or remote durable execution service
- No desktop/browser control, images, audio, messaging integrations, or native MCP
- No real execution sandbox, fine-grained interactive approval UI, or credential broker
- No subagents, team scheduling, task queues, or multi-model cost/rate accounting
- No model backend, AirLLM runtime, GPU offload, streaming server, or speculative decoding

The older README outlined a planned LLM inference proxy; its adaptive caching/routing server has not yet been implemented in the repository. The harness can connect to a compatible backend independently and can later connect to an ironasu proxy.
