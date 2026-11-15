# ironasu

**Local coding-agent harness for OpenAI-compatible LLMs, with a roadmap toward adaptive inference and low-memory model serving.**

## What exists today (v0.2)

This repository currently provides a working Python **coding-agent harness**, including:

- A model → tool → model loop with bounded steps, persisted SQLite sessions and restart handling
- Workspace file read/search/list, permission-gated atomic writing/editing, and fixed test commands
- SHA256 preconditions for overwrites and explicit operator handling of crash-uncertain tools
- Session-scoped durable notes and bounded context-window trimming
- Standard-library-only runtime with unit tests and CI

See [Harness guide](docs/HARNESS.md) for setup, permissions, limitations and recovery steps.

## Quick start

```bash
python -m unittest discover -s tests -v
export IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1
export IRONASU_QUALITY_MODEL=qwen2.5-coder-7b
python -m ironasu "Inspect this project and identify the biggest bottleneck" --workspace . --session trial
```

Use `--allow-writes` to let the agent change project files; use `--allow-tests` to execute test/check commands. Checks can execute arbitrary code already in the project: **run untrusted projects in a dedicated OS/container sandbox.** The connected model server must support `/v1/chat/completions` and function tool calls. This package does not start or download a model.

## Project direction

- **Harness layer:** durable task execution, approvals, context, tool integrations, subagents, scheduled workflows and evaluation.
- **Inference gateway layer:** adaptive quality/fast routing, exact response cache, singleflight, backpressure, telemetry and optional backend speculative decoding.
- **Low-memory runtime layer:** profiling, AirLLM-style streaming experiments, careful CPU/GPU placement and I/O optimizations.

The gateway and low-memory runtime layers above are **planned, not implemented here**. Streaming huge weights from SSD reduces peak memory but does not intrinsically reduce the dense model's arithmetic per token. No claim of unmeasured inference speedups or Grokbot feature parity is made.

## License

MIT (a license file should be added before public distribution).
