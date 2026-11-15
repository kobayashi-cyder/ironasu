# ironasu

**Experimental local AI coding harness and bounded multi-agent orchestrator.**

ironasu originally targets compute-efficient use of large local models, including AirLLM-style weight paging. **The v0.3 implementation does not itself serve model weights, reduce dense inference FLOPs, or provide a GPU scheduler.** It coordinates calls to a separate OpenAI-compatible model backend.

## Implemented

- **v0.2 Harness:** persistent tool-call sessions, local code inspection, permission-gated edits/checks, manual crash-uncertainty resolution, bounded context. See [Harness guide](docs/HARNESS.md).
- **v0.3 Orchestrator:** default 16 logical agents, 2 simultaneous model requests, 4 simultaneous tool invocations *within one runner process*, durable dependency-aware task DAG, bounded priority aging, independent event cursors with explicit acknowledgement, a shared brief + event metadata, lease-fenced crash recovery, audited operator resolution, and unit/integration tests. See [Orchestrator guide](docs/ORCHESTRATOR.md).

### Quick start

```bash
python -m unittest discover -s tests -v
python -m compileall -q ironasu tests

python -m ironasu.orchestrator --workspace . submit examples/orchestration_plan.json
python -m ironasu.orchestrator --workspace . status

export IRONASU_BACKEND_URL=http://127.0.0.1:8080/v1
export IRONASU_QUALITY_MODEL=qwen2.5-coder-7b
python -m ironasu.orchestrator --workspace . run
```

Configure and start your own compatible LLM server separately. Source edits and running tests are disabled by default; add `--allow-writes` / `--allow-tests` after reviewing the task plan. This is **not a security sandbox**. Test changes in an isolated environment.

## What 16 agents means

Sixteen distinct sessions can be busy on independent tasks, with at most two model HTTP requests and four tool executions running concurrently **in that single orchestrator process**. Other workers may be waiting. Agent summaries are small prompt inputs and shared event details are stored once; however, there is no guarantee that every agent has semantically understood everything. `critical-status` measures delivery acknowledgements rather than comprehension.

## Known gaps / roadmap

This is not Grokbot feature parity: no automatic hierarchical planning, agent-to-agent chat, distributed inference semaphore, speculative decoding, production browsing/MCP integration, GPU KV-cache sharing, OS sandbox, AirLLM weight streaming, or sustained live-backend 16-agent benchmark. The baseline v0.2 implementation remains compatible; v0.3 is a separate module and CLI.
