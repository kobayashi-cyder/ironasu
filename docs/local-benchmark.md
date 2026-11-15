# Local LM Studio benchmark

This measures the local **ironasu gateway** and LM Studio, rather than claiming
a model's pure tokens/second. Run while LM Studio is serving a loaded model.

PowerShell:

```powershell
cd $HOME\ironasu
git pull
.\.venv\Scripts\python.exe -m ironasu.local_benchmark `
  --endpoint http://127.0.0.1:8765/v1 `
  --model qwen3.5-4b-weight-space-programming `
  --repeats 3 `
  --output benchmark-local.json
```

The first phase issues distinct, cache-disabled prompts. The second repeats
one deterministic request to check ironasu's exact-cache path. The JSON has
per-request wall-clock latency, ironasu `_ironasu.source` and `route`, and
usage counters only when the upstream reports them. The median of uncached
requests is a rough baseline, **not** a measure of exact-token throughput.

No prompt content, completion, or key is written to the report. Calls still
reach the configured endpoint. The benchmark does not support streaming,
hardware telemetry or tool-call tracing. Run the benchmark multiple times
after warm-up and compare runs under the same model, context and hardware
configuration. The cache-probe first request might itself hit a prior cache
entry in the gateway's TTL window.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_benchmark -v
```
