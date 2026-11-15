# Reasoning-preserving local inference optimization

The coding harness deliberately leaves model reasoning enabled and never rewrites
model chat templates or truncates internal reasoning. As a low-risk prefill
optimization, it sends only the tool schemas allowed for the current session:
read-only defaults omit `write_file`, `edit_file`, and `run_checks`; permission
checks still run when tools are executed. This reduces request size but **is not
a guaranteed speedup of reasoning token generation**.

For Windows + LM Studio on an integrated Radeon:

1. Keep `Context Length` at a working value (e.g. 8192). The 4096 setting
   may reject a tool-heavy prompt with HTTP 400.
2. Compare GPU offload Auto versus Max, not just Max in isolation. Integrated
   GPUs share RAM; memory pressure or CPU↔GPU scheduling can make Max slower.
3. In LM Studio's My Models → model settings, enable Flash Attention if the
   installed backend supports it. Reload the model to apply load settings.
4. Keep the same model, prompt, context length, and Reasoning setting in both
   runs. Compare end-to-end success, latency, input/output tokens and available
   tokens-per-second stats. Repeat runs to reduce warm-up variance.
5. With a single 16 GiB system, begin with one concurrent backend request.

PowerShell (from the repository root):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_lean_harness_tools -v
.\.venv\Scripts\python.exe -m ironasu.harness `
  "Explain README.md and pyproject.toml briefly in Japanese." `
  --workspace . --session reasoning-perf-a `
  --endpoint http://127.0.0.1:8765/v1 `
  --model qwen3.5-4b-weight-space-programming --max-steps 10
```

Ensure ironasu is listening on 8765 and LM Studio on 1235 before running the
harness. Use a fresh session ID per comparative run. This does not implement
speculative token decoding or alter model weights.
