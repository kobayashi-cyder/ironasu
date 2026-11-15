# Reproducible, explicit on-device LLM timing

The new ironasu-benchmark-runtime CLI compares the old paged inference path with opt-in KV prefill caching and fixed/adaptive one-stage prefetch. It does not download checkpoints.

Example:

    ironasu-benchmark-runtime LOCAL_MODEL_DIR --prompt "Write Python code" --device cpu --tokens 8 --repeats 4 --kv-cache-mb 128 --modes baseline fixed_prefetch adaptive_prefetch kv kv_adaptive --output report.json

All candidate modes run repeated identical prompts. The old baseline runs again afterward to check reference drift. The tool records initial and repeated wall time, p50/p95 warm time, generated-text equality and per-engine KV stats. CUDA peak allocator bytes are reported if available. The code rejects changed outputs: no performance improvement is reported for a mismatching candidate or unstable reference.

Do not treat this as TTFT, an energy benchmark, token throughput, or true physical SSD read measurement. CPU memory/RAM peak and hardware SSD traffic need external OS monitoring. The pager's disk_read_bytes is logical tensor payload read via its index, including OS page-cache hits. The tool does not claim a real-model speedup absent a user-supplied local checkpoint and hardware run.

Keep prompt, model, generation length, quantization and device identical across modes. A single prompt does not establish accuracy or broad performance. This tool intentionally avoids speculative expert prediction and automatic adoption of performance winners.
