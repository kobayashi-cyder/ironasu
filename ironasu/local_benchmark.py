"""Measure local ironasu latency, upstream/cache routing, and reported token usage.

Stdlib only. Does not save prompts, completions, or credentials.
Example:
  python -m ironasu.local_benchmark --model qwen3.5-4b-weight-space-programming
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.read(512).decode('utf-8', 'replace')}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc
    elapsed = time.perf_counter() - start
    if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
        raise RuntimeError("Invalid chat/completions response")
    return result, elapsed


def measure(endpoint: str, model: str, repeats: int, timeout: float) -> dict:
    url = endpoint.rstrip("/") + "/chat/completions"
    rows = []
    # Vary uncached prompts to measure actual upstream work, not exact-response reuse.
    for i in range(repeats):
        payload = {
            "model": model, "temperature": 0,
            "x_ironasu_cache": False,
            "messages": [{"role": "user", "content": f"Reply in one short Japanese sentence. Benchmark case {i + 1}."}],
        }
        result, elapsed = post_json(url, payload, timeout)
        rows.append(record("uncached", i + 1, result, elapsed))
    # Same deterministic payload repeated to validate exact cache and avoided inference.
    payload = {
        "model": model, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly: ironasu benchmark cache check"}],
    }
    for i in range(repeats):
        result, elapsed = post_json(url, payload, timeout)
        rows.append(record("cache_probe", i + 1, result, elapsed))
    upstream = [x["elapsed_s"] for x in rows if x["phase"] == "uncached"]
    hit_rows = [x for x in rows if x["source"] == "cache"]
    # Token usage is optional and may be missing from compatible backends.
    return {
        "endpoint": endpoint,
        "model": model,
        "runs": rows,
        "uncached_median_s": round(statistics.median(upstream), 4),
        "cache_probe_hits": len(hit_rows),
        "cache_probe_requests": repeats,
        "note": "Wall-clock timings include local HTTP overhead; token counts depend on backend usage reporting.",
    }


def record(phase: str, index: int, result: dict, elapsed: float) -> dict:
    meta = result.get("_ironasu") or {}
    usage = result.get("usage") or {}
    return {
        "phase": phase,
        "run": index,
        "elapsed_s": round(elapsed, 4),
        "source": meta.get("source", "unknown"),
        "route": meta.get("route", "unknown"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ironasu on a local OpenAI-compatible endpoint")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", help="Optional JSON output file; omit to print only")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100 or args.timeout <= 0:
        parser.error("--repeats must be 1..100 and --timeout must be positive")
    try:
        result = measure(args.endpoint, args.model, args.repeats, args.timeout)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
