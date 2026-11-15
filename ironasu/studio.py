"""Installable Studio launcher and local LLM-server compatibility CLI.

Run:
    python -m ironasu.studio
    ironasu-studio --provider ollama
    ironasu-studio --backend-url http://127.0.0.1:1234/v1
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from urllib.parse import urlsplit

from .compat import AutoHTTPBackend, PRESETS, normalize_backend_url
from .core import Config, HTTPBackend
from .server import serve


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ironasu-studio",
        description="Offline-first Studio UI for local OpenAI-compatible model servers",
    )
    p.add_argument("--provider", default="auto", choices=("auto", *PRESETS),
                   help="Local server preset; auto detects loopback servers (default: auto)")
    p.add_argument("--backend-url", metavar="URL", help="Custom OpenAI API base URL (explicit opt-in)")
    p.add_argument("--backend-key", metavar="TOKEN", help="Backend API token (prefer IRONASU_BACKEND_KEY)")
    p.add_argument("--fast-model", metavar="ID", help="Explicit fast model ID (default: first detected model)")
    p.add_argument("--quality-model", metavar="ID", help="Explicit quality model ID (default: first detected model)")
    p.add_argument("--port", type=int, help="ironasu HTTP port (default: 8765)")
    p.add_argument("--host", help="ironasu bind host (default: 127.0.0.1; non-local requires IRONASU_SERVER_KEY)")
    p.add_argument("--no-browser", action="store_true", help="Do not automatically open the UI browser tab")
    p.add_argument("--doctor", action="store_true", help="Probe configured local backend and exit")
    p.add_argument("--version", action="version", version="ironasu Studio 1.2.0")
    return p


def configure(args: argparse.Namespace) -> tuple[Config, object]:
    """Return config + backend without touching network or downloading anything."""
    env = os.environ
    base = Config.from_env()
    specified = args.backend_url or env.get("IRONASU_BACKEND_URL")
    if specified:
        url = normalize_backend_url(specified)
    elif args.provider != "auto":
        url = PRESETS[args.provider]
    else:
        # Informational fallback only: auto backend probes its own fixed list.
        url = PRESETS["lmstudio"]
    config = replace(
        base,
        backend_url=url,
        backend_key=args.backend_key if args.backend_key is not None else base.backend_key,
        fast_model=args.fast_model if args.fast_model is not None else env.get("IRONASU_FAST_MODEL", ""),
        quality_model=args.quality_model if args.quality_model is not None else env.get("IRONASU_QUALITY_MODEL", ""),
        auto_models=True,
        listen_host=args.host if args.host is not None else base.listen_host,
        listen_port=args.port if args.port is not None else base.listen_port,
    )
    config.validate()
    if not specified and args.provider == "auto":
        return config, AutoHTTPBackend(config)
    return config, HTTPBackend(config)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config, backend = configure(args)
    except ValueError as exc:
        parser().error(str(exc))
    if args.doctor:
        models = backend.list_models()
        source = backend.endpoint() if isinstance(backend, AutoHTTPBackend) else {
            "provider": args.provider if not args.backend_url else "custom",
            "url": config.backend_url,
        }
        report = {
            "connected": models is not None,
            "provider": source["provider"],
            "api_base": source["url"],
            "models": models or [],
            "fast_model": config.fast_model or (models[0] if models else None),
            "quality_model": config.quality_model or (models[0] if models else None),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if models else 1

    if args.backend_url and urlsplit(config.backend_url).hostname not in (
        "localhost", "127.0.0.1", "::1"
    ):
        print("WARNING: custom remote backend receives your prompts and conversation content.")
    print("ironasu Studio: inference requires a separately running local model server.")
    if isinstance(backend, AutoHTTPBackend):
        print("Auto discovery: LM Studio / Ollama / llama.cpp / LocalAI / vLLM / Jan.")
    serve(config, backend=backend, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
