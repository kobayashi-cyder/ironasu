"""Private HTTP worker; no project code or GUI is imported in this process."""
from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any
from urllib import error, request


class ModelError(RuntimeError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def complete(settings, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    url = settings.endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = _json({"model": settings.model, "messages": messages, "tools": tools,
                     "tool_choice": "auto", "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = "Bearer " + settings.api_key
    req = request.Request(url, data=payload, headers=headers, method="POST")
    for attempt in range(3):
        try:
            with request.urlopen(req, timeout=settings.timeout_s) as response:
                raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Backend response exceeds 1 MiB")
                data = json.loads(raw)
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
            return message
        except error.HTTPError as exc:
            body = exc.read(1024).decode("utf-8", "replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise ModelError(f"Backend HTTP {exc.code}: {body}") from exc
        except (error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise ModelError(f"Backend unavailable: {exc}") from exc
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ModelError("Backend returned invalid chat/completions JSON") from exc
        time.sleep(0.5 * (2 ** attempt))
    raise AssertionError("unreachable")



def _streams():
    if sys.stdin is not None and sys.stdout is not None:
        return sys.stdin.buffer, sys.stdout.buffer
    # PyInstaller --windowed intentionally sets sys.std* to None. Popen still
    # supplies inherited standard handles; open those without creating a window.
    if os.name != "nt":
        raise RuntimeError("Worker has no redirected standard streams")
    import msvcrt
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [w.DWORD]
    kernel.GetStdHandle.restype = w.HANDLE
    inputs = kernel.GetStdHandle(w.DWORD(-10))
    outputs = kernel.GetStdHandle(w.DWORD(-11))
    if not inputs or not outputs or inputs == w.HANDLE(-1).value or outputs == w.HANDLE(-1).value:
        raise RuntimeError("Worker standard handles are unavailable")
    return (os.fdopen(msvcrt.open_osfhandle(inputs, os.O_RDONLY | os.O_BINARY), "rb"),
            os.fdopen(msvcrt.open_osfhandle(outputs, os.O_WRONLY | os.O_BINARY), "wb"))


def main():
    source, target = _streams()
    try:
        if os.name == "posix":
            import resource
            budget = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
        raw = source.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Model worker input exceeds budget")
        payload = json.loads(raw)
        settings = SimpleNamespace(**payload["settings"])
        if not math.isfinite(settings.timeout_s) or settings.timeout_s <= 0:
            raise ValueError("Invalid model timeout")
        result = {"message": complete(settings, payload["messages"], payload["tools"])}
    except Exception as exc:
        result = {"error": type(exc).__name__ + ": " + str(exc)[:1500]}
    target.write(_json(result).encode("utf-8"))
    target.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
