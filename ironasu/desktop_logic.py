"""Pure, UI-independent helpers for the native ironasu Studio desktop window.

No browser, Tkinter, disk writes or third-party packages are required here.
"""
from __future__ import annotations

import codecs
import json
import time
from dataclasses import dataclass, field
from typing import Any


def extract_text(delta: Any) -> str:
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def extract_reasoning(delta: Any) -> str:
    if not isinstance(delta, dict):
        return ""
    for field in ("reasoning_content", "reasoning"):
        value = delta.get(field)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value["text"]
    details = delta.get("reasoning_details")
    if isinstance(details, list):
        return "".join(part["text"] for part in details
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


class SSEEvents:
    """Streaming UTF-8 SSE parser, tolerant of chunk and CRLF boundaries."""
    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.buffer = ""
        self.max_pending = 2 * 1024 * 1024
        self.done = False

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        if self.done:
            return []
        self.buffer = (self.buffer + self.decoder.decode(data)).replace("\r\n", "\n")
        if len(self.buffer) > self.max_pending:
            raise ValueError("SSE event exceeds 2 MiB")
        output: list[dict[str, Any]] = []
        while "\n\n" in self.buffer:
            raw, self.buffer = self.buffer.split("\n\n", 1)
            values = [line[5:].lstrip() for line in raw.splitlines()
                      if line.startswith("data:")]
            if not values:
                continue
            item = "\n".join(values).strip()
            if item == "[DONE]":
                self.done = True
                break
            try:
                obj = json.loads(item)
            except ValueError:
                continue
            if isinstance(obj, dict):
                output.append(obj)
        return output


@dataclass
class GenerationStats:
    started: float = field(default_factory=time.perf_counter)
    first: float | None = None
    first_reasoning: float | None = None
    last_reasoning: float | None = None
    first_answer: float | None = None
    ended: float | None = None
    tokens: int | None = None
    answer: str = ""
    reasoning: str = ""
    thinking: bool = False
    tag_tail: str = ""

    def _push(self, kind: str, text: str, moment: float) -> list[tuple[str, str]]:
        if not text:
            return []
        if self.first is None:
            self.first = moment
        if kind == "reasoning":
            if self.first_reasoning is None:
                self.first_reasoning = moment
            self.last_reasoning = moment
            self.reasoning += text
        else:
            if self.first_answer is None:
                self.first_answer = moment
            self.answer += text
        return [(kind, text)]

    def append_content(self, content: str, moment: float, *, flush: bool = False
                       ) -> list[tuple[str, str]]:
        text = self.tag_tail + content
        self.tag_tail = ""
        events: list[tuple[str, str]] = []
        while text:
            marker = "</think>" if self.thinking else "<think>"
            pos = text.find(marker)
            if pos >= 0:
                events += self._push("reasoning" if self.thinking else "answer",
                                     text[:pos], moment)
                self.thinking = not self.thinking
                text = text[pos + len(marker):]
                continue
            keep = 0
            if not flush:
                for n in range(1, min(len(marker), len(text) + 1)):
                    if marker.startswith(text[-n:]):
                        keep = n
            events += self._push("reasoning" if self.thinking else "answer",
                                 text[:-keep] if keep else text, moment)
            self.tag_tail = text[-keep:] if keep else ""
            break
        return events

    def ingest(self, chunk: dict[str, Any], moment: float) -> list[tuple[str, str]]:
        if isinstance(chunk.get("error"), dict):
            raise RuntimeError(str(chunk["error"].get("message") or "Upstream error"))
        usage = chunk.get("usage")
        if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int:
            self.tokens = usage["completion_tokens"]
        events: list[tuple[str, str]] = []
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return events
        for choice in choices[:1]:
            if not isinstance(choice, dict):
                continue
            value = choice.get("delta") or choice.get("message") or {}
            events += self._push("reasoning", extract_reasoning(value), moment)
            events += self.append_content(extract_text(value), moment)
        return events

    def finish(self, moment: float) -> dict[str, float | None]:
        self.append_content("", moment, flush=True)
        self.ended = moment
        duration = max(0.001, moment - self.started)
        ttft = None if self.first is None else max(0, self.first - self.started)
        if self.first_reasoning is None:
            reasoning_s = None
        else:
            endpoint = self.first_answer or self.last_reasoning or moment
            reasoning_s = max(0, endpoint - self.first_reasoning)
        # Token throughput only when usage is provided by the model server.
        tps = (self.tokens / max(0.001, moment - self.first)
               if self.tokens is not None and self.first is not None else None)
        return {
            "duration_s": duration, "ttft_s": ttft,
            "reasoning_s": reasoning_s, "tokens_per_s": tps,
            "chars_per_s": len(self.answer) / duration,
        }


def normalized_model_selection(selection: str) -> tuple[str, str | None]:
    """Preserve explicit routing aliases; never infer 'fast' from labels."""
    if selection in ("auto", "fast", "quality"):
        return selection, None
    if selection and "\n" not in selection and len(selection) <= 256:
        return "auto", selection
    raise ValueError("Model selection is invalid")
