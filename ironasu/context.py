"""Explicitly opt-in, deterministic lexical retrieval for coding tasks.

This reduces *prefill input*, not the FLOPs per output token. It does not guess
which omitted source code is semantically unnecessary; callers must evaluate quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[\u3040-\u30ff\u3400-\u9fff]+")
_STOP = frozenset("the a an and or to in for from with how what is are of this that code file function fix please explain make use add can you i need".split())


@dataclass(frozen=True)
class Selection:
    text: str
    original_chars: int
    selected_chars: int
    chunks_selected: int
    chunks_total: int

    def report(self) -> dict[str, Any]:
        return {
            "original_source_chars": self.original_chars,
            "selected_source_chars": self.selected_chars,
            "source_chars_omitted": max(0, self.original_chars - self.selected_chars),
            "chunks_selected": self.chunks_selected,
            "chunks_total": self.chunks_total,
            "note": "Character counts are not token counts or measured FLOPs; omissions may affect correctness.",
        }


def _terms(text: str) -> set[str]:
    words = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        for term in raw.split("_"):
            if len(term) > 1 and term not in _STOP:
                words.add(term)
    return words


def select_code(files: list[dict[str, str]], query: str, budget_chars: int = 12000) -> Selection:
    """Select newline-based code windows ranked by exact lexical overlap.

    This is intentionally not semantic search. Every snippet includes its file and
    line range; original input isn't modified. No files are opened on disk.
    """
    if not isinstance(files, list) or not files or len(files) > 100:
        raise ValueError("x_ironasu_files must contain 1-100 files")
    if not isinstance(query, str):
        raise ValueError("last user message must contain text for code retrieval")
    if not isinstance(budget_chars, int) or not 1000 <= budget_chars <= 100000:
        raise ValueError("x_ironasu_context_chars must be an integer in [1000, 100000]")
    queries = _terms(query)
    parts: list[tuple[float, int, str, str, int, int, int]] = []
    original_chars = 0
    for file_idx, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"path", "content"}:
            raise ValueError("each file needs exactly path and content")
        path, content = item["path"], item["content"]
        if not isinstance(path, str) or not isinstance(content, str) or not path or len(path) > 300:
            raise ValueError("invalid file path or content")
        if "\n" in path or "\r" in path or "\x00" in path:
            raise ValueError("file paths cannot contain control characters")
        original_chars += len(content)
        if original_chars > 2_000_000:
            raise ValueError("combined source content exceeds 2,000,000 characters")
        lines = content.splitlines(keepends=True)
        path_terms = _terms(path)
        path_overlap = len(queries & path_terms)
        for start in range(0, len(lines), 32):
            chunk = "".join(lines[start:start + 32])
            if not chunk:
                continue
            tokens = _terms(chunk)
            match = len(queries & tokens)
            # Code identifiers + a matching path are substantially more useful than raw length.
            score = match * 5 + path_overlap * 3
            if match and any(line.lstrip().startswith(("def ", "async def ", "class ", "function "))
                             for line in lines[start:start + 32]):
                score += 1
            parts.append((score, file_idx, path, chunk, start + 1, min(start + 32, len(lines)), len(chunk)))
    # Stable deterministic ordering for ties: earlier file, earlier lines.
    parts.sort(key=lambda v: (-v[0], v[1], v[4]))
    chosen: list[str] = []
    selected = 0
    used = 0
    for _, _, path, chunk, first, last, n in parts:
        header = f"\n--- {path} [lines {first}-{last}] ---\n"
        available = budget_chars - used - len(header)
        if available <= 0:
            break
        if len(chunk) > available:
            # Take partial text only if there is enough room to be useful.
            if available < 120:
                continue
            chunk = chunk[:available]
            n = len(chunk)
        chosen.append(header + chunk)
        selected += n
        used += len(header) + n
        if used >= budget_chars:
            break
    note = "Selected source excerpts (lexical matching only; other code omitted; verify before editing):"
    return Selection(note + "".join(chosen), original_chars, selected, len(chosen), len(parts))
