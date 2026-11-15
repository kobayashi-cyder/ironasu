"""Safely preview lexical context extraction against a local coding repository."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context import select_code

EXTENSIONS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".md", ".toml", ".yaml", ".yml"})
EXCLUDE_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".next", ".cache", "coverage"})


def collect_files(root: Path, limit: int = 100) -> list[dict[str, str]]:
    """Collect text source under root without following symlinks or reading secrets."""
    root = root.resolve(strict=True)
    if not root.is_dir() or not 1 <= limit <= 100:
        raise ValueError("--repo must be a directory; limit must be in 1..100")
    files: list[dict[str, str]] = []
    for p in sorted(root.rglob("*")):
        if len(files) >= limit:
            break
        relative = p.relative_to(root)
        if any(part.startswith(".") or part in EXCLUDE_DIRS for part in relative.parts):
            continue
        if p.is_symlink() or not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        try:
            if p.stat().st_size > 150_000:
                continue
            content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        files.append({"path": relative.as_posix(), "content": content})
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview source context before sending it to an LLM")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--budget-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print JSON rather than the selected text")
    args = parser.parse_args()
    files = collect_files(Path(args.repo), args.limit)
    if not files:
        parser.error("no supported source files found")
    result = select_code(files, args.query, args.budget_chars)
    print(json.dumps({"report": result.report(), "selected_text": result.text}, ensure_ascii=False, indent=2)
          if args.json else result.text + "\n\n" + json.dumps(result.report(), ensure_ascii=False))


if __name__ == "__main__":
    main()
