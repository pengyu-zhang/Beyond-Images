"""Incremental JSONL persistence with resume support.

Every long-running stage appends one JSON record per completed unit of work,
so interrupted runs lose nothing and re-runs skip finished units.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_keys(path: str | Path, key: str) -> set[str]:
    """Collect the values of `key` from an existing JSONL file (for resume)."""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


class JsonlWriter:
    """Append-only JSONL writer that flushes after every record."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_json_atomic(data: Any, path: str | Path, indent: int | None = 4) -> None:
    """Write JSON via a temp file + rename so interrupts never corrupt output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
