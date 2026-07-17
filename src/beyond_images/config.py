"""YAML configuration loading with dot-notation overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Read-mostly nested config with attribute/dot access.

    >>> cfg = Config({"fusion": {"model": "google/flan-t5-base"}})
    >>> cfg.get("fusion.model")
    'google/flan-t5-base'
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls(data)
        for item in overrides or []:
            key, _, value = item.partition("=")
            if not _:
                raise ValueError(f"Override must look like key=value, got: {item!r}")
            cfg.set(key.strip(), yaml.safe_load(value.strip()))
        return cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override non-dict node at {part!r}")
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._data.get(name, {}))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)
