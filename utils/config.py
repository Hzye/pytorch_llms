"""
Config loading with _base_ inheritance.

Usage:
    cfg = load_config("configs/transformer_small.yaml")
    cfg.model.d_model   # 256  (overridden)
    cfg.training.epochs # 30   (inherited from base.yaml)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Dict wrapper with dot-access and pretty repr."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            object.__setattr__(self, key, Config(value) if isinstance(value, dict) else value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def __repr__(self) -> str:
        lines = ["Config("]
        for key, value in self.__dict__.items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base; nested dicts are merged, not replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> Config:
    """
    Load a YAML config, resolving _base_ inheritance.

    Raises:
        FileNotFoundError: If path or any _base_ file does not exist.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw: dict = yaml.safe_load(f) or {}

    if "_base_" not in raw:
        return Config(raw)

    base_path = path.parent / raw.pop("_base_")
    base_cfg = load_config(base_path)

    return Config(_deep_merge(base_cfg.to_dict(), raw))