"""
src/utils/config.py
Load and expose configs/config.yaml as a simple namespace.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the YAML config file.

    Searches in this order:
    1. ``path`` argument
    2. ``HAN_CONFIG`` environment variable
    3. ``<repo_root>/configs/config.yaml``
    """
    if path is None:
        path = os.environ.get(
            "HAN_CONFIG",
            Path(__file__).resolve().parents[2] / "configs" / "config.yaml",
        )
    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return _to_namespace(raw)


# Module-level singleton so callers can just do:
#   from src.utils.config import CFG
CFG = load_config()
