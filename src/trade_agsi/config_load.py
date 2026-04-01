"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trade_agsi.repo import resolve_path


def load_config(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
