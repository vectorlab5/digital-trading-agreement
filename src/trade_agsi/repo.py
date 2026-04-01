"""Repository root resolution for portable relative paths."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the project root (parent of ``configs/`` and ``src/``)."""
    here = Path(__file__).resolve()
    return here.parents[2]


def resolve_path(maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return repository_root() / p
