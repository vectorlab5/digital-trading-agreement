"""Repository root and path resolution."""

from pathlib import Path

from trade_agsi.repo import repository_root, resolve_path


def test_repository_root_exists():
    root = repository_root()
    assert (root / "configs").is_dir() or (root / "pyproject.toml").exists()


def test_resolve_relative():
    root = repository_root()
    p = resolve_path("configs/default.yaml")
    assert p == root / "configs/default.yaml"
