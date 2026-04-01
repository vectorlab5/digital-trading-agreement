#!/usr/bin/env python3
"""
Run the full replication pipeline: embeddings → topics → weights → Ω(t) → panel FE.

Usage (from repository root):
    python scripts/run_replication.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trade_agsi.config_load import load_config
from trade_agsi.pipeline import run_pipeline, write_run_artifacts
from trade_agsi.repo import resolve_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="AGSI replication pipeline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (relative to repo root unless absolute)",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    out_path = resolve_path(cfg.get("paths", {}).get("output_dir", "experiments/last_run"))

    result = run_pipeline(cfg)
    write_run_artifacts(result, out_path)
    logging.info("Done. Summary: %s", out_path / "summary.json")


if __name__ == "__main__":
    main()
