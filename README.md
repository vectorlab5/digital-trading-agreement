# Trade–AGSI replication code

This repository builds treaty-level governance indices from a provision-level corpus, aligns thematic mass with an external policy benchmark table, and runs a two-way fixed-effects panel on a firm–year table.

## Data

The pipeline expects three workbooks: **D1** (treaty corpus), **D2** (firm–year panel), and **D3** (policy benchmark tables used for the exogenous thematic baseline). Paths and filenames are set in `configs/default.yaml` (default layout: `data/d1/`, `data/d2/`, `data/d3/`). This repository ships **analysis code only**; it does not emit those spreadsheets.

## What you need

- Python 3.10+
- The three workbooks in place as configured
- ~4GB RAM is enough for the default TF–IDF + SVD settings

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[umap]"    # optional: umap-learn for non-linear manifold step
pip install pytest
```

Core dependencies are listed in `requirements.txt` / `pyproject.toml`.

## Run the pipeline

From the repository root:

```bash
python scripts/run_replication.py --config configs/default.yaml
```

Outputs land in `experiments/last_run/` (or the path set under `paths.output_dir` in the YAML): `summary.json`, `omega_by_treaty.csv`, and `topic_strictness_long.csv`.

## Configuration

- `embedding.*` — n-gram TF–IDF and SVD dimensionality
- `manifold.backend` — `pca` (default) or `umap` if installed
- `topics.backend` — `kmeans` (default) or `hdbscan` if installed
- `transport.method` — `hadamard` (default) or `sinkhorn` for entropic OT on centroid costs
- `strictness.*` — linear composite weights for the annotated slice and ridge extrapolation
- `panel.*` — entity/time keys, outcome, lagged treatment, controls

## Tests

```bash
pytest -q
```

## Layout

| Path | Role |
|------|------|
| `src/trade_agsi/` | Library code |
| `configs/` | YAML parameters |
| `scripts/` | CLI entry points |
| `tests/` | `pytest` checks |

## License

MIT
