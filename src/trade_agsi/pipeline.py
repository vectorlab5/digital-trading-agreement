"""End-to-end AGSI construction and panel validation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_agsi.config_load import load_config
from trade_agsi.embeddings import build_text_embedder, embed_texts
from trade_agsi.io_workbooks import (
    assert_required_columns,
    load_d1_workbook,
    load_d2_workbook,
    load_d3_workbook,
)
from trade_agsi.manifold import project_manifold
from trade_agsi.panel_regression import fit_two_way_fe, mae_f1_at_median
from trade_agsi.repo import resolve_path
from trade_agsi.strictness import expert_composite_b, ridge_b_on_full_corpus, topic_strictness_table, treaty_omega
from trade_agsi.topics import discover_topics, order_topics_by_mass, topic_ids_from_codes
from trade_agsi.transport_weights import compute_omega_weights, load_nu_y_vector, topic_mass_from_labels

logger = logging.getLogger(__name__)


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("random_seed", 42))
    paths = config["paths"]
    d1 = load_d1_workbook(paths["d1_workbook"])
    d2 = load_d2_workbook(paths["d2_workbook"])
    d3 = load_d3_workbook(paths["d3_workbook"])

    prov = d1.provisions
    expert = d1.expert_subset
    assert_required_columns(
        prov,
        ["provision_id", "treaty_id", "text"],
        "provisions",
    )
    assert_required_columns(
        expert,
        [
            "provision_id",
            "gold_M_modal",
            "gold_S_specificity",
            "gold_E_enforceability",
            "gold_X_exceptions",
        ],
        "expert_subset",
    )

    texts = prov["text"].astype(str).tolist()
    emb_cfg = config["embedding"]
    embedder = build_text_embedder(
        texts,
        max_features=int(emb_cfg["max_features"]),
        ngram_min=int(emb_cfg["ngram_min"]),
        ngram_max=int(emb_cfg["ngram_max"]),
        svd_dim=int(emb_cfg["svd_dim"]),
        min_df=int(emb_cfg["min_df"]),
        random_state=seed,
    )
    x_embed = embed_texts(embedder, texts)

    man_cfg = config["manifold"]
    z = project_manifold(
        x_embed,
        backend=str(man_cfg["backend"]),
        random_state=seed,
        umap_n_neighbors=int(man_cfg.get("umap_n_neighbors", 30)),
        umap_min_dist=float(man_cfg.get("umap_min_dist", 0.08)),
        umap_metric=str(man_cfg.get("umap_metric", "cosine")),
    )

    top_cfg = config["topics"]
    n_topics = int(top_cfg["n_topics"])
    labels_raw, _ = discover_topics(
        z,
        n_topics=n_topics,
        backend=str(top_cfg["backend"]),
        random_state=seed,
        hdbscan_min_cluster_size=int(top_cfg.get("hdbscan_min_cluster_size", 40)),
    )
    order = order_topics_by_mass(labels_raw, n_topics)
    remap = {old: pos for pos, old in enumerate(order)}
    labels_canon = np.vectorize(remap.get)(labels_raw)
    topic_codes = topic_ids_from_codes(labels_canon.astype(int))

    centroids = np.stack([z[labels_canon == j].mean(axis=0) for j in range(n_topics)])

    mu = topic_mass_from_labels(labels_canon.astype(int), n_topics)
    nu, topic_order = load_nu_y_vector(d3.nu_y_aggregate, n_topics)
    tr_cfg = config["transport"]
    omega_vec = compute_omega_weights(
        mu,
        nu,
        method=str(tr_cfg["method"]),
        centroids=centroids,
        sinkhorn_reg=float(tr_cfg.get("sinkhorn_reg", 0.05)),
        sinkhorn_iterations=int(tr_cfg.get("sinkhorn_iterations", 200)),
    )

    wcfg = config["strictness"]
    weights = {
        "w_m": float(wcfg["w_m"]),
        "w_s": float(wcfg["w_s"]),
        "w_e": float(wcfg["w_e"]),
        "w_x": float(wcfg["w_x"]),
    }
    b_hat = ridge_b_on_full_corpus(
        prov,
        expert,
        x_embed,
        [str(p) for p in prov["provision_id"]],
        weights,
        ridge_alpha=float(wcfg["ridge_alpha"]),
    )

    topic_scores = topic_strictness_table(prov, b_hat, topic_codes)
    omega_tbl = treaty_omega(topic_scores, omega_vec, topic_order)

    y_true = expert_composite_b(expert, weights).to_numpy()
    pid_to_pos = {str(p): i for i, p in enumerate(prov["provision_id"].astype(str))}
    rows = [pid_to_pos[str(p)] for p in expert["provision_id"].astype(str)]
    metrics = mae_f1_at_median(y_true, b_hat.iloc[rows].to_numpy())

    panel_cfg = config["panel"]
    fy = d2.firm_year
    assert_required_columns(
        fy,
        [panel_cfg["entity"], panel_cfg["time"], panel_cfg["dependent"], panel_cfg["treatment_lag"]]
        + list(panel_cfg["controls"]),
        "firm_year_panel",
    )
    covariates = [panel_cfg["treatment_lag"]] + list(panel_cfg["controls"])
    fe = fit_two_way_fe(
        fy,
        entity=str(panel_cfg["entity"]),
        time=str(panel_cfg["time"]),
        dependent=str(panel_cfg["dependent"]),
        covariates=covariates,
        cluster_entity=bool(panel_cfg.get("cluster_entity", True)),
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "n_provisions": len(prov),
        "n_treaties": int(omega_tbl["treaty_id"].nunique()),
        "topic_mass_mu": mu.tolist(),
        "nu_y": nu.tolist(),
        "omega_weights": omega_vec.tolist(),
        "topic_order": topic_order,
        "provision_metrics": metrics,
        "panel": {
            "coefficients": fe.params.to_dict(),
            "std_errors": fe.std_errors.to_dict(),
            "pvalues": fe.pvalues.to_dict(),
            "rsquared": fe.rsquared,
            "rsquared_within": fe.rsquared_within,
            "nobs": fe.nobs,
        },
        "tables": {
            "omega_by_treaty": omega_tbl,
            "topic_scores": topic_scores,
        },
    }


def write_run_artifacts(result: dict[str, Any], output_dir: str | Path) -> None:
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "tables"}
    (out / "summary.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    result["tables"]["omega_by_treaty"].to_csv(out / "omega_by_treaty.csv", index=False)
    result["tables"]["topic_scores"].to_csv(out / "topic_strictness_long.csv", index=False)
    logger.info("Wrote artifacts under %s", out)
