"""Couple corpus topic mass with external policy mass for omega (OT-style baselines)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_nu_y_vector(nu_df: pd.DataFrame, n_topics: int) -> tuple[np.ndarray, list[str]]:
    """Return nu on the simplex plus the topic_id list (T01..)."""
    df = nu_df
    if len(df) != n_topics:
        logger.warning("nu_Y rows %d != n_topics %d — reindexing by topic code", len(df), n_topics)
    topic_ids = [f"T{i + 1:02d}" for i in range(n_topics)]
    mass = []
    for tid in topic_ids:
        row = df.loc[df["topic_id_hst"] == tid]
        if row.empty:
            v = np.nan
        else:
            v = float(row["nu_y_mass_normalised"].iloc[0])
        mass.append(v)
    arr = np.array(mass, dtype=np.float64)
    med = np.nanmedian(arr)
    arr = np.where(np.isnan(arr), med, arr)
    arr = np.clip(arr, 1e-8, None)
    arr = arr / arr.sum()
    return arr, topic_ids


def topic_mass_from_labels(labels_canon: np.ndarray, n_topics: int) -> np.ndarray:
    counts = np.bincount(labels_canon, minlength=n_topics).astype(np.float64)
    counts += 1e-8
    return counts / counts.sum()


def sinkhorn_coupling(
    mu: np.ndarray,
    nu: np.ndarray,
    cost: np.ndarray,
    reg: float,
    n_iter: int,
) -> np.ndarray:
    """Symmetric entropic OT; returns Gamma with approximate marginals mu, nu."""
    k = mu.shape[0]
    kexp = np.exp(-cost / max(reg, 1e-12))
    u = np.ones(k) / k
    v = np.ones(k) / k
    for _ in range(n_iter):
        u = mu / (kexp @ v + 1e-12)
        v = nu / (kexp.T @ u + 1e-12)
    return np.diag(u) @ kexp @ np.diag(v)


def compute_omega_weights(
    mu: np.ndarray,
    nu: np.ndarray,
    method: str,
    centroids: np.ndarray | None,
    sinkhorn_reg: float,
    sinkhorn_iterations: int,
) -> np.ndarray:
    """
    Combine corpus mass ``mu`` with policy mass ``nu``.

    *hadamard* emphasises themes that are salient both in treaties and in the
    benchmark tables. *sinkhorn* adds a geometric cost between cluster centroids.
    """
    mu = mu / mu.sum()
    nu = nu / nu.sum()
    if method == "hadamard":
        w = mu * nu
        return w / w.sum()

    if method == "sinkhorn":
        if centroids is None:
            raise ValueError("sinkhorn requires cluster centroids for cost matrix")
        from scipy.spatial.distance import cdist

        cost = cdist(centroids, centroids, metric="euclidean")
        g = sinkhorn_coupling(mu, nu, cost, sinkhorn_reg, sinkhorn_iterations)
        # Collapse onto treaty-topic indices using the policy marginal emphasis.
        w = g.sum(axis=1)
        return w / w.sum()

    raise ValueError(f"Unknown transport method {method}")
