"""Provision-level binding intensity B(j) and ridge extrapolation to the full corpus."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def composite_b_row(
    m: float,
    s: float,
    e: float,
    x: float,
    w_m: float,
    w_s: float,
    w_e: float,
    w_x: float,
) -> float:
    """Eq. composite B: weighted modal, specificity, enforceability minus exceptions."""
    return w_m * m + w_s * s + w_e * e - w_x * x


def expert_composite_b(expert: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Gold dimensions live on the stratified annotation export."""
    return expert.apply(
        lambda r: composite_b_row(
            float(r["gold_M_modal"]),
            float(r["gold_S_specificity"]),
            float(r["gold_E_enforceability"]),
            float(r["gold_X_exceptions"]),
            weights["w_m"],
            weights["w_s"],
            weights["w_e"],
            weights["w_x"],
        ),
        axis=1,
    )


def _modal_from_tag(tag: str) -> float:
    t = str(tag).lower()
    if "shall" in t:
        return 1.0
    if "should" in t:
        return 0.55
    if "undertake" in t:
        return 0.62
    if "may" in t:
        return 0.28
    if "recognize" in t:
        return 0.22
    return 0.48


def ridge_b_on_full_corpus(
    provisions: pd.DataFrame,
    expert: pd.DataFrame,
    x_embed: np.ndarray,
    provision_ids_order: list[str],
    weights: dict[str, float],
    ridge_alpha: float,
) -> pd.Series:
    """
    Fit a linear smoother on the annotated slice, then predict everywhere.

    Features are the same SVD text coordinates used for manifold work so the
    extrapolation stays in-distribution relative to the embedding geometry.
    """
    id_to_row = {pid: i for i, pid in enumerate(provision_ids_order)}
    expert = expert.copy()
    expert["__b"] = expert_composite_b(expert, weights)
    rows = [id_to_row[str(p)] for p in expert["provision_id"].astype(str)]
    x_train = x_embed[rows]
    y_train = expert["__b"].to_numpy(dtype=np.float64)
    model = Ridge(alpha=ridge_alpha, random_state=0)
    model.fit(x_train, y_train)
    y_all = model.predict(x_embed)
    return pd.Series(y_all, index=provisions.index, name="B_hat")


def topic_strictness_table(
    provisions: pd.DataFrame,
    b_hat: pd.Series,
    topic_codes: np.ndarray,
) -> pd.DataFrame:
    """S(t, theta_k): mean B within (treaty, topic)."""
    df = provisions[["treaty_id", "provision_id"]].copy()
    df["B_hat"] = b_hat.values
    df["topic_code"] = topic_codes
    g = df.groupby(["treaty_id", "topic_code"], as_index=False)["B_hat"].mean()
    g = g.rename(columns={"B_hat": "S_topic"})
    return g


def treaty_omega(
    topic_scores: pd.DataFrame,
    omega_weights: np.ndarray,
    topic_order: list[str],
) -> pd.DataFrame:
    """Omega_AGSI(t) = sum_k omega_k * S(t,k) with zeros for missing pairs."""
    wmap = {t: omega_weights[i] for i, t in enumerate(topic_order)}
    out = []
    for tid, sub in topic_scores.groupby("treaty_id"):
        ssum = 0.0
        wsum = 0.0
        for _, row in sub.iterrows():
            tc = str(row["topic_code"])
            if tc not in wmap:
                continue
            wk = wmap[tc]
            ssum += wk * float(row["S_topic"])
            wsum += wk
        omega = ssum / wsum if wsum > 0 else 0.0
        out.append({"treaty_id": tid, "omega_agsi": omega})
    return pd.DataFrame(out)
