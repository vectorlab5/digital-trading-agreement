"""Two-way fixed-effects panel (entity + time) with optional entity clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


@dataclass
class PanelFitResult:
    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    rsquared: float
    rsquared_within: float
    nobs: int
    summary: str


def fit_two_way_fe(
    df: pd.DataFrame,
    entity: str,
    time: str,
    dependent: str,
    covariates: list[str],
    cluster_entity: bool = True,
) -> PanelFitResult:
    """Estimate Eq. FE specification with linearmodels."""
    data = df.copy()
    for c in [dependent] + covariates:
        if c not in data.columns:
            raise KeyError(f"Column {c} not in panel")
    data = data.dropna(subset=[dependent] + covariates)
    data = data.set_index([entity, time]).sort_index()
    y = data[dependent]
    x = data[covariates]
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    fit = mod.fit(
        cov_type="clustered",
        cluster_entity=cluster_entity,
    )
    return PanelFitResult(
        params=fit.params,
        std_errors=fit.std_errors,
        pvalues=fit.pvalues,
        rsquared=float(fit.rsquared),
        rsquared_within=float(fit.rsquared_within),
        nobs=int(fit.nobs),
        summary=str(fit.summary),
    )


def mae_f1_at_median(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Provision-level diagnostics: MAE and F1 on a high/low split at the median."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    med = np.median(y_true)
    yt = (y_true >= med).astype(int)
    yp = (y_pred >= med).astype(int)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"mae": mae, "f1_median_split": float(f1)}
