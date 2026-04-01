"""Low-dimensional geometry before topic discovery (PCA by default; UMAP optional)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def project_manifold(
    x: np.ndarray,
    backend: str,
    random_state: int,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.08,
    umap_metric: str = "cosine",
    pca_dim: int | None = None,
) -> np.ndarray:
    """
    Reduce dimensionality for clustering.

    PCA keeps the pipeline deterministic on CPU; UMAP trades runtime for
    non-linear structure when the optional dependency is present.
    """
    if backend == "pca":
        dim = pca_dim or min(50, x.shape[1], x.shape[0] - 1)
        dim = max(2, dim)
        pca = PCA(n_components=dim, random_state=random_state)
        return pca.fit_transform(x)

    if backend == "umap":
        try:
            import umap
        except ImportError as e:
            logger.warning("umap-learn not installed; falling back to PCA (%s)", e)
            return project_manifold(x, "pca", random_state, pca_dim=pca_dim)

        reducer = umap.UMAP(
            n_neighbors=min(umap_n_neighbors, max(5, x.shape[0] // 20)),
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    raise ValueError(f"Unknown manifold backend: {backend}")
