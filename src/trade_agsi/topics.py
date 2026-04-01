"""Topic discovery (KMeans default; HDBSCAN optional)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)


def discover_topics(
    z: np.ndarray,
    n_topics: int,
    backend: str,
    random_state: int,
    hdbscan_min_cluster_size: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cluster_labels 0..K-1, cluster_centroids).

    Centroids live in the same space as ``z`` and support cost construction
    for entropic transport when enabled in config.
    """
    n_topics = min(n_topics, z.shape[0])
    if backend == "kmeans":
        km = MiniBatchKMeans(
            n_clusters=n_topics,
            random_state=random_state,
            batch_size=2048,
            n_init="auto",
        )
        labels = km.fit_predict(z)
        return labels, km.cluster_centers_

    if backend == "hdbscan":
        try:
            import hdbscan
        except ImportError as e:
            logger.warning("hdbscan not installed; falling back to k-means (%s)", e)
            return discover_topics(z, n_topics, "kmeans", random_state)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        raw = clusterer.fit_predict(z)
        noise = raw == -1
        if noise.any():
            km = KMeans(n_clusters=n_topics, random_state=random_state, n_init=10)
            km.fit(z[noise])
            raw[noise] = km.labels_ + raw.max() + 1
        uniq = np.unique(raw)
        remap = {old: i for i, old in enumerate(uniq)}
        labels = np.vectorize(remap.get)(raw)
        centroids = np.array([z[labels == j].mean(axis=0) for j in range(len(uniq))])
        return labels, centroids

    raise ValueError(f"Unknown topic backend: {backend}")


def order_topics_by_mass(labels: np.ndarray, n_topics: int) -> list[int]:
    """Map cluster ids to stable topic indices sorted by descending mass."""
    counts = np.bincount(labels, minlength=n_topics)
    order = list(np.argsort(-counts))
    return order


def labels_to_topic_codes(labels: np.ndarray, order: list[int]) -> np.ndarray:
    """Remap cluster ids so T01 is the largest cluster, etc."""
    remap = {old: new for new, old in enumerate(order)}
    return np.vectorize(remap.get)(labels)


def topic_ids_from_codes(codes: np.ndarray) -> np.ndarray:
    return np.array([f"T{int(i) + 1:02d}" for i in codes])
