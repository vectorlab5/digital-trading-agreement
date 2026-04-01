"""Bag-of-ngrams embeddings with L2-normalised truncated SVD (linear-time baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


@dataclass
class TextEmbeddingModel:
    """Fitted TF–IDF + SVD pipeline."""

    pipeline: Pipeline
    svd_dim: int

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self.pipeline.transform(texts)
        if sparse.issparse(x):
            x = x.toarray()
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return x / norms


def build_text_embedder(
    texts: list[str],
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    svd_dim: int,
    min_df: int,
    random_state: int,
) -> TextEmbeddingModel:
    """Fit on the full provision corpus so downstream stages share one vocabulary."""
    svd_dim = min(svd_dim, max_features - 1, max(2, len(texts) // 2))
    svd_dim = max(2, svd_dim)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    pipe = Pipeline([("tfidf", tfidf), ("svd", svd)])
    pipe.fit(texts)
    return TextEmbeddingModel(pipeline=pipe, svd_dim=svd_dim)


def embed_texts(model: TextEmbeddingModel, texts: list[str]) -> np.ndarray:
    return model.transform(texts)
