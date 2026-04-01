"""Unit tests for OT-style weight helpers."""

import numpy as np

from trade_agsi.transport_weights import compute_omega_weights, topic_mass_from_labels


def test_hadamard_normalised():
    mu = np.array([0.5, 0.3, 0.2])
    nu = np.array([0.1, 0.6, 0.3])
    w = compute_omega_weights(mu, nu, "hadamard", None, 0.05, 200)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w > 0).all()


def test_topic_mass():
    labels = np.array([0, 0, 1, 1, 1])
    m = topic_mass_from_labels(labels, n_topics=2)
    assert abs(m.sum() - 1.0) < 1e-9
