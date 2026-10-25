"""Data generating process for the synthetic experiment."""

import numpy as np


def generate_dgp(n, seed=42):
    """2D response with nonlinear mean function and non-Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, n)

    f1 = 3 * X**2 - 1.5 * X + np.sin(4 * np.pi * X)
    f2 = 2 * X**3 - X + 0.5 * np.cos(3 * np.pi * X)
    f_true = np.column_stack([f1, f2])

    z1 = rng.normal(0, 0.3, n)
    z2 = rng.normal(0, 0.1, n)
    eps = np.column_stack([z1, z2 + 0.5 * z1**2 - 0.5 * 0.3**2])
    eps -= eps.mean(axis=0)

    Y = f_true + eps
    return X, Y, f_true
