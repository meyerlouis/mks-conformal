
import numpy as np


def compute_wsc(X_test, covered, n_slabs=200, min_slab_size=30, seed=42):
    """Worst-slab coverage."""
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    n, d = X_test.shape
    rng = np.random.default_rng(seed)
    worst = 1.0

    for _ in range(n_slabs):
        w = rng.standard_normal(d)
        w /= np.linalg.norm(w)
        proj = X_test @ w

        for q in np.linspace(0.0, 0.8, 20):
            mask = (proj >= np.quantile(proj, q)) & (proj <= np.quantile(proj, q + 0.2))

            if mask.sum() >= min_slab_size:
                worst = min(worst, covered[mask].mean())

    return float(worst)
