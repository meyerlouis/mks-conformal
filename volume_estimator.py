"""Volume estimation for the synthetic experiments (d = 2)."""

import numpy as np
from scipy.special import gammaln


def mc_volume(scorer, threshold, reference_points, n_samples=100_000, pad=0.01, seed=0):
    """Estimate volume of {ε : scorer.score(ε) ≤ threshold} via bounding-box MC."""
    ref = np.asarray(reference_points)
    if ref.ndim == 1:
        ref = ref[:, None]

    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    margin = (hi - lo) * pad
    lo -= margin
    hi += margin

    rng = np.random.default_rng(seed)
    samples = rng.uniform(lo, hi, size=(n_samples, ref.shape[1]))

    BATCH = 10_000
    n_inside = 0
    for i in range(0, n_samples, BATCH):
        batch = samples[i:i+BATCH]
        n_inside += int(np.sum(scorer.score(batch) <= threshold))

    bbox_volume = np.prod(hi - lo)
    return float(bbox_volume * n_inside / n_samples)


def ellipsoid_volume(cov, threshold, d):
    """Exact volume of {ε : (ε-μ)'Σ⁻¹(ε-μ) ≤ threshold}."""
    log_ball = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1) + (d / 2) * np.log(threshold)
    sign, log_det = np.linalg.slogdet(cov)
    return float(np.exp(log_ball + 0.5 * log_det))
