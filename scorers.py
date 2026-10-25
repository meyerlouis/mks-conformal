"""Nonconformity scores."""

import numpy as np
from scipy.spatial.distance import pdist


def _ensure_2d(x):
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def rbf_matrix(X, Y, lengthscale):
    """RBF (Gaussian) kernel Gram matrix."""
    X, Y = _ensure_2d(X), _ensure_2d(Y)
    sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ Y.T + np.sum(Y**2, axis=1)
    return np.exp(-np.maximum(sq_dists, 0) / (2 * lengthscale**2))


def median_lengthscale(eps, subsample=5000):
    """d * median pairwise distance / 2. The paper uses this divided by d."""
    eps = _ensure_2d(eps)
    d = eps.shape[1]
    T = eps.shape[0]
    if T > subsample:
        idx = np.random.default_rng(24).choice(T, subsample, replace=False)
        eps = eps[idx]
    return float(np.median(pdist(eps, 'euclidean'))) * d / 2.0


def auto_gamma(Kc):
    """gamma = 90th percentile of the eigenvalues of the centered Gram matrix."""
    eigvals = np.linalg.eigvalsh(Kc)[::-1]
    tol = eigvals[0] * 1e-10
    active = eigvals[eigvals > tol]
    return float(np.percentile(active, 90))


def conformal_quantile(scores, alpha):
    """ceil((1 - alpha)(T + 1)) / T quantile of the calibration scores."""
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    T = len(scores)
    level = min(np.ceil((1 - alpha) * (T + 1)) / T, 1.0)
    return float(np.quantile(scores, level))


class BonferroniScorer:
    """Coordinate-wise Bonferroni baseline."""

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, calibration_set):
        eps = _ensure_2d(calibration_set)
        self.mu = eps.mean(axis=0)
        sigma = eps.std(axis=0, ddof=1)
        self.sigma = np.where(sigma > 0, sigma, 1e-10)
        return self

    def score(self, eps_new):
        eps_new = _ensure_2d(eps_new)
        return np.max(np.abs(eps_new - self.mu) / self.sigma, axis=1)

    def __repr__(self):
        return "BonferroniScorer()"


class MahalanobisScorer:
    """Mahalanobis distance score."""

    def __init__(self, regularization=1e-8):
        self.regularization = regularization
        self.mu = None
        self.cov = None
        self.cov_inv = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)
        p = eps_cal.shape[1]
        self.mu = eps_cal.mean(axis=0)
        self.cov = np.cov(eps_cal.T).reshape(p, p) + self.regularization * np.eye(p)
        self.cov_inv = np.linalg.inv(self.cov)
        return self

    def score(self, eps):
        eps = _ensure_2d(eps)
        d = eps - self.mu
        return np.sum(d @ self.cov_inv * d, axis=1)

    def __repr__(self):
        return f"MahalanobisScorer(reg={self.regularization})"


class DensityScorer:
    """Density (MMD^2) score, the gamma -> infinity limit of the MKS."""

    def __init__(self, lengthscale=None, auto_parameters=True):
        self.lengthscale = lengthscale
        self.auto_parameters = auto_parameters
        self.eps_cal = None
        self.grand_mean = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)

        if self.auto_parameters:
            self.lengthscale = median_lengthscale(eps_cal)

        K = rbf_matrix(eps_cal, eps_cal, self.lengthscale)
        self.grand_mean = K.mean()
        self.eps_cal = eps_cal
        return self

    def score(self, eps_new):
        eps_new = _ensure_2d(eps_new)
        Kv = rbf_matrix(eps_new, self.eps_cal, self.lengthscale)
        p_hat = Kv.mean(axis=1)
        return 1.0 - 2 * p_hat + self.grand_mean

    def __repr__(self):
        return f"DensityScorer(ℓ={self.lengthscale}, auto={self.auto_parameters})"


class KernelScorer:
    """Multivariate Kernel Score (MKS)."""

    def __init__(self, lengthscale=None, gamma=None, auto_parameters=True):
        self.auto_parameters = auto_parameters

        if not auto_parameters:
            if lengthscale is None:
                raise ValueError(
                    "When auto_parameters=False, lengthscale must be provided."
                )

        self.lengthscale = lengthscale
        self.gamma = gamma

        self.eps_cal = None
        self.col_mean = None
        self.grand_mean = None
        self.A = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)
        T = eps_cal.shape[0]

        if self.auto_parameters:
            self.lengthscale = median_lengthscale(eps_cal)

        K = rbf_matrix(eps_cal, eps_cal, self.lengthscale)

        self.col_mean = K.mean(axis=0)
        self.grand_mean = K.mean()

        Kc = K - self.col_mean[None, :] - self.col_mean[:, None] + self.grand_mean

        if self.gamma == None:
            self.gamma = auto_gamma(Kc)

        self.A = np.linalg.solve(Kc + self.gamma * np.eye(T), np.eye(T))
        self.eps_cal = eps_cal

        return self

    def score(self, eps):
        eps = _ensure_2d(eps)

        Kv = rbf_matrix(eps, self.eps_cal, self.lengthscale)
        kcm = Kv.mean(axis=1)

        Kv_c = Kv - kcm[:, None] - self.col_mean[None, :] + self.grand_mean

        self_c = 1.0 - 2 * kcm + self.grand_mean

        return self_c - np.sum(Kv_c @ self.A * Kv_c, axis=1)

    def __repr__(self):
        return f"KernelScorer(ℓ={self.lengthscale}, γ={self.gamma}, auto={self.auto_parameters})"
