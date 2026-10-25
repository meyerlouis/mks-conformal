"""MKS with the lengthscale and gamma set explicitly, for the hyperparameter grid."""

import hashlib

import numpy as np

from scorers import KernelScorer, _ensure_2d, rbf_matrix, median_lengthscale


_EIG_CACHE = {}


def _cached_gamma(eps_cal, ell, Kc, q):
    """gamma = q-th percentile of the active spectrum (q = 90 is scorers.auto_gamma)."""
    key = (hashlib.sha1(np.ascontiguousarray(eps_cal).tobytes()).hexdigest(),
           float(ell))
    ev = _EIG_CACHE.get(key)
    if ev is None:
        eigvals = np.linalg.eigvalsh(Kc)[::-1]
        tol = eigvals[0] * 1e-10
        ev = eigvals[eigvals > tol]
        _EIG_CACHE[key] = ev
    return float(np.percentile(ev, q))


class GridKernelScorer(KernelScorer):
    """MKS at ell = ell_mult * median/2 and gamma at the gamma_q-th spectrum percentile."""

    def __init__(self, ell_mult=1.0, gamma_q=90.0):
        super().__init__(lengthscale=1.0, gamma=None, auto_parameters=False)
        self.ell_mult = float(ell_mult)
        self.gamma_q = float(gamma_q)

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)
        T, d = eps_cal.shape

        self.lengthscale = self.ell_mult * median_lengthscale(eps_cal) / d

        K = rbf_matrix(eps_cal, eps_cal, self.lengthscale)
        self.col_mean = K.mean(axis=0)
        self.grand_mean = K.mean()
        Kc = K - self.col_mean[None, :] - self.col_mean[:, None] + self.grand_mean

        self.gamma = _cached_gamma(eps_cal, self.lengthscale, Kc, self.gamma_q)

        self.A = np.linalg.solve(Kc + self.gamma * np.eye(T), np.eye(T))
        self.eps_cal = eps_cal
        return self

    def __repr__(self):
        return (f"GridKernelScorer(ell_mult={self.ell_mult}, "
                f"gamma_q={self.gamma_q}, ell={self.lengthscale}, "
                f"gamma={self.gamma})")


ELL_MULTS = [16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]
GAMMA_QS = [5, 10, 30, 50, 70, 90, 95]

ADOPTED = (1.0, 90)


def _tag(x):
    s = f"{x:g}"
    return s.replace('.', 'p')


def grid_name(ell_mult, gamma_q):
    return f"G_x{_tag(ell_mult)}_q{int(gamma_q)}"


def register(factories, ell_mults=None, gamma_qs=None):
    """Add the grid scorers to SCORER_FACTORIES and return their names."""
    ell_mults = ELL_MULTS if ell_mults is None else ell_mults
    gamma_qs = GAMMA_QS if gamma_qs is None else gamma_qs
    names = []
    for m in ell_mults:
        for q in gamma_qs:
            nm = grid_name(m, q)
            factories[nm] = (lambda seed, _m=m, _q=q: GridKernelScorer(_m, _q))
            names.append(nm)
    return names
