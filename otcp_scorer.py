"""
OT-CP score of Klein et al. (2025).

NumPy/SciPy port of ott.tools.conformal.OTCP (OTT-JAX, Apache-2.0). Modified to
add an optional quantile scaling of the residuals (scale_mode='quantile').
"""

import math

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


def _ensure_2d(x):
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _sqdist(A, B):
    """Squared Euclidean cost matrix, (n, m). Matches ott.geometry.costs.SqEuclidean."""
    d = (np.sum(A ** 2, axis=1, keepdims=True)
         - 2.0 * (A @ B.T)
         + np.sum(B ** 2, axis=1))
    return np.maximum(d, 0.0)


def sobol_ball(n_target, d, seed=0):
    """Low-discrepancy approximation to the uniform measure on the unit d-ball."""
    n_per_radius = int(math.ceil(math.sqrt(n_target)))
    n_sphere, n_zeros = divmod(n_target, n_per_radius)
    n_sphere = max(n_sphere, 1)

    m_pow = int(math.ceil(math.log2(max(n_sphere, 2))))
    u = qmc.Sobol(d=d, seed=seed, scramble=True).random_base2(m=m_pow)[:n_sphere]
    gauss = ndtri(np.clip(u, 1e-12, 1.0 - 1e-12))
    sphere = gauss / np.maximum(
        np.linalg.norm(gauss, axis=1, keepdims=True), np.finfo(float).tiny
    )

    radii = np.linspace(1.0 / n_per_radius, 1.0, n_per_radius)
    Z = (sphere[None, :, :] * radii[:, None, None]).reshape(-1, d)
    b = np.full(Z.shape[0], 1.0 / n_target)

    if n_zeros:
        Z = np.vstack([Z, np.zeros((1, d))])
        b = np.append(b, n_zeros / n_target)

    return Z, b / b.sum()


def _sinkhorn_log(C, a, b, eps, max_iter=2000, tol=1e-6, verbose=False):
    """Log-domain Sinkhorn, returns the dual potentials."""
    log_a = np.log(a)
    log_b = np.log(b)

    f = np.zeros(C.shape[0])
    g = np.zeros(C.shape[1])

    M = np.empty_like(C)

    err = np.inf
    n_iters = 0

    for it in range(1, max_iter + 1):
        np.subtract(g[None, :], C, out=M)
        M /= eps
        M += log_b[None, :]
        f = -eps * logsumexp(M, axis=1)

        np.subtract(f[:, None], C, out=M)
        M /= eps
        M += log_a[:, None]
        g_new = -eps * logsumexp(M, axis=0)

        delta = np.clip((g - g_new) / eps, -60.0, 60.0)
        err = float(np.sum(b * np.abs(np.expm1(delta))))

        g = g_new
        n_iters = it

        if verbose and (it == 1 or it % 50 == 0):
            print(f"      sinkhorn it={it} marginal_err={err:.3e}", flush=True)

        if err < tol:
            break

    return f, g, n_iters, err


class OTCPScorer:
    """Optimal-transport conformal scorer (Klein et al., 2025)."""

    def __init__(self, epsilon=0.1, relative_epsilon=None, n_target=4096,
                 max_fit_points=None, chunk_size=4096, seed=0,
                 max_iter=2000, tol=1e-6, verbose=False,
                 scale_mode='max', scale_quantile=0.99, blur_ratio=None):
        self.epsilon = epsilon
        self.relative_epsilon = relative_epsilon
        self.n_target = n_target
        self.max_fit_points = max_fit_points
        self.chunk_size = chunk_size
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.scale_mode = scale_mode
        self.scale_quantile = scale_quantile
        self.blur_ratio = blur_ratio

        self.offset = None          # (1, d) residual mean
        self.scale = None           # float, max residual norm after centring
        self.Z = None               # (m, d) reference ball support
        self.log_b = None           # (m,) log reference weights
        self.g = None               # (m,) dual potential
        self.eps_used = None        # float, epsilon actually applied

        self.n_fit_points = None
        self.sinkhorn_iters = None
        self.sinkhorn_error = None
        self.converged = None
        self.norm_stats = None

    def fit(self, calibration_set):
        """Fit the transport map on the fit fold residuals."""
        S = _ensure_2d(calibration_set)
        n, d = S.shape

        if self.max_fit_points is not None and n > self.max_fit_points:
            idx = np.random.default_rng(self.seed).choice(
                n, self.max_fit_points, replace=False
            )
            S = S[idx]
            n = S.shape[0]

        self.n_fit_points = int(n)

        self.offset = S.mean(axis=0, keepdims=True)
        centred = S - self.offset
        norms = np.linalg.norm(centred, axis=1)

        if self.scale_mode == 'quantile':
            self.scale = float(np.quantile(norms, self.scale_quantile))
        elif self.scale_mode == 'max':
            self.scale = float(norms.max())
        else:
            raise ValueError(f"scale_mode must be 'max' or 'quantile', "
                             f"got {self.scale_mode!r}")

        if not np.isfinite(self.scale) or self.scale <= 0:
            self.scale = 1.0

        with np.errstate(invalid='ignore', divide='ignore'):
            rel = norms / self.scale
        self.norm_stats = {
            'norm_median': float(np.median(norms)),
            'norm_q99': float(np.quantile(norms, 0.99)),
            'norm_max': float(norms.max()),
            'tail_ratio_max_over_q99': float(
                norms.max() / max(np.quantile(norms, 0.99), 1e-300)),
            'data_radius_median': float(np.median(rel)),
            'data_radius_q99': float(np.quantile(rel, 0.99)),
            'frac_outside_ball': float(np.mean(rel > 1.0)),
        }

        S = centred / self.scale

        self.Z, b = sobol_ball(self.n_target, d, seed=0)
        self.log_b = np.log(b)

        C = _sqdist(S, self.Z)

        if self.blur_ratio is not None:
            r_med = self.norm_stats['data_radius_median']
            if not np.isfinite(r_med) or r_med <= 0:
                raise ValueError("blur_ratio requires a positive median data radius")
            self.eps_used = float((self.blur_ratio * r_med) ** 2)
        elif self.relative_epsilon == 'std':
            self.eps_used = float(self.epsilon) * float(C.std())
        elif self.relative_epsilon == 'mean':
            self.eps_used = float(self.epsilon) * float(C.mean())
        else:
            self.eps_used = float(self.epsilon)

        a = np.full(n, 1.0 / n)
        _, self.g, self.sinkhorn_iters, self.sinkhorn_error = _sinkhorn_log(
            C, a, b, self.eps_used,
            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
        )
        self.converged = bool(self.sinkhorn_error < self.tol)

        del C
        return self

    def transport(self, eps_new):
        """Push residuals through the entropic map into the reference ball."""
        if self.g is None:
            raise RuntimeError("OTCPScorer.transport called before fit().")

        E = (_ensure_2d(eps_new) - self.offset) / self.scale
        out = np.empty_like(E)

        for i in range(0, len(E), self.chunk_size):
            block = E[i:i + self.chunk_size]

            logits = _sqdist(block, self.Z)
            np.subtract(self.g[None, :], logits, out=logits)
            logits /= self.eps_used
            logits += self.log_b[None, :]
            logits -= logits.max(axis=1, keepdims=True)

            np.exp(logits, out=logits)
            logits /= logits.sum(axis=1, keepdims=True)

            out[i:i + len(block)] = logits @ self.Z

        return out

    def score(self, eps_new):
        """Nonconformity score: the center-outward rank of each residual."""
        return np.linalg.norm(self.transport(eps_new), axis=1)

    def params(self):
        """Diagnostics for logging."""
        out = {
            'epsilon': float(self.epsilon),
            'eps_used': None if self.eps_used is None else float(self.eps_used),
            'blur_length': None if self.eps_used is None else float(np.sqrt(self.eps_used)),
            'relative_epsilon': self.relative_epsilon,
            'blur_ratio': None if self.blur_ratio is None else float(self.blur_ratio),
            'scale_mode': self.scale_mode,
            'scale_quantile': float(self.scale_quantile),
            'n_target': int(self.n_target),
            'n_ball_points': None if self.Z is None else int(self.Z.shape[0]),
            'n_shells': int(math.ceil(math.sqrt(self.n_target))),
            'n_fit_points': self.n_fit_points,
            'scale': None if self.scale is None else float(self.scale),
            'sinkhorn_iters': self.sinkhorn_iters,
            'sinkhorn_error': None if self.sinkhorn_error is None else float(self.sinkhorn_error),
            'converged': self.converged,
        }
        if self.norm_stats:
            out.update(self.norm_stats)
            if self.eps_used is not None and out['data_radius_median'] > 0:
                out['blur_over_data_radius'] = float(
                    np.sqrt(self.eps_used) / out['data_radius_median'])
        return out

    def __repr__(self):
        return (f"OTCPScorer(eps={self.epsilon}, n_target={self.n_target}, "
                f"rel={self.relative_epsilon}, seed={self.seed})")
