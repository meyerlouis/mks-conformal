"""Real-data pipeline: one seed, one model, all scorers and alphas."""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from scorers import (
    KernelScorer, MahalanobisScorer, BonferroniScorer, DensityScorer,
    conformal_quantile, median_lengthscale,
)
from otcp_scorer import OTCPScorer
from volume_estimator_new import (
    estimate_volume_ratios, containment_check,
    log_ellipsoid_volume, log_box_volume,
)
from metrics import compute_wsc


DEFAULT_SPLIT = (0.50, 0.25, 0.25)
DEFAULT_ALPHAS = [0.01, 0.02, 0.05, 0.1]

WSC_N_SLABS = 200

DEFAULT_VOLUME_KWARGS = dict(
    kappa=3.0,
    target_rel_se=0.01,
    block_size=8192,
    min_samples=32_768,
    max_samples=400_000,
    seed=0,
)


MODEL_REGISTRY = {
    'Ridge': lambda: MultiOutputRegressor(Ridge(alpha=1.0)),
    'MLP': lambda: MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=0,
    ),
}


SCORER_FACTORIES = {
    'Mahal':        lambda seed: MahalanobisScorer(),
    'Bonferroni':   lambda seed: BonferroniScorer(),

    'Kernel_med2':  lambda seed: MedianKernelScorer(),
    'Density_med2': lambda seed: MedianDensityScorer(),

    'OTCP':         lambda seed: OTCPScorer(
        epsilon=0.1,
        relative_epsilon=None,
        scale_mode='quantile',
        scale_quantile=0.99,
        n_target=4096,
        max_fit_points=None,
        chunk_size=4096,
        seed=seed,
    ),
}

SCORER_DISPLAY_NAMES = {
    'Mahal': 'Mahalanobis',
    'Bonferroni': 'Bonferroni',
    'Kernel_med2': 'MKS ($\\ell{=}\\mathrm{med}/2$)',
    'Density_med2': 'Density ($\\ell{=}\\mathrm{med}/2$)',
    'OTCP': 'OT-CP (Klein et al.)',
}

ANALYTIC_VOLUME_NAMES = ('Mahal', 'Bonferroni')

PARAM_ATTRS = ('lengthscale', 'gamma')


class MedianKernelScorer(KernelScorer):
    """MKS with lengthscale median/2."""

    def __init__(self):
        super().__init__(lengthscale=1.0, gamma=None, auto_parameters=False)

    def fit(self, calibration_set):
        eps = np.asarray(calibration_set, dtype=float)
        d_ = eps.shape[1] if eps.ndim > 1 else 1
        self.lengthscale = median_lengthscale(eps) / d_
        self.gamma = None
        return super().fit(eps)


class MedianDensityScorer(DensityScorer):
    """Density score with lengthscale median/2."""

    def __init__(self):
        super().__init__(lengthscale=1.0, auto_parameters=False)

    def fit(self, calibration_set):
        eps = np.asarray(calibration_set, dtype=float)
        d_ = eps.shape[1] if eps.ndim > 1 else 1
        self.lengthscale = median_lengthscale(eps) / d_
        return super().fit(eps)


def _split_and_scale(X, Y, seed, split):
    """Train/calibration/test split, standardized on the training set."""
    n = len(X)
    perm = np.random.RandomState(seed).permutation(n)

    n_train = int(n * split[0])
    n_cal = int(n * split[1])

    X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
    X_cal, Y_cal = X[perm[n_train:n_train + n_cal]], Y[perm[n_train:n_train + n_cal]]
    X_te, Y_te = X[perm[n_train + n_cal:]], Y[perm[n_train + n_cal:]]

    scaler_x = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(Y_tr)

    return {
        'X_train': scaler_x.transform(X_tr),
        'Y_train': scaler_y.transform(Y_tr),
        'X_cal':   scaler_x.transform(X_cal),
        'Y_cal':   scaler_y.transform(Y_cal),
        'X_test':  scaler_x.transform(X_te),
        'Y_test':  scaler_y.transform(Y_te),
    }


def _collect_params(scorer):
    if hasattr(scorer, 'params') and callable(scorer.params):
        try:
            return scorer.params()
        except Exception as exc:
            return {'params_error': repr(exc)}

    params = {}
    for attr in PARAM_ATTRS:
        val = getattr(scorer, attr, None)
        if val is not None and np.isscalar(val):
            params[attr] = float(val)
    return params


def _q_level(T, alpha):
    """The conformal quantile level actually used, and whether it saturated."""
    raw = np.ceil((1.0 - alpha) * (T + 1)) / T
    return float(min(raw, 1.0)), bool(raw > 1.0)


def _score_summary(scores):
    """Summary of the test scores."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {}
    qs = [0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
    smax = float(s.max())
    return {
        'quantiles': {str(q): float(np.quantile(s, q)) for q in qs},
        'mean': float(s.mean()),
        'frac_within_1e3_of_max': float(np.mean(s >= smax - 1e-3)),
        'n_distinct_approx': int(np.unique(np.round(s, 6)).size),
    }


def run_single(X, Y, seed, model_name, split=None, alphas=None,
               scorer_names=None, volume_kwargs=None, estimate_volumes=True,
               compute_wsc_metric=True, max_kappa=2000.0, verbose=True):
    """One seed, one model, all scorers, all alphas."""
    t0 = time.time()
    split = split or DEFAULT_SPLIT
    alphas = list(alphas) if alphas else list(DEFAULT_ALPHAS)

    vk = dict(DEFAULT_VOLUME_KWARGS)
    if volume_kwargs:
        vk.update(volume_kwargs)

    if scorer_names is None:
        scorer_names = list(SCORER_FACTORIES.keys())
    scorer_names = list(scorer_names)
    if 'Mahal' not in scorer_names:
        scorer_names.insert(0, 'Mahal')

    timings = {}

    data = _split_and_scale(X, Y, seed, split)
    d = data['Y_train'].shape[1]
    p = data['X_train'].shape[1]

    if verbose:
        print(f'  Fitting {model_name}...', flush=True)
    t = time.time()
    model = MODEL_REGISTRY[model_name]()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(data['X_train'], data['Y_train'])
    timings['model_fit'] = time.time() - t

    pred_cal = model.predict(data['X_cal'])
    pred_test = model.predict(data['X_test'])
    if pred_cal.ndim == 1:
        pred_cal = pred_cal[:, None]
    if pred_test.ndim == 1:
        pred_test = pred_test[:, None]

    rc = data['Y_cal'] - pred_cal
    rt = data['Y_test'] - pred_test

    n_c = len(rc)
    cal_perm = np.random.RandomState(seed + 10_000).permutation(n_c)
    rc_fit = rc[cal_perm[: n_c // 2]]
    rc_q = rc[cal_perm[n_c // 2:]]
    assert len(rc_fit) == n_c // 2 and len(rc_q) == n_c - n_c // 2, \
        "calibration split sizes are wrong"

    model_rmse = float(np.sqrt(np.mean((data['Y_test'] - pred_test) ** 2)))

    if d > 1:
        _cov = np.cov(rc_fit.T)
        _eig = np.linalg.eigvalsh(_cov)
        _cond = float(_eig[-1] / max(_eig[0], 1e-300))
        if _cond > 1e10 or _eig[0] <= 0:
            warnings.warn(
                f"residual covariance is near-singular (condition {_cond:.3e}, "
                f"min eigenvalue {_eig[0]:.3e}). Check for collinear targets."
            )
        elif verbose:
            print(f'  residual cov condition: {_cond:.3e}', flush=True)

    fitted_scorers = {}
    scores_cal = {}
    scores_test = {}

    for name in scorer_names:
        t = time.time()
        if verbose:
            print(f'  Fitting {name}...', flush=True)
        scorer = SCORER_FACTORIES[name](seed)
        scorer.fit(rc_fit)
        fitted_scorers[name] = scorer
        timings[f'fit_{name}'] = time.time() - t

        t = time.time()
        scores_cal[name] = scorer.score(rc_q)
        scores_test[name] = scorer.score(rt)
        timings[f'score_{name}'] = time.time() - t

        if verbose:
            print(f'    {name}: fit {timings[f"fit_{name}"]:.1f}s, '
                  f'score {timings[f"score_{name}"]:.1f}s', flush=True)

    thresholds = {
        name: {alpha: float(conformal_quantile(scores_cal[name], alpha))
               for alpha in alphas}
        for name in scorer_names
    }

    results = {
        'seed': seed,
        'model': model_name,
        'model_rmse': model_rmse,
        'meta': {
            'n_train': data['X_train'].shape[0],
            'n_cal': data['X_cal'].shape[0],
            'n_fit': len(rc_fit),
            'n_q': len(rc_q),
            'n_test': data['X_test'].shape[0],
            'd': d,
            'p': p,
            'alphas': alphas,
            'scorers': scorer_names,
            'display_names': {k: SCORER_DISPLAY_NAMES.get(k, k) for k in scorer_names},
        },
    }

    covered_flags = {}
    for name in scorer_names:
        results[name] = {
            'params': _collect_params(fitted_scorers[name]),
            'score_summary': _score_summary(scores_test[name]),
        }
        covered_flags[name] = {}

        for alpha in alphas:
            covered = scores_test[name] <= thresholds[name][alpha]
            covered_flags[name][alpha] = covered

            t = time.time()
            wsc = (compute_wsc(data['X_test'], covered, n_slabs=WSC_N_SLABS)
                   if compute_wsc_metric else float('nan'))
            timings['wsc'] = timings.get('wsc', 0.0) + (time.time() - t)

            lvl, capped = _q_level(len(scores_cal[name]), alpha)
            results[name][alpha] = {
                'coverage': float(covered.mean()),
                'wsc': float(wsc),
                'threshold': float(thresholds[name][alpha]),
                'q_level': lvl,
                'level_capped': capped,
                'volume': float('nan'),
                'volume_ratio': float('nan'),
            }

    if estimate_volumes:
        t = time.time()

        cov = fitted_scorers['Mahal'].cov
        analytic = {}
        if 'Mahal' in fitted_scorers:
            analytic['Mahal'] = (
                lambda a, _cov=cov: log_ellipsoid_volume(_cov, thresholds['Mahal'][a], d)
            )
        if 'Bonferroni' in fitted_scorers:
            _sigma = fitted_scorers['Bonferroni'].sigma
            analytic['Bonferroni'] = (
                lambda a, _s=_sigma: log_box_volume(_s, thresholds['Bonferroni'][a], d)
            )
        analytic = {k: v for k, v in analytic.items() if k in ANALYTIC_VOLUME_NAMES}

        pre = containment_check(
            scores_test=scores_test,
            mahal_scores_test=scores_test['Mahal'],
            thresholds=thresholds,
            alphas=alphas, d=d, kappa=vk['kappa'],
        )
        kpm = {}
        for name in scorer_names:
            if name in analytic:
                continue
            need = max(pre[name][a].get('kappa_required') or 1.0 for a in alphas)
            kpm[name] = float(min(max(need, vk['kappa']), max_kappa))

        if verbose:
            print(f'  Estimating volume ratios (target rel SE='
                  f'{vk["target_rel_se"]})...', flush=True)
            for n, k in sorted(kpm.items()):
                print(f'    proposal for {n}: kappa={k:.1f}', flush=True)

        vol = estimate_volume_ratios(
            fitted_scorers=fitted_scorers,
            thresholds=thresholds,
            alphas=alphas,
            mahal_scorer=fitted_scorers['Mahal'],
            d=d,
            kappa_per_method=kpm,
            analytic_log_volume=analytic,
            verbose=verbose,
            **vk,
        )
        timings['volume'] = time.time() - t

        contain = containment_check(
            scores_test=scores_test,
            mahal_scores_test=scores_test['Mahal'],
            thresholds=thresholds,
            alphas=alphas,
            d=d,
            kappa=kpm,
        )

        results['volume_meta'] = vol['_meta']

        for name in scorer_names:
            for alpha in alphas:
                v = vol[name][alpha]
                results[name][alpha].update({
                    'volume': v['volume'],
                    'log_volume': v['log_volume'],
                    'volume_ratio': v['ratio'],
                    'volume_ratio_se': v['ratio_se'],
                    'volume_method': v['volume_method'],
                    'volume_ratio_mc': v['ratio_mc'],
                    'volume_ratio_mc_se': v['ratio_mc_se'],
                    'hits': v['hits'],
                    'n_samples': v['n_samples'],
                    'kappa_used': v.get('kappa_used'),
                    'p_hat': v['p_hat'],
                    'zero_hits': v['zero_hits'],
                    'saturated': v['saturated'],
                    'leaks': contain[name][alpha]['leaks'],
                    'n_outside': contain[name][alpha]['n_outside'],
                    'leak_frac': contain[name][alpha]['leak_frac'],
                    'reach_over_qmahal': contain[name][alpha].get('reach_over_qmahal'),
                    'kappa_required': contain[name][alpha].get('kappa_required'),
                })

    elapsed = time.time() - t0
    results['meta']['elapsed'] = elapsed
    results['meta']['timings'] = timings

    if verbose:
        print(f'  seed={seed} model={model_name} rmse={model_rmse:.4f} '
              f'[{elapsed:.1f}s]', flush=True)

    return results
