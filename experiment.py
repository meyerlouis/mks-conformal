"""Synthetic experiment runner."""

import numpy as np

from scorers import conformal_quantile, MahalanobisScorer
from volume_estimator import mc_volume, ellipsoid_volume
from metrics import compute_wsc
from pipeline import generate_and_fit


def run_single_seed(dgp_fn, models, scorer_factories, alphas,
                    n=20_000, split=(0.5, 0.25, 0.25), seed=21,
                    volume_n_samples=100_000):
    """One seed: generate data, fit the models and scorers, evaluate at every alpha."""
    data = generate_and_fit(dgp_fn, models, n=n, split=split, seed=seed)

    fitted_scorers = {}   # model -> scorer_name -> fitted scorer
    auto_params = {}      # model -> scorer_name -> {lengthscale, gamma, ...}
    results = {}          # model -> scorer_name -> alpha -> metrics

    for model_name in data['residuals_cal']:
        rc = data['residuals_cal'][model_name]
        rt = data['residuals_test'][model_name]
        n_c = len(rc)
        cal_perm = np.random.RandomState(seed + 10_000).permutation(n_c)
        rc_fit = rc[cal_perm[: n_c // 2]]
        rc_q = rc[cal_perm[n_c // 2:]]
        assert len(rc_q) == n_c - n_c // 2 and len(rc_fit) == n_c // 2, "calibration split sizes are wrong"
        X_test = data['X_test']
        d = rc.shape[1] if rc.ndim > 1 else 1

        fitted_scorers[model_name] = {}
        auto_params[model_name] = {}
        results[model_name] = {}

        for scorer_name, factory in scorer_factories.items():
            scorer = factory()
            scorer.fit(rc_fit)
            fitted_scorers[model_name][scorer_name] = scorer

            params = {}
            if hasattr(scorer, 'lengthscale') and scorer.lengthscale is not None:
                params['lengthscale'] = float(scorer.lengthscale)
            if hasattr(scorer, 'gamma') and scorer.gamma is not None:
                params['gamma'] = float(scorer.gamma)
            auto_params[model_name][scorer_name] = params

            scores_cal = scorer.score(rc_q)
            scores_test = scorer.score(rt)
            results[model_name][scorer_name] = {}

            for alpha in alphas:
                threshold = conformal_quantile(scores_cal, alpha)
                covered = scores_test <= threshold
                cov = float(covered.mean())
                wsc = compute_wsc(X_test, covered)

                if isinstance(scorer, MahalanobisScorer) and scorer.cov is not None:
                    vol = ellipsoid_volume(scorer.cov, threshold, d)
                else:
                    vol = mc_volume(scorer, threshold, rc,
                                    n_samples=volume_n_samples, seed=0)

                results[model_name][scorer_name][alpha] = {
                    'coverage': cov,
                    'volume': vol,
                    'wsc': wsc,
                    'threshold': threshold,
                }

    return {
        'seed': seed,
        'fitted_scorers': fitted_scorers,
        'auto_params': auto_params,
        'results': results,
        'data': data,
    }
