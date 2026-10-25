"""Volume ratios to the Mahalanobis ellipsoid, sampled in an enlarged ellipsoid (real data)."""

import numpy as np
from scipy.special import gammaln
import warnings


def log_ellipsoid_volume(cov, threshold, d):
    """log volume of {eps : (eps-mu)' Sigma^-1 (eps-mu) <= threshold}."""
    if threshold <= 0:
        return -np.inf
    log_ball = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0) \
        + (d / 2.0) * np.log(threshold)
    _sign, log_det = np.linalg.slogdet(cov)
    return float(log_ball + 0.5 * log_det)


def log_box_volume(sigma, threshold, d):
    """log volume of the Bonferroni region."""
    if threshold <= 0:
        return -np.inf
    sigma = np.asarray(sigma, dtype=float)
    return float(d * np.log(2.0 * threshold) + np.log(sigma).sum())


def _safe_exp(log_v):
    """exp that degrades gracefully instead of raising / warning."""
    if not np.isfinite(log_v):
        return 0.0 if log_v == -np.inf else float('inf')
    if log_v > 700.0:
        return float('inf')
    if log_v < -700.0:
        return 0.0
    return float(np.exp(log_v))


def sample_uniform_ellipsoid(mu, chol, threshold, n_samples, rng):
    """Sample uniformly from {eps : (eps-mu)' Sigma^-1 (eps-mu) <= threshold}."""
    d = len(mu)

    z = rng.standard_normal((n_samples, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    np.maximum(norms, np.finfo(float).tiny, out=norms)
    z /= norms

    r = rng.random((n_samples, 1)) ** (1.0 / d)

    return mu + np.sqrt(threshold) * (r * z) @ chol.T


def _rel_se(hits, n):
    """Relative standard error of the Bernoulli rate p = hits/n."""
    if n <= 0 or hits <= 0:
        return np.inf
    p = hits / n
    if p >= 1.0:
        return 0.0
    return float(np.sqrt((1.0 - p) / (p * n)))


def estimate_volume_ratios(
    fitted_scorers,
    thresholds,
    alphas,
    mahal_scorer,
    d,
    kappa=3.0,
    kappa_per_method=None,
    target_rel_se=0.01,
    block_size=8192,
    min_samples=32_768,
    max_samples=1_000_000,
    seed=0,
    analytic_log_volume=None,
    verbose=True,
):
    """Volume ratios to Mahalanobis, by ratio Monte Carlo."""
    analytic_log_volume = analytic_log_volume or {}
    names = list(fitted_scorers.keys())

    if 'Mahal' not in names or 'Mahal' not in thresholds:
        raise ValueError("'Mahal' must be among the fitted scorers and thresholds")

    mu = np.asarray(mahal_scorer.mu, dtype=float)
    cov = np.asarray(mahal_scorer.cov, dtype=float)
    chol = np.linalg.cholesky(cov)

    def kappa_for(nm):
        k = (kappa_per_method or {}).get(nm, kappa)
        return float(max(k, 1.0))

    results = {name: {} for name in names}
    meta = {'kappa_default': float(kappa), 'target_rel_se': float(target_rel_se),
            'block_size': int(block_size), 'max_samples': int(max_samples),
            'd': int(d), 'kappa_per_method': {n: kappa_for(n) for n in names},
            'per_alpha': {}}

    for a_i, alpha in enumerate(alphas):
        q_mahal = float(thresholds['Mahal'][alpha])
        log_V_mahal = log_ellipsoid_volume(cov, q_mahal, d)

        groups = {}
        for name in names:
            if name in analytic_log_volume:
                continue
            groups.setdefault(round(kappa_for(name), 6), []).append(name)
        if not groups:
            groups = {float(kappa): []}

        total_n = 0
        worst_check = 1.0
        group_info = {}

        for k, gnames in sorted(groups.items()):
            q_prop = (k ** (2.0 / d)) * q_mahal
            log_V_prop = log_ellipsoid_volume(cov, q_prop, d)

            implied = np.exp(log_V_prop - log_V_mahal)
            if not np.isclose(implied, k, rtol=1e-6):
                warnings.warn(f"proposal factor {implied:.6g} != kappa {k} "
                              f"at alpha={alpha}, d={d}")

            rng = np.random.default_rng(seed + 1000 * (a_i + 1) + int(k) % 997)
            hits = {n: 0 for n in gnames}
            hits_mahal = 0
            n_done = 0

            while n_done < max_samples:
                n_this = int(min(block_size, max_samples - n_done))
                samples = sample_uniform_ellipsoid(mu, chol, q_prop, n_this, rng)

                hits_mahal += int(np.count_nonzero(
                    mahal_scorer.score(samples) <= q_mahal))
                for n in gnames:
                    sc = fitted_scorers[n].score(samples)
                    hits[n] += int(np.count_nonzero(sc <= thresholds[n][alpha]))

                n_done += n_this
                del samples

                if n_done >= min_samples and gnames and all(
                        _rel_se(hits[n], n_done) <= target_rel_se for n in gnames):
                    break

            total_n += n_done
            chk = k * hits_mahal / n_done
            if abs(chk - 1.0) > abs(worst_check - 1.0):
                worst_check = chk
            group_info[k] = {'methods': gnames, 'n_samples': int(n_done),
                             'q_prop': q_prop, 'mahal_self_check': float(chk)}

            if verbose:
                print(f"    [volume] alpha={alpha} kappa={k:g} N={n_done} "
                      f"self-check={chk:.4f} -> {gnames}", flush=True)

            for n in gnames:
                h = hits[n]
                p = h / n_done
                ratio = k * p
                se = k * np.sqrt(max(p * (1.0 - p), 0.0) / n_done)
                log_v = log_V_prop + np.log(p) if p > 0 else -np.inf
                results[n][alpha] = {
                    'hits': int(h), 'n_samples': int(n_done), 'p_hat': float(p),
                    'kappa_used': float(k),
                    'ratio': float(ratio), 'ratio_se': float(se),
                    'ratio_mc': float(ratio), 'ratio_mc_se': float(se),
                    'rel_se': float(_rel_se(h, n_done)),
                    'volume_method': 'mc_ratio',
                    'log_volume': float(log_v), 'volume': _safe_exp(log_v),
                    'zero_hits': bool(h == 0), 'saturated': bool(p > 0.90),
                }
                if p > 0.90:
                    warnings.warn(f"{n} alpha={alpha}: hit rate {p:.3f} > 0.9, "
                                  f"region nearly fills the proposal; raise kappa")
                if h == 0:
                    warnings.warn(f"{n} alpha={alpha}: zero hits in {n_done} samples")

        for n in analytic_log_volume:
            if n not in names:
                continue
            log_v = float(analytic_log_volume[n](alpha))
            results[n][alpha] = {
                'hits': None, 'n_samples': 0, 'p_hat': float('nan'),
                'kappa_used': float('nan'),
                'ratio': _safe_exp(log_v - log_V_mahal), 'ratio_se': 0.0,
                'ratio_mc': float('nan'), 'ratio_mc_se': float('nan'),
                'rel_se': 0.0, 'volume_method': 'analytic',
                'log_volume': log_v, 'volume': _safe_exp(log_v),
                'zero_hits': False, 'saturated': False,
            }

        meta['per_alpha'][alpha] = {
            'q_mahal': q_mahal, 'log_V_mahal': log_V_mahal,
            'n_samples': int(total_n), 'mahal_self_check': float(worst_check),
            'groups': group_info,
        }

    results['_meta'] = meta
    return results


def containment_check(scores_test, mahal_scores_test, thresholds, alphas,
                      d, kappa=3.0, mahal_name='Mahal', safety=1.15):
    """Detect regions that leak outside the proposal ellipsoid."""
    mahal_scores_test = np.asarray(mahal_scores_test, dtype=float)

    def _kappa_for(nm):
        if isinstance(kappa, dict):
            return float(kappa.get(nm, 3.0))
        return float(kappa)

    out = {name: {} for name in scores_test}

    for alpha in alphas:
        q_mahal = float(thresholds[mahal_name][alpha])

        for name, s in scores_test.items():
            q_prop = (_kappa_for(name) ** (2.0 / d)) * q_mahal
            outside = mahal_scores_test > q_prop
            n_outside = int(np.count_nonzero(outside))
            s = np.asarray(s, dtype=float)
            inside_region = s <= thresholds[name][alpha]
            leaks = int(np.count_nonzero(outside & inside_region))

            if np.any(inside_region) and q_mahal > 0:
                reach = float(mahal_scores_test[inside_region].max())
                need_radius = safety * reach / q_mahal
                kappa_req = float(max(need_radius, 1.0) ** (d / 2.0))
            else:
                reach = 0.0
                kappa_req = 1.0

            out[name][alpha] = {
                'n_outside': n_outside,
                'leaks': leaks,
                'leak_frac': float(leaks / n_outside) if n_outside > 0 else 0.0,
                'reach_over_qmahal': float(reach / q_mahal) if q_mahal > 0 else 0.0,
                'kappa_required': kappa_req,
            }

    return out
