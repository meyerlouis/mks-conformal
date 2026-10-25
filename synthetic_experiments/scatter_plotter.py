"""Contour plotting utilities for synthetic experiment residuals."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


SCORER_COLORS = {
    'Kernel':     '#d62728',   # red
    'Density':    '#2ca02c',   # green
    'Mahal':      '#1f77b4',   # blue
    'Bonferroni': '#ff7f0e',   # orange
}

ALPHA_ALPHAS = {
    0.1:  1.0,
    0.05: 0.75,
    0.02: 0.5,
    0.01: 0.35,
}


def _get_color(scorer_name):
    return SCORER_COLORS.get(scorer_name, '#7f7f7f')


def _get_line_alpha(alpha):
    return ALPHA_ALPHAS.get(alpha, 0.4)


def _make_grid(residuals_cal, pad=0.3, n_grid=200):
    """Create evaluation grid from calibration residuals."""
    eps = np.asarray(residuals_cal)
    if eps.ndim == 1:
        eps = eps[:, None]

    lo = eps.min(axis=0)
    hi = eps.max(axis=0)
    margin = (hi - lo) * pad
    lo -= margin
    hi += margin

    e1 = np.linspace(lo[0], hi[0], n_grid)
    e2 = np.linspace(lo[1], hi[1], n_grid)
    E1, E2 = np.meshgrid(e1, e2)
    grid_pts = np.column_stack([E1.ravel(), E2.ravel()])

    return E1, E2, grid_pts


def _score_grid(scorer, grid_pts, batch_size=5000):
    """Score grid points in batches to avoid OOM."""
    n = grid_pts.shape[0]
    scores = np.empty(n)
    for i in range(0, n, batch_size):
        scores[i:i+batch_size] = scorer.score(grid_pts[i:i+batch_size])
    return scores


def plot_contours_grid(out, models_order, scorers_order, xlim, ylim, alphas=None,
                       n_grid=200, pad=0.3, figsize=None, scatter_kw=None, show_ylabel=True):
    """One row of contour panels, one per (model, scorer)."""
    n_panels = len(models_order)
    assert len(scorers_order) == n_panels

    if alphas is None:
        first_model = models_order[0]
        first_scorer = list(out['results'][first_model].keys())[0]
        alphas = sorted(out['results'][first_model][first_scorer].keys(), reverse=True)

    if figsize is None:
        figsize = (3.8 * n_panels, 3.8)

    sharey_groups = []
    i = 0
    while i < n_panels:
        j = i + 1
        while j < n_panels and models_order[j] == models_order[i]:
            j += 1
        sharey_groups.append((i, j))
        i = j

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize,
        gridspec_kw={'wspace': 0.08},
    )
    if n_panels == 1:
        axes = [axes]

    for start, end in sharey_groups:
        for k in range(start + 1, end):
            axes[k].sharey(axes[start])

    skw = dict(s=0.9, alpha=0.35, c='black')
    if scatter_kw:
        skw.update(scatter_kw)

    for idx, (ax, mdl, scr) in enumerate(zip(axes, models_order, scorers_order)):
        rc = out['data']['residuals_cal'][mdl]
        E1, E2, grid_pts = _make_grid(rc, pad=pad, n_grid=n_grid)

        scorer = out['fitted_scorers'][mdl][scr]
        scores_grid = _score_grid(scorer, grid_pts).reshape(E1.shape)
        color = _get_color(scr)

        ax.scatter(rc[:, 0], rc[:, 1], **skw)

        for alpha in alphas:
            threshold = out['results'][mdl][scr][alpha]['threshold']
            line_alpha = _get_line_alpha(alpha)
            ax.contour(E1, E2, scores_grid, levels=[threshold],
                       colors=[to_rgba(color, line_alpha)], linewidths=1.5)

        ax.set_xlabel(r'$\varepsilon_1$', fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        is_group_start = any(s == idx for s, _ in sharey_groups)
        if is_group_start:
            if show_ylabel:
                ax.set_ylabel(r'$\varepsilon_2$', fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    fig.tight_layout()
    return fig


def run_and_plot(dgp_fn, models, scorer_factories, alphas, xlim, ylim,
                 seed=42, n=20_000, volume_n_samples=50_000, **kwargs):
    """Run one seed and plot contours."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from experiment import run_single_seed

    out = run_single_seed(
        dgp_fn=dgp_fn,
        models=models,
        scorer_factories=scorer_factories,
        alphas=alphas,
        n=n,
        seed=seed,
        volume_n_samples=volume_n_samples,
    )

    model_names = list(models.keys())
    scorer_names = list(scorer_factories.keys())
    models_order = [m for m in model_names for _ in scorer_names]
    scorers_order = [s for _ in model_names for s in scorer_names]
    fig = plot_contours_grid(out, models_order, scorers_order,
                             alphas=alphas, xlim=xlim, ylim=ylim, **kwargs)

    return fig, out
