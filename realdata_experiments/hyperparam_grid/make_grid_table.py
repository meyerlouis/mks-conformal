"""LaTeX tables for the lengthscale x gamma grid."""

import glob
import json

import numpy as np
import pandas as pd

def _parse(name):
    if not name.startswith('G_x'):
        return None
    ell_s, _, q_s = name[3:].partition('_q')
    try:
        return float(ell_s.replace('p', '.')), int(q_s)
    except ValueError:
        return None


def load_grid(glob_pat, alpha="0.1"):
    """Long df: seed, model, ell_mult, gamma_q, coverage, volume_ratio, ell, gamma."""
    rows = []
    for path in sorted(glob.glob(glob_pat)):
        top = json.load(open(path))
        R = top.get('result', top)
        for name, md in R.items():
            pg = _parse(name) if isinstance(name, str) else None
            if pg is None or not isinstance(md, dict):
                continue
            m, q = pg
            cell = md.get(alpha) or md.get(str(float(alpha)))
            if not isinstance(cell, dict):
                continue
            par = md.get('params', {})
            rows.append(dict(
                seed=top.get('seed'), model=top.get('model'),
                ell_mult=m, gamma_q=q,
                coverage=cell.get('coverage'),
                volume_ratio=cell.get('volume_ratio'),
                ell=par.get('lengthscale'), gamma=par.get('gamma'),
            ))
    return pd.DataFrame(rows)


def grid_summary(df, metric="volume_ratio", stat="mean"):
    """Wide table: rows ell_mult (desc), cols gamma_q (asc)."""
    g = df.groupby(['ell_mult', 'gamma_q'])[metric]
    if stat == "mean":
        v = g.mean()
    elif stat == "std":
        v = g.std(ddof=1)
    elif stat == "se":
        v = g.std(ddof=1) / np.sqrt(g.count())
    else:
        raise ValueError(stat)
    return v.unstack('gamma_q').sort_index(ascending=False)


def _ell_label(m):
    if m >= 1:
        return f"${m:g}$"
    inv = 1.0 / m
    return f"$1/{inv:g}$"


def _cell(v, e, dp):
    if not np.isfinite(v):
        return "---"
    core = f"{v:.{dp}f}"
    if e is not None and np.isfinite(e):
        core += f"{{\\tiny$\\pm${e:.{dp}f}}}"
    return core


def _panel(piv, err, dp, qs, ells):
    out = []
    for m in ells:
        cells = []
        for q in qs:
            e = err.loc[m, q] if err is not None else None
            cells.append(_cell(piv.loc[m, q], e, dp))
        out.append(f"    {_ell_label(m)} & {' & '.join(cells)} \\\\")
    return "\n".join(out)


def latex_grid(df, dataset="House $d{=}4$", alpha="0.1", dp=3, pm="std",
               gap="1.5em", label="hpgrid"):
    """Volume ratio (left) and coverage (right) in one table."""
    vol = grid_summary(df, "volume_ratio")
    cov = grid_summary(df, "coverage")
    if vol.empty:
        return "% no rows"
    err = grid_summary(df, "volume_ratio", pm) if pm else None
    n_seeds = df.seed.nunique()
    ells, qs = list(vol.index), list(vol.columns)

    hdr = " & ".join(f"${q}$" for q in qs)
    spec = "l " + " ".join(["c"] * len(qs))
    nq = len(qs)

    left = _panel(vol, err, dp, qs, ells)
    right = _panel(cov, None, 4, qs, ells)

    return f"""\\begin{{table}}[h!]
\\small
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}[t]{{{spec}}}
    \\toprule
    & \\multicolumn{{{nq}}}{{c}}{{Volume ratio to Mahalanobis}} \\\\
    \\cmidrule(lr){{2-{nq+1}}}
    $\\ell \\,/\\, (\\mathrm{{med}}/2)$ & {hdr} \\\\
    \\midrule
{left}
    \\bottomrule
\\end{{tabular}}%
\\hspace{{{gap}}}%
\\begin{{tabular}}[t]{{{spec}}}
    \\toprule
    & \\multicolumn{{{nq}}}{{c}}{{Coverage}} \\\\
    \\cmidrule(lr){{2-{nq+1}}}
    $\\ell \\,/\\, (\\mathrm{{med}}/2)$ & {hdr} \\\\
    \\midrule
{right}
    \\bottomrule
\\end{{tabular}}}}
\\vspace{{0.5em}}
\\caption{{{dataset}, $1{{-}}\\alpha{{=}}{1-float(alpha):g}$, {n_seeds} seeds.
Rows: kernel lengthscale as a multiple of the median heuristic
$\\mathrm{{med}}/2$. Columns: $\\gamma$ at the given percentile of the
centred-Gram spectrum. Left: volume ratio to Mahalanobis. Right: empirical
coverage, target ${1-float(alpha):g}$, which is unaffected by either parameter.}}
\\label{{tab:{label}}}
\\end{{table}}"""
