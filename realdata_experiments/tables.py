"""LaTeX tables for the real-data results."""

import glob
import json
import math
import re

import numpy as np
import pandas as pd
from scipy import stats as _ss

_SKIP_METHOD = {"seed", "model", "model_rmse", "meta", "volume_meta"}
_SKIP_ALPHA = {"params", "score_summary"}

ALPHAS_DISPLAY = ["0.1", "0.05", "0.02", "0.01"]

ORDER = ["Bonferroni", "Mahal", "Density_med2", "OTCP", "Kernel_med2"]
PRETTY = {
    "Mahal": "Mahalanobis", "Bonferroni": "Bonferroni",
    "Density_med2": "Density (med$/2$)", "Kernel_med2": "MKS (med$/2$)",
    "OTCP": "OT-CP",
}
MKS = "Kernel_med2"


def build_df(experiments):
    """Load the result files {dataset: {d: glob}} into one long DataFrame."""
    rows = []
    for dataset, dims in experiments.items():
        for d, pattern in dims.items():
            for path in sorted(glob.glob(pattern)):
                data = json.load(open(path))
                R = data.get("result", data)
                seed, model = R.get("seed", data.get("seed")), R.get("model", data.get("model"))
                for method, mdata in R.items():
                    if method in _SKIP_METHOD or not isinstance(mdata, dict):
                        continue
                    for alpha, m in mdata.items():
                        if alpha in _SKIP_ALPHA or not isinstance(m, dict):
                            continue
                        try:
                            a = float(alpha)
                        except (TypeError, ValueError):
                            continue
                        rows.append({
                            "dataset": dataset, "d": int(d), "seed": seed, "model": model,
                            "method": method, "alpha": f"{a:g}",
                            "coverage": m.get("coverage"),
                            "volume": m.get("volume"),
                            "volume_ratio": m.get("volume_ratio"),
                            "wsc": m.get("wsc"),
                            "leaks": m.get("leaks") or 0,
                        })
    return pd.DataFrame(rows)


def _paired_ratio(df, num, den, model, dataset, d, alpha):
    """Per-seed volume ratio num / den: mean, std, n, p-value, wins."""
    sub = df[(df.model == model) & (df.dataset == dataset) & (df.d == d) & (df.alpha == alpha)]
    N = sub[sub.method == num][["seed", "volume"]].rename(columns={"volume": "vn"})
    D = sub[sub.method == den][["seed", "volume"]].rename(columns={"volume": "vd"})
    m = N.merge(D, on="seed")
    if not len(m):
        return None
    r = (m.vn / m.vd).values
    p = _ss.ttest_1samp(np.log(r), 0.0).pvalue if len(r) > 1 else np.nan
    return dict(mean=r.mean(), std=r.std(ddof=1) if len(r) > 1 else 0.0,
                n=len(r), p=p, wins=int((m.vn < m.vd).sum()))


def latex_main_table(df, model="Ridge", dataset_order=("Bio", "Blog", "House"),
                     dims=(2, 3, 4)):
    """Main real-data table: MKS / Mahalanobis volume ratio and MKS coverage."""
    alphas = ["0.1", "0.05", "0.02", "0.01"]
    cov = (df[(df.model == model) & (df.method == "Kernel_med2")]
           .groupby(["dataset", "d", "alpha"]).coverage.agg(["mean", "std"]).reset_index())

    def cov_cell(ds, d, a):
        g = cov[(cov.dataset == ds) & (cov.d == d) & (cov.alpha == a)]
        return (f"{g['mean'].values[0]:.5f}\\tiny{{$\\pm${g['std'].values[0]:.5f}}}"
                if len(g) else "---")

    def rat_cell(ds, d, a):
        r = _paired_ratio(df, "Kernel_med2", "Mahal", model, ds, d, a)
        return f"{r['mean']:.5f}\\tiny{{$\\pm${r['std']:.3f}}}" if r else "---"

    def block(ds, ds_dims):
        rows = []
        for j, d in enumerate(ds_dims):
            cells = [rat_cell(ds, d, a) for a in alphas] + [cov_cell(ds, d, a) for a in alphas]
            pre = (f"    \\multirow{{{len(ds_dims)}}}{{*}}{{{ds}}}\n    & {d}"
                   if j == 0 else f"    & {d}")
            rows.append(f"{pre} & {' & '.join(cells)} \\\\")
        return rows

    lines = []
    for i, ds in enumerate(dataset_order):
        if i:
            lines.append("    \\midrule")
        lines += block(ds, dims[ds] if isinstance(dims, dict) else dims)

    n = df[df.model == model].seed.nunique()
    body = "\n".join(lines)
    return f"""\\begin{{table}}[H]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{ll cccc cccc}}
    \\toprule
    & & \\multicolumn{{4}}{{c}}{{Volume Ratio (MKS\\,/\\,Mahalanobis)}} & \\multicolumn{{4}}{{c}}{{MKS Coverage}} \\\\
    \\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-10}}
    Dataset & $d$ & $1{{-}}\\alpha{{=}}.90$ & $.95$ & $.98$ & $.99$ & $.90$ & $.95$ & $.98$ & $.99$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}}}
  \\vspace{{0.5em}}
  \\caption{{Real-data results under {model} residuals, {n} seeds, with split
  calibration and $\\ell{{=}}$med$/2$. Left: per-seed volume ratio of MKS to
  Mahalanobis (below~1 = smaller MKS region). Right: MKS empirical coverage
  (target $1{{-}}\\alpha$).}}
  \\label{{tab:real_data}}
\\end{{table}}"""


def latex_appendix(df, dataset, d, caption="", label="", mks=MKS):
    sub = df[(df["dataset"] == dataset) & (df["d"] == d)]
    stats = sub.groupby(["model","method","alpha"])[["coverage","volume","wsc"]].agg(["mean","std"]).sort_index()
    means = stats.xs("mean", axis=1, level=1)[["coverage","volume","wsc"]]
    stds  = stats.xs("std",  axis=1, level=1)[["coverage","volume","wsc"]]
    best = means.groupby(["model","alpha"])["volume"].idxmin()

    models = [m for m in ["Ridge","MLP"] if m in sub["model"].unique()]
    methods = [m for m in ORDER if m in sub["method"].unique()]
    alphas = [a for a in ALPHAS_DISPLAY if a in sub["alpha"].unique()]

    f  = lambda m,s: f"{m:.5f}\\tiny{{$\\pm${s:.5f}}}"
    fb = lambda m,s: f"\\textbf{{{m:.5f}\\tiny{{$\\pm${s:.5f}}}}}"

    lines = []
    for ai, a in enumerate(alphas):
        if ai: lines.append("    \\midrule")
        for mi, meth in enumerate(methods):
            cells = []
            for mdl in models:
                idx = (mdl, meth, a)
                if idx not in means.index:
                    cells += ["---"]*3; continue
                mc,mv,mw = means.loc[idx]; sc,sv,sw = stds.loc[idx]
                isbest = best.get((mdl,a)) == idx
                cells += [f(mc,sc), fb(mv,sv) if isbest else f(mv,sv), f(mw,sw)]
            prefix = f"    \\multirow{{{len(methods)}}}{{*}}{{{a}}}" if mi==0 else "   "
            name = f"\\textbf{{{PRETTY[meth]}}}" if meth==mks else PRETTY[meth]
            lines.append(f"{prefix} & {name} & {' & '.join(cells)} \\\\")

    nm = len(models)
    mh = " & ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models)
    cm = " ".join(f"\\cmidrule(lr){{{3+i*3}-{5+i*3}}}" for i in range(nm))
    hh = " & ".join(["Coverage & Volume & WSC"]*nm)
    spec = "ll " + " ".join(["ccc"]*nm)
    body = "\n".join(lines)
    return f"""\\begin{{table}}[H]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{{spec}}}
    \\toprule
    & & {mh} \\\\
    {cm}
    $\\alpha$ & Method & {hh} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}}}
  \\vspace{{0.5em}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{table}}"""


CELL = re.compile(r'^\s*(?:\\textbf\{)?\s*([-\d.]+)\\tiny\{\$\\pm\$([-\d.]+)\}\}?\s*$')

def _parse(cell):
    m = CELL.match(cell)
    return (float(m.group(1)), float(m.group(2))) if m else None

def _fmt(mean, std, sig=4):
    """mean +- std with a shared power of ten."""
    e = int(math.floor(math.log10(abs(mean)))) if mean else 0
    if -1 <= e <= 2:
        dec = max(sig - 1 - e, 0)
        return f"{mean:.{dec}f}\\tiny{{$\\pm${std:.{dec}f}}}"
    m, s = mean / 10**e, std / 10**e
    return f"{m:.{sig-1}f}\\tiny{{$\\pm${s:.{sig-1}f}}}$\\times10^{{{e}}}$"

def compact_volumes(tex, vol_cols=(3, 6)):
    """Compact volume columns, smallest volume in bold."""
    lines, block, lines_cells = tex.split('\n'), [], {}

    def flush():
        for c in vol_cols:
            vals = [(i, _parse(cells[c])) for i, cells in block]
            vals = [(i, v) for i, v in vals if v is not None]
            if not vals:
                continue
            best = min(vals, key=lambda t: t[1][0])[0]
            for i, (mean, std) in vals:
                txt = _fmt(mean, std)
                lines_cells[i][c] = f" \\textbf{{{txt}}} " if i == best else f" {txt} "
        block.clear()

    for i, line in enumerate(lines):
        if '\\tiny' in line and ' & ' in line:
            cells = line.rsplit('\\\\', 1)[0].split('&')
            lines_cells[i] = cells
            block.append((i, cells))
        elif '\\midrule' in line or '\\bottomrule' in line:
            flush()
    flush()
    for i, cells in lines_cells.items():
        lines[i] = '&'.join(cells).rstrip() + ' \\\\'
    return '\n'.join(lines)
