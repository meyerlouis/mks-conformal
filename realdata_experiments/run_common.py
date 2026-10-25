"""Load a dataset configuration, run one seed and one model, save the result."""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets as DS
from run_experiment_new import run_single, DEFAULT_ALPHAS

SPLIT = (0.5, 0.25, 0.25)


def load_dataset(cfg, data_dir=None, verbose=True):
    """Load, coerce to numeric, drop NaN rows, check the targets are usable."""
    csv = cfg['csv']
    if data_dir:
        csv = str(Path(data_dir) / 'feldman' / Path(csv).name)

    df = pd.read_csv(csv, header=cfg['header'])

    if cfg['drop']:
        df = df.drop(columns=[c for c in cfg['drop'] if c in df.columns],
                     errors='ignore')

    targets = list(cfg['targets'])
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"targets not in {csv}: {missing}")

    if cfg['features'] is None:
        feat_cols = [c for c in df.columns if c not in targets]
    else:
        feat_cols = list(cfg['features'])
        bad = [f for f in feat_cols if f not in df.columns]
        if bad:
            raise ValueError(f"features not in {csv}: {bad}")
        overlap = set(map(str, feat_cols)) & set(map(str, targets))
        if overlap:
            raise ValueError(f"column is both feature and target: {sorted(overlap)}")

    num = df[feat_cols + targets].apply(pd.to_numeric, errors='coerce')
    X = num[feat_cols].values.astype(np.float64)
    Y = num[targets].values.astype(np.float64)

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    n_dropped = int((~mask).sum())
    X, Y = X[mask], Y[mask]

    if verbose:
        print(f"  csv      {csv}", flush=True)
        print(f"  targets  ({len(targets)}) {targets}", flush=True)
        shown = feat_cols if len(feat_cols) <= 25 else feat_cols[:25]
        print(f"  features ({len(feat_cols)}) {shown}"
              f"{' ...' if len(feat_cols) > 25 else ''}", flush=True)
        if cfg['drop']:
            print(f"  dropped cols {cfg['drop']}", flush=True)
        print(f"  dropped {n_dropped} rows (NaN / non-numeric)", flush=True)

    check_targets(Y, targets)
    return X, Y, feat_cols


def check_targets(Y, targets, cond_fatal=1e10, cond_warn=1e6, corr_warn=0.999):
    """Check that the targets are non-constant and not collinear."""
    sd = Y.std(axis=0)
    dead = [t for t, s in zip(targets, sd) if s < 1e-10]
    if dead:
        raise ValueError(f"constant target columns (zero variance): {dead}")

    if Y.shape[1] < 2:
        return

    C = np.corrcoef(Y.T)
    if not np.all(np.isfinite(C)):
        raise ValueError("target correlation matrix has non-finite entries")

    A = np.abs(np.triu(C, 1))
    worst = float(A.max())
    eig = np.linalg.eigvalsh(C)
    cond = float(eig[-1] / max(eig[0], 1e-300))

    if cond > cond_fatal or eig[0] <= 0:
        i, j = np.unravel_index(np.argmax(A), C.shape)
        raise ValueError(
            f"targets are rank-deficient (condition {cond:.3e}, worst pair "
            f"{targets[i]!r}/{targets[j]!r} at |corr| = {worst:.6f}). "
            f"Drop one of the collinear columns."
        )

    if cond > cond_warn or worst > corr_warn:
        warnings.warn(
            f"targets are strongly correlated (max |corr| {worst:.4f}, "
            f"condition {cond:.3e})."
        )

    print(f"  target conditioning OK (max |corr| {worst:.4f}, cond {cond:.3e})",
          flush=True)


def print_sanity(result, alphas, scorer_names):
    """Print coverage, volume ratios and the volume diagnostics."""
    vm = result.get('volume_meta')
    print()
    print('  ' + '-' * 86)
    print('  SANITY')
    print('  ' + '-' * 86)

    if vm:
        for alpha in alphas:
            pa = vm['per_alpha'][alpha]
            for k, g in sorted(pa.get('groups', {}).items()):
                chk, n = g['mahal_self_check'], g['n_samples']
                se = np.sqrt(max(k - 1.0, 0.0) / max(n, 1))
                z = abs(chk - 1.0) / se if se > 0 else 0.0
                flag = 'OK' if z < 3.0 else 'FAIL'
                print(f"    alpha={alpha:<5} kappa={k:>9.1f} N={n:>7}  "
                      f"self-check={chk:.4f} +-{se:.4f} ({z:.1f} sigma)  {flag}")

    sampled = [n for n in scorer_names
               if result[n][alphas[0]].get('volume_method') != 'analytic']

    leaks = [(n, a, result[n][a]['leaks'])
             for n in sampled for a in alphas
             if result[n][a].get('leaks')]
    for n, a, k in leaks:
        print(f"    LEAK {n} alpha={a}: {k}/{result[n][a]['n_outside']} "
              f"test residuals outside the proposal are inside the region")
    if not leaks:
        print('    containment OK (no leaks among sampled methods)')

    need = [result[n][a].get('kappa_required') or 0.0
            for n in sampled for a in alphas]
    if need and max(need) > 0:
        print("    proposal sizing (kappa used vs required):")
        for n in sampled:
            per = max(result[n][a].get('kappa_required') or 1.0 for a in alphas)
            reach = max(result[n][a].get('reach_over_qmahal') or 0.0 for a in alphas)
            used = result[n][alphas[0]].get('kappa_used')
            used_s = f"{used:.1f}" if used else "?"
            ok = 'OK' if (used and used >= per * (1 - 1e-6)) else 'TOO SMALL'
            print(f"      {n:<12s} reach {reach:.2f} x q_mahal  "
                  f"needs {per:>8.1f}  used {used_s:>8s}  {ok}")

    zero = [(n, a) for n in scorer_names for a in alphas
            if result[n][a].get('zero_hits')]
    if zero:
        print(f"    ZERO HITS at {zero}")

    print()
    for alpha in alphas:
        print(f"  alpha = {alpha}  (target coverage {1 - alpha:.3f})")
        print(f"    {'Scorer':<12s} {'Coverage':>10s} {'VolRatio':>11s} "
              f"{'+-SE':>9s} {'Method':>10s}")
        for n in scorer_names:
            r = result[n][alpha]
            print(f"    {n:<12s} {r['coverage']:>10.4f} "
                  f"{r.get('volume_ratio', float('nan')):>11.4f} "
                  f"{r.get('volume_ratio_se', float('nan')):>9.4f} "
                  f"{r.get('volume_method', '-'):>10s}")
        print()

    otcp = result.get('OTCP', {}).get('params', {})
    if otcp:
        print(f"    OT-CP: eps={otcp.get('eps_used')} "
              f"scale_mode={otcp.get('scale_mode')} "
              f"blur/data={otcp.get('blur_over_data_radius')} "
              f"iters={otcp.get('sinkhorn_iters')} "
              f"converged={otcp.get('converged')}")
        if otcp.get('converged') is False:
            print("    OT-CP Sinkhorn did not converge")
    print('  ' + '-' * 86, flush=True)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(name, argv=None):
    ap = argparse.ArgumentParser(
        description=f"{name}: one seed x one model.")
    ap.add_argument('seed', type=int)
    ap.add_argument('model', choices=['Ridge', 'MLP'])
    ap.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    ap.add_argument('--scorers', nargs='+', default=None,
                    help="subset of Mahal Bonferroni Density_med2 Kernel_med2 OTCP; "
                         "Mahal is always included")
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--kappa', type=float, default=3.0)
    ap.add_argument('--max-kappa', type=float, default=2000.0,
                    help='cap on the volume proposal inflation kappa')
    ap.add_argument('--target-rel-se', type=float, default=0.01)
    ap.add_argument('--max-samples', type=int, default=400_000)
    ap.add_argument('--block-size', type=int, default=8192)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    args.alphas = sorted(args.alphas)

    cfg = DS.get(name)
    out_dir = args.out_dir or f'results/{name}'
    outfile = os.path.join(out_dir, f'seed_{args.seed:04d}_{args.model}.json')

    if os.path.exists(outfile) and not args.overwrite:
        print(f"  {outfile} exists, skipping (--overwrite to force)")
        return

    print('=' * 88, flush=True)
    print(f"  Run:     {name}  (d = {cfg['d']})")
    print(f"  Seed:    {args.seed}")
    print(f"  Model:   {args.model}")
    print(f"  Alphas:  {args.alphas}")
    print(f"  Scorers: {args.scorers or 'all'}")
    print(f"  Node:    {os.environ.get('SLURMD_NODENAME', 'local')}")
    print('=' * 88, flush=True)

    X, Y, feat_cols = load_dataset(cfg, data_dir=args.data_dir)
    print(f"  Loaded X={X.shape}, Y={Y.shape}", flush=True)

    t0 = time.time()
    result = run_single(
        X, Y,
        seed=args.seed,
        model_name=args.model,
        split=SPLIT,
        alphas=args.alphas,
        scorer_names=args.scorers,
        max_kappa=args.max_kappa,
        volume_kwargs=dict(
            kappa=args.kappa,
            target_rel_se=args.target_rel_se,
            max_samples=args.max_samples,
            block_size=args.block_size,
        ),
        verbose=True,
    )
    elapsed = time.time() - t0

    print_sanity(result, args.alphas, result['meta']['scorers'])

    payload = {
        'dataset': name,
        'seed': args.seed,
        'model': args.model,
        'split': list(SPLIT),
        'alphas': args.alphas,
        'targets': [str(t) for t in cfg['targets']],
        'features': [str(f) for f in feat_cols],
        'd': cfg['d'],
        'p': len(feat_cols),
        'result': make_serializable(result),
        'elapsed_s': elapsed,
    }

    os.makedirs(out_dir, exist_ok=True)
    tmp = outfile + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, outfile)

    print(f"\n  Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Saved {outfile}", flush=True)
