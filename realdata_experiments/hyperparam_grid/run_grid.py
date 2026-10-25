#!/usr/bin/env python
"""Lengthscale x gamma grid for MKS, one seed per call."""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets as DS
from run_experiment_new import run_single, SCORER_FACTORIES
from run_common import load_dataset, SPLIT, make_serializable, print_sanity

import grid_scorers as G


def main():
    ap = argparse.ArgumentParser(description="MKS (ell, gamma) selection grid.")
    ap.add_argument('seed', type=int)
    ap.add_argument('model', choices=['Ridge', 'MLP'], nargs='?',
                    default='Ridge')
    ap.add_argument('--run', default='house_4d', help='dataset key in datasets.py')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1])
    ap.add_argument('--ell-mults', type=float, nargs='+', default=None)
    ap.add_argument('--gamma-qs', type=float, nargs='+', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--target-rel-se', type=float, default=0.02)
    ap.add_argument('--max-samples', type=int, default=400_000)
    ap.add_argument('--max-kappa', type=float, default=20000.0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    alphas = sorted(args.alphas)

    names = G.register(SCORER_FACTORIES, args.ell_mults, args.gamma_qs)
    ell_mults = args.ell_mults or G.ELL_MULTS
    gamma_qs = args.gamma_qs or G.GAMMA_QS

    out_dir = args.out_dir or os.path.join(_HERE, 'results')
    outfile = os.path.join(out_dir, f'seed_{args.seed:04d}_{args.model}.json')
    if os.path.exists(outfile) and not args.overwrite:
        print(f"  {outfile} exists, skipping (--overwrite to force)")
        return

    cfg = DS.get(args.run)
    print('=' * 88)
    print(f"  HYPERPARAMETER GRID: {args.run}  (d = {cfg['d']})")
    print(f"  Seed {args.seed}   Model {args.model}   alphas {alphas}")
    print(f"  ell mults   {ell_mults}   (x median/2; 1 = adopted)")
    print(f"  gamma quant {gamma_qs}    (spectrum percentile; 90 = adopted)")
    print(f"  {len(names)} grid cells + Mahalanobis")
    print(f"  Node: {os.environ.get('SLURMD_NODENAME', 'local')}")
    print('=' * 88, flush=True)

    X, Y, feat_cols = load_dataset(cfg, data_dir=args.data_dir)
    print(f"  Loaded X={X.shape}, Y={Y.shape}", flush=True)

    t0 = time.time()
    result = run_single(
        X, Y,
        seed=args.seed,
        model_name=args.model,
        split=SPLIT,
        alphas=alphas,
        scorer_names=names,
        max_kappa=args.max_kappa,
        volume_kwargs=dict(target_rel_se=args.target_rel_se,
                           max_samples=args.max_samples),
        verbose=True,
    )
    elapsed = time.time() - t0

    print_sanity(result, alphas, result['meta']['scorers'])

    payload = {
        'experiment': 'hyperparam_grid',
        'dataset': args.run,
        'seed': args.seed,
        'model': args.model,
        'split': list(SPLIT),
        'alphas': alphas,
        'ell_mults': list(ell_mults),
        'gamma_qs': list(gamma_qs),
        'adopted': list(G.ADOPTED),
        'd': cfg['d'],
        'p': len(feat_cols),
        'result': make_serializable(result),
        'elapsed_s': elapsed,
    }
    os.makedirs(out_dir, exist_ok=True)
    tmp = outfile + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(make_serializable(payload), f, indent=2)
    os.replace(tmp, outfile)
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved {outfile}", flush=True)

    def _cell(md, a):
        return md[a] if a in md else md[str(a)]

    try:
        for a in alphas:
            for metric, label in (('volume_ratio', 'VOLUME RATIO to Mahalanobis'),
                                  ('coverage', 'COVERAGE')):
                print(f"\n  {label}   alpha = {a}")
                print("    " + f"{'ell x':>7}" +
                      "".join(f"{('q' + str(int(q))):>11}" for q in gamma_qs))
                for m in ell_mults:
                    row = f"    {m:>7g}"
                    for q in gamma_qs:
                        v = _cell(result[G.grid_name(m, q)], a).get(metric, float('nan'))
                        mark = '*' if (m, int(q)) == G.ADOPTED else ' '
                        row += f"{v:>10.4f}{mark}"
                    print(row)
        print("\n    (* = the configuration reported in the paper)", flush=True)
    except Exception as e:
        print(f"\n  (summary print failed: {e}; the JSON above is complete)")


if __name__ == '__main__':
    main()
