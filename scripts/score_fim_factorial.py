#!/usr/bin/env python3
"""Score whatever cells/seeds are present in a fim_factorial run directory.

The driver scores in-process, so a run split across processes/GPUs never gets a
merged score.  This tool rebuilds each cell's model from its checkpoint, scores it on
held-out test splits, and computes the paired contrasts over the seeds that have all
the cells a contrast needs.  Missing checkpoints are skipped.

Usage::

    python scripts/score_fim_factorial.py runs/fim_factorial --seeds 0 --device cuda
    python scripts/score_fim_factorial.py runs/fim_factorial --cells 1,2 --seeds 0-2 --device cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import torch

from fim_factorial import CELLS, contrasts, score  # sibling script: one cell table
from ftnode.experiments.cli import parse_seeds
from ftnode.physical.config import PhysicalConfig
from ftnode.systems import DuffingDataConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--cells", default=",".join(map(str, sorted(CELLS))))
    ap.add_argument("--seeds", nargs="+", default=["0-2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--L-eval", type=int, default=600)
    ap.add_argument("--val-ntraj", type=int, default=64)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--u-range", type=float, default=0.25)
    ap.add_argument("--x-range", type=float, default=1.6)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--test-seeds", nargs="+", default=["1000-1009"])
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output JSON (default: <run_dir>/score_eval.json)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cells = [int(c) for c in args.cells.split(",")]
    seeds = parse_seeds(args.seeds)
    mcfg = PhysicalConfig(tau=args.tau, h=args.h)
    val = DuffingDataConfig(n_traj=args.val_ntraj, L=args.L_eval, tau=args.tau, h=args.h,
                            u_range=args.u_range, x_range=args.x_range, delta=args.delta, seed=1)
    test_seeds = parse_seeds(args.test_seeds)
    test_cfgs = [(ts, dataclasses.replace(val, seed=ts)) for ts in test_seeds]

    scores: dict[str, dict[int, float]] = {}
    metrics: dict[str, dict[int, dict]] = {}
    for cell in cells:
        name, build, _ = CELLS[cell]
        for seed in seeds:
            ckpt = args.run_dir / name / f"seed{seed}.pth"
            if not ckpt.exists():
                continue
            torch.manual_seed(seed)  # match the head-init RNG used at build time
            model = build(mcfg).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            sc = score(model, test_cfgs, args.L_eval, args.h, device)
            scores.setdefault(name, {})[seed] = sc["test_mse_mean"]
            metrics.setdefault(name, {})[seed] = m = sc["metrics"]
            nan = float("nan")
            print(f"{name} seed{seed}: test {sc['test_mse_mean']:.4e}  "
                  f"wrong wells {m.get('wrong_wells', '-')}/{m['n_traj']}  "
                  f"early {m.get('mse_0-50', nan):.3e}  "
                  f"transient {m.get('transient_mse', nan):.3e}  "
                  f"osc {m.get('osc_energy_ratio', nan):.2f}")

    con = contrasts(scores)
    print("\n=== contrasts (paired over seeds, test MSE) ===")
    for k, v in con.items():
        print(f"  {k}: " + ("--" if v is None else f"{v['mean']:+.4e} +/- {v['std']:.3e} (n={v['n']})"))

    out = args.out or args.run_dir / "score_eval.json"
    with open(out, "w") as fh:
        json.dump({"test_seeds": test_seeds,
                   "scores": {k: {str(s): v for s, v in d.items()} for k, d in scores.items()},
                   "metrics": {k: {str(s): v for s, v in d.items()} for k, d in metrics.items()},
                   "contrasts": con}, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
