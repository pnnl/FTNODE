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

import numpy as np
import torch

from ftnode.experiments.cli import parse_seeds
from ftnode.physical.config import (
    PhysicalConfig,
    build_cold_structured,
    build_cold_unstructured,
    build_fim_structured,
    build_fim_unstructured,
)
from ftnode.physical.train import rollout_q
from ftnode.systems import DuffingDataConfig, make_dataset

# cell number -> (dir name, builder)
CELLS = {
    1: ("cold-unstructured", build_cold_unstructured),
    2: ("cold-structured", build_cold_structured),
    3: ("fim-unstructured", build_fim_unstructured),
    4: ("fim-structured", build_fim_structured),
}


def _score(model, test_cfgs, L, h, device):
    model.eval()
    per = {}
    with torch.no_grad():
        for ts, cfg in test_cfgs:
            ds = make_dataset(cfg).to(device)
            if hasattr(model, "prepare"):
                model.prepare(ds.W)
            qhat = rollout_q(model, ds.W, ds.U, L, h)
            mse = ((qhat - ds.Y) ** 2).mean().item()
            per[ts] = mse if np.isfinite(mse) else float("nan")
    vals = np.array(list(per.values()), float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    d = d[np.isfinite(d)]
    return None if d.size == 0 else {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std())}


def _paired2(a, b, c, dd):
    seeds = sorted(set(a) & set(b) & set(c) & set(dd))
    d = np.array([(a[s] - b[s]) - (c[s] - dd[s]) for s in seeds], float)
    d = d[np.isfinite(d)]
    return None if d.size == 0 else {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--cells", default="1,2,3,4")
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
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cells = [int(c) for c in args.cells.split(",")]
    seeds = parse_seeds(args.seeds)
    mcfg = PhysicalConfig(tau=args.tau, h=args.h)
    val = DuffingDataConfig(n_traj=args.val_ntraj, L=args.L_eval, tau=args.tau, h=args.h,
                            u_range=args.u_range, x_range=args.x_range, delta=args.delta, seed=1)
    test_cfgs = [(ts, dataclasses.replace(val, seed=ts)) for ts in parse_seeds(args.test_seeds)]

    scores: dict[str, dict[int, float]] = {}
    for cell in cells:
        name, build = CELLS[cell]
        for seed in seeds:
            ckpt = args.run_dir / name / f"seed{seed}.pth"
            if not ckpt.exists():
                continue
            torch.manual_seed(seed)  # match the head-init RNG used at build time
            model = build(mcfg).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            mse = _score(model, test_cfgs, args.L_eval, args.h, device)
            scores.setdefault(name, {})[seed] = mse
            print(f"{name} seed{seed}: test {mse:.4e}")

    by = {CELLS[c][0]: scores.get(CELLS[c][0], {}) for c in CELLS}
    c1, c2, c3, c4 = (by[CELLS[k][0]] for k in (1, 2, 3, 4))
    contrasts = {
        "structure_cold (cell2-cell1)": _paired(c2, c1),
        "structure_fim (cell4-cell3)": _paired(c4, c3),
        "pretraining (cell3-cell1)": _paired(c3, c1),
        "interaction ((cell4-cell3)-(cell2-cell1))": _paired2(c4, c3, c2, c1),
    }
    print("\n=== contrasts (paired over seeds, test MSE) ===")
    for k, v in contrasts.items():
        print(f"  {k}: " + ("--" if v is None else f"{v['mean']:+.4e} +/- {v['std']:.3e} (n={v['n']})"))

    out = args.run_dir / "score_eval.json"
    with open(out, "w") as fh:
        json.dump({"scores": {k: {str(s): v for s, v in d.items()} for k, d in scores.items()},
                   "contrasts": contrasts}, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
