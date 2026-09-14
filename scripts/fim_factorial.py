#!/usr/bin/env python3
"""Driver for the FIM x ftnode-structure 2x2 factorial on physical-state Duffing.

The four cells cross {cold, FIM-backbone} init with {unstructured, structured} field,
all in the physical state ``x = (q, q_dot)`` on one shared pipeline (see
``ftnode.physical`` and the plan).  Each ``(cell, seed)`` is reseeded immediately
before its build, trained with :func:`ftnode.physical.train.train_physical`, and its
best-validation checkpoint is scored on held-out test splits.

The cold cells (1, 2) run now.  The FIM cells (3, 4) need the vendored OpenFIM
backbone (``ftnode.fim``); until it is present their builders raise
``NotImplementedError`` and the driver reports them as skipped rather than failing.

Usage::

    python scripts/fim_factorial.py --cells all --seeds 0-9 --epochs 600
    python scripts/fim_factorial.py --cells 1,2 --seeds 0 --epochs 20 --n-traj 64 --val-ntraj 16
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
from ftnode.physical.train import PhysTrainConfig, rollout_q, train_physical
from ftnode.systems import DuffingDataConfig, make_dataset
from ftnode.train import restore_best

# cell number -> (name, builder, needs_fim)
CELLS = {
    1: ("cold-unstructured", build_cold_unstructured, False),
    2: ("cold-structured", build_cold_structured, False),
    3: ("fim-unstructured", build_fim_unstructured, True),
    4: ("fim-structured", build_fim_structured, True),
}


def _parse_cells(spec: str) -> list[int]:
    if spec.strip() == "all":
        return [1, 2, 3, 4]
    out = [int(s) for s in spec.split(",")]
    for c in out:
        if c not in CELLS:
            raise ValueError(f"unknown cell {c}; choose from {sorted(CELLS)} or 'all'")
    return out


def _data_configs(args):
    """Train, validation, and (seed-swapped) test data configs.

    Every split shares geometry and step ``h``; only the seed differs.  The test
    seeds must be disjoint from the train and validation seeds.
    """
    common = dict(tau=args.tau, h=args.h, u_range=args.u_range, x_range=args.x_range,
                  delta=args.delta)
    train = DuffingDataConfig(n_traj=args.n_traj, L=args.L, seed=args.train_seed, **common)
    val = DuffingDataConfig(n_traj=args.val_ntraj, L=args.L_eval, seed=args.val_seed, **common)
    test_seeds = parse_seeds(args.test_seeds)
    forbidden = {args.train_seed, args.val_seed}
    bad = sorted(set(test_seeds) & forbidden)
    if bad:
        raise ValueError(f"test seeds {bad} collide with train/val seeds {sorted(forbidden)}")
    return train, val, test_seeds


def _score(model, test_cfgs, L, h, device):
    """Per-test-seed measured-q MSE, plus mean and std across seeds."""
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
    return {
        "test_mse_by_seed": per,
        "test_mse_mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
        "test_mse_std": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
    }


def _contrasts(scores):
    """Simple effects and the interaction, paired over model seeds.

    ``scores[cell_name][seed]`` is a per-seed test-mean MSE.  The interaction is
    ``(cell4 - cell3) - (cell2 - cell1)``, the two simple effects are
    ``cell2 - cell1`` and ``cell4 - cell3``, and the pretraining effect is
    ``cell3 - cell1``.  Each is reported as a paired mean +/- std over seeds that have
    all the cells it needs.
    """
    by = {CELLS[c][0]: scores.get(CELLS[c][0], {}) for c in CELLS}
    c1, c2, c3, c4 = (by[CELLS[k][0]] for k in (1, 2, 3, 4))

    def paired(a, b):
        seeds = sorted(set(a) & set(b))
        d = np.array([a[s] - b[s] for s in seeds], float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
        return {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std())}

    def paired2(a, b, cc, dd):
        seeds = sorted(set(a) & set(b) & set(cc) & set(dd))
        d = np.array([(a[s] - b[s]) - (cc[s] - dd[s]) for s in seeds], float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
        return {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std())}

    return {
        "structure_cold (cell2-cell1)": paired(c2, c1),
        "structure_fim (cell4-cell3)": paired(c4, c3),
        "pretraining (cell3-cell1)": paired(c3, c1),
        "interaction ((cell4-cell3)-(cell2-cell1))": paired2(c4, c3, c2, c1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="all", help="cells to run: 'all' or e.g. '1,2'")
    ap.add_argument("--seeds", nargs="+", default=["0-9"], help="model seeds, e.g. 0-9 or 0 1 2")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/fim_factorial"))
    # Data geometry (defaults match the kappa_sat reference).
    ap.add_argument("--n-traj", type=int, default=512)
    ap.add_argument("--val-ntraj", type=int, default=64)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--L-eval", type=int, default=600)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--u-range", type=float, default=0.25)
    ap.add_argument("--x-range", type=float, default=1.6)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=1)
    ap.add_argument("--test-seeds", nargs="+", default=["1000-1009"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after N epochs with no val improvement (default: off)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cells = _parse_cells(args.cells)
    seeds = parse_seeds(args.seeds)
    mcfg = PhysicalConfig(tau=args.tau, h=args.h)
    train_cfg, val_cfg, test_seeds = _data_configs(args)
    test_cfgs = [(ts, dataclasses.replace(val_cfg, seed=ts)) for ts in test_seeds]

    train = make_dataset(train_cfg).to(device)
    val = make_dataset(val_cfg).to(device)
    print(f"[fim_factorial] cells {cells} seeds {seeds} on {device} -> {args.out}")
    print(f"[fim_factorial] train {tuple(train.W.shape)}  val {tuple(val.W.shape)}  "
          f"test seeds {test_seeds}")

    scores: dict[str, dict[int, float]] = {}
    skipped = []
    for cell in cells:
        name, build, needs_fim = CELLS[cell]
        cell_dir = args.out / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                model = build(mcfg).to(device)
            except NotImplementedError as e:
                if name not in dict(skipped):
                    skipped.append((name, str(e)))
                continue
            tcfg = PhysTrainConfig(
                n_epochs=args.epochs, lr=args.lr,
                lr_backbone=args.lr_backbone if needs_fim else None,
                batch=args.batch, L=args.L, L_eval=args.L_eval, h=args.h,
                patience=args.patience,
            )
            ckpt = cell_dir / f"seed{seed}.pth"
            model, hist = train_physical(
                model, train, val, tcfg, ckpt_path=ckpt,
                label=f"{name}-s{seed}", device=device,
            )
            restore_best(model, hist, device, verbose=False)
            with open(cell_dir / f"seed{seed}.hist.json", "w") as fh:
                json.dump(hist, fh, indent=1)
            sc = _score(model, test_cfgs, args.L_eval, args.h, device)
            scores.setdefault(name, {})[seed] = sc["test_mse_mean"]
            print(f"[{name}-s{seed}] best_val {hist['best_val']:.3e} @ ep {hist['best_epoch']}  "
                  f"test {sc['test_mse_mean']:.3e} +/- {sc['test_mse_std']:.3e}")

    contrasts = _contrasts(scores)
    payload = {
        "cells_run": [CELLS[c][0] for c in cells],
        "seeds": seeds,
        "test_seeds": test_seeds,
        "epochs": args.epochs,
        "scores": {k: {str(s): v for s, v in d.items()} for k, d in scores.items()},
        "contrasts": contrasts,
        "skipped": skipped,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "test_eval.json", "w") as fh:
        json.dump(payload, fh, indent=1)

    print("\n=== contrasts (paired over seeds, test MSE) ===")
    for k, v in contrasts.items():
        print(f"  {k}: " + ("--" if v is None else f"{v['mean']:+.4e} +/- {v['std']:.3e} (n={v['n']})"))
    if skipped:
        print("\nskipped cells (FIM backbone not vendored):")
        for name, msg in skipped:
            print(f"  {name}: {msg.splitlines()[0]}")
    print(f"\nwrote {args.out / 'test_eval.json'}")


if __name__ == "__main__":
    main()
