#!/usr/bin/env python3
"""Score finished ``ftnode-train`` runs on held-out test splits, and read the
within-run truncation signal.

This is a **read-only** analysis tool.  It never trains and never touches the
training loop or the experiment spec.  It exists because the runner only ever
measures a *validation* split (``data_val``); this scores the same best-validation
checkpoints on fresh, independently generated **test** datasets.

Two things come out:

1. **Held-out test MSE.**  For each test data seed it builds an independent dataset
   from the run's own ``data_val`` geometry with the seed swapped
   (``dataclasses.replace(run.spec.data_val, seed=ts)``), rolls every best
   checkpoint out over it, and takes the same measured-output MSE the trainer uses
   for validation (``((yhat - Y) ** 2).mean()``, see ``ftnode/train.py:170-171``),
   under ``model.eval()`` / ``torch.no_grad()``.  Many seeds are used by default so
   the score reflects test-data variability, not one 64-trajectory draw.

2. **Within-run truncation read.**  From each seed's per-epoch ``val_extrap``
   history it compares ``min(val_extrap[:split])`` against ``min(val_extrap[split:])``
   (``split`` default 200).  Because this is one trajectory under one schedule, the
   comparison is free of the confound that dogs comparing a 600-epoch run to a
   separate 200-epoch run: the two halves share the identical LR schedule.

Data seeds here are a **different axis** from the model seeds ``0..9`` that index the
checkpoints; a numeric collision (e.g. a test seed of 0) would still be harmless
because dataset generation uses its own ``numpy`` generator, but the defaults sit in
a high block (``1000..1009``) to keep it obvious.

Usage::

    python scripts/eval_test_split.py runs/kappa_sat --test-seeds 1000-1009 --device cpu
    python scripts/eval_test_split.py runs/kappa_sat --baseline runs/kappa_full_cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib

import numpy as np
import torch

from ftnode.experiments.cli import parse_seeds
from ftnode.experiments.run import load_run
from ftnode.systems import make_dataset
from ftnode.train import rollout_y


def _build_test_sets(reference_run, test_seeds, device):
    """One independent test dataset per seed, from the run's own ``data_val`` geometry.

    Swapping only the seed inherits ``n_traj``, ``L`` (= ``L_eval``), ``tau``, the
    ranges and ``delta``, so a test set matches the run it scores.  ``make_dataset``
    is a pure function of its config, so this is deterministic.
    """
    dv = reference_run.spec.data_val
    forbidden = {reference_run.spec.data_train.seed, dv.seed}
    sets = []
    for ts in test_seeds:
        if ts in forbidden:
            raise ValueError(
                f"test seed {ts} collides with a train/val DATA seed {sorted(forbidden)}; "
                "pick a disjoint block (default 1000-1009)"
            )
        cfg = dataclasses.replace(dv, seed=ts)
        sets.append((ts, make_dataset(cfg).to(device)))
    return sets


def _mse(model, ds, L, h):
    """Measured-output MSE of one model over one dataset -- the trainer's val metric."""
    model.eval()
    with torch.no_grad():
        yhat, _ = rollout_y(model, ds.W, ds.U, L, h)
        val = ((yhat - ds.Y) ** 2).mean().item()
    return val if math.isfinite(val) else float("nan")


def score_run(run, test_sets, *, device):
    """``slug -> [per-model-seed dict]`` with best_epoch/best_val and per-testset MSEs.

    A model seed with no checkpoint (never trained, or a diverged job) comes back with
    ``test_mse`` all ``NaN`` so a partial run still scores.
    """
    L, h = run.spec.train.L_eval, run.spec.train.h
    models = run.models(device)
    histories = run.histories
    out = {}
    for v in run.variants:
        rows = []
        for si, (seed, model) in enumerate(zip(run.seeds, models[v.slug])):
            hist = histories[v.slug][si]
            per_ts = {}
            for ts, ds in test_sets:
                per_ts[ts] = float("nan") if model is None else _mse(model, ds, L, h)
            vals = np.array(list(per_ts.values()), float)
            rows.append({
                "model_seed": seed,
                "best_epoch": None if hist is None else hist.get("best_epoch"),
                "best_val": None if hist is None else hist.get("best_val"),
                "test_mse_mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                "test_mse_std": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
                "test_mse_by_seed": per_ts,
            })
        out[v.slug] = rows
    return out


def truncation_read(run, split):
    """Within-run floor comparison per variant/seed: min val before vs after ``split``.

    ``improved`` is the drop in the validation floor achieved by the epochs past
    ``split``; a floor that is flat there is evidence the run had converged by then,
    a floor that keeps dropping is evidence the shorter budget was truncating.
    """
    histories = run.histories
    out = {}
    for v in run.variants:
        rows = []
        for si, seed in enumerate(run.seeds):
            hist = histories[v.slug][si]
            row = {"model_seed": seed, "best_epoch": None, "n_epochs": None,
                   "floor_pre": None, "floor_post": None, "improved": None,
                   "best_in_tail": None}
            if hist is not None and hist.get("val_extrap"):
                ve = np.array(hist["val_extrap"], float)
                n = len(ve)
                row["best_epoch"] = hist.get("best_epoch")
                row["n_epochs"] = n
                if n > split:
                    pre = float(np.nanmin(ve[:split]))
                    post = float(np.nanmin(ve[split:]))
                    row["floor_pre"] = pre
                    row["floor_post"] = post
                    row["improved"] = pre - post
                be = hist.get("best_epoch")
                row["best_in_tail"] = (be is not None) and (be >= 0.9 * n)
            rows.append(row)
        out[v.slug] = rows
    return out


def _fmt(x, spec="{:.4e}"):
    return "   --    " if x is None or (isinstance(x, float) and math.isnan(x)) else spec.format(x)


def print_report(run, scores, trunc, baseline_scores=None):
    print(f"\n=== held-out test MSE  ({run.spec.name}) ===")
    print(f"{'variant':20s} {'seed':>4s} {'best_ep':>7s} {'best_val':>11s} {'test_mean':>11s} {'test_std':>10s}")
    for v in run.variants:
        vals = []
        for r in scores[v.slug]:
            vals.append(r["test_mse_mean"])
            print(f"{v.slug:20s} {r['model_seed']:>4d} {str(r['best_epoch']):>7s} "
                  f"{_fmt(r['best_val'])} {_fmt(r['test_mse_mean'])} {_fmt(r['test_mse_std'],'{:.3e}')}")
        arr = np.array(vals, float)
        if np.isfinite(arr).any():
            print(f"{'  -> across seeds':20s} {'':>4s} {'':>7s} {'':>11s} "
                  f"median {np.nanmedian(arr):.4e}  min {np.nanmin(arr):.4e}  max {np.nanmax(arr):.4e}")

    print(f"\n=== within-run truncation read (split at epoch {trunc['_split']}) ===")
    print(f"{'variant':20s} {'seed':>4s} {'best_ep':>7s} {'floor<=split':>13s} {'floor>split':>12s} {'improved':>11s} {'tail?':>6s}")
    for v in run.variants:
        for r in trunc[v.slug]:
            print(f"{v.slug:20s} {r['model_seed']:>4d} {str(r['best_epoch']):>7s} "
                  f"{_fmt(r['floor_pre'])} {_fmt(r['floor_post'])} {_fmt(r['improved'])} "
                  f"{str(r['best_in_tail']):>6s}")

    if baseline_scores is not None:
        print("\n=== paired test MSE delta  (this run - baseline), per model seed ===")
        print(f"{'variant':20s} {'seed':>4s} {'this':>11s} {'baseline':>11s} {'delta':>11s}")
        for v in run.variants:
            base = {r["model_seed"]: r["test_mse_mean"] for r in baseline_scores.get(v.slug, [])}
            for r in scores[v.slug]:
                b = base.get(r["model_seed"])
                d = (r["test_mse_mean"] - b) if (b is not None and math.isfinite(b)
                                                 and math.isfinite(r["test_mse_mean"])) else None
                print(f"{v.slug:20s} {r['model_seed']:>4d} {_fmt(r['test_mse_mean'])} "
                      f"{_fmt(b)} {_fmt(d)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a finished run on held-out test splits.")
    ap.add_argument("run_dir", type=pathlib.Path, help="run directory containing run.yaml")
    ap.add_argument("--test-seeds", nargs="+", default=["1000-1009"],
                    help="DATA seeds for the test splits; accepts 1000-1009 or a list (default 1000-1009)")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu, to match committed numbers)")
    ap.add_argument("--split-epoch", type=int, default=200,
                    help="epoch to split the within-run floor comparison at (default 200)")
    ap.add_argument("--baseline", type=pathlib.Path, default=None,
                    help="optional second run to score on the SAME test sets for a paired delta")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where to write the JSON report (default <run_dir>/test_eval.json)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    test_seeds = parse_seeds(args.test_seeds)

    run = load_run(args.run_dir)
    test_sets = _build_test_sets(run, test_seeds, device)
    scores = score_run(run, test_sets, device=device)
    trunc = truncation_read(run, args.split_epoch)
    trunc["_split"] = args.split_epoch

    baseline_scores = None
    if args.baseline is not None:
        baseline = load_run(args.baseline)
        # Reuse the primary run's test sets so the delta is paired on identical data;
        # this assumes matching data_val geometry, which the kappa_* family shares.
        baseline_scores = score_run(baseline, test_sets, device=device)

    print_report(run, scores, trunc, baseline_scores)

    out_path = args.out or (args.run_dir / "test_eval.json")
    payload = {
        "run": str(args.run_dir),
        "test_seeds": test_seeds,
        "split_epoch": args.split_epoch,
        "scores": scores,
        "truncation": {k: v for k, v in trunc.items() if k != "_split"},
    }
    if baseline_scores is not None:
        payload["baseline"] = str(args.baseline)
        payload["baseline_scores"] = baseline_scores
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
