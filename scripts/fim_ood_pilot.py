#!/usr/bin/env python3
"""Pilot for the OOD sweep budget: train a few cells long and log convergence diagnostics.

Sets the optimizer-step budget of ``scripts/fim_ood_sweep.py`` without touching test data.
Each cell trains on the split's training region and is logged every ``--log-every`` epochs:

- the training loss;
- the drift MSE against the true field on an in-distribution validation set (the training
  region, a different data seed) -- never the test region;
- for FIM cells, the relative backbone displacement ``||theta - theta_0|| / ||theta_0||``
  over the drift path, and the mean absolute change of the backbone drift from the zero-shot
  drift on the validation states (does the backbone move at all?).

Usage::

    python scripts/fim_ood_pilot.py --split trapped_lobe
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fim_ood_sweep as sweep  # noqa: E402

DEFAULT_CELLS = "C2-roll,C2-drift,M-R2-roll,S-R0-roll,S-R2-roll,S-R2-drift"


def _drift_path_params(backbone):
    u_ids = {id(p) for p in backbone.fim.u_model.parameters()}
    return [p for p in backbone.parameters() if id(p) not in u_ids]


@torch.no_grad()
def _backbone_change(backbone, ref, val, stride=10):
    """Mean |f_backbone - f_ref| over every ``stride``-th validation state (true states: eval only)."""
    backbone.prepare(val.W, sweep.H)
    ref.prepare(val.W, sweep.H)
    diffs = [(backbone.drift(val.Xfull[:, t]) - ref.drift(val.Xfull[:, t])).abs().mean()
             for t in range(0, val.Xfull.shape[1], stride)]
    return float(torch.stack(diffs).mean())


def run_cell(split, cell, args_dict):
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(args.threads)
    from ftnode.fim import quick
    from ftnode.fim.backbone import FIMBackbone
    from ftnode.systems import DuffingParams

    arm, scope, obj = sweep.CELLS[cell]
    train_region, _ = sweep.SPLITS[split]
    train = sweep._data(args, train_region, args.n_train, args.L_train, args.train_seed)
    val = sweep._data(args, train_region, args.n_val, args.L_train, args.val_seed)
    field = sweep._build(arm, scope, args.seed, lambda: FIMBackbone.from_pretrained(device="cpu", state_dim=2))
    fim = hasattr(field, "backbone")
    if fim:
        ref = quick.zero_shot_snapshot(field.backbone)
        quick.disable_backbone_dropout(ref)
        theta0 = [p.detach().clone() for p in _drift_path_params(field.backbone)]
        norm0 = float(torch.sqrt(sum((p ** 2).sum() for p in theta0)))

    log = []
    t0 = time.time()

    def on_epoch(ep, loss):
        if ep % args.log_every and ep != epochs - 1:
            return
        row = {"epoch": ep + 1, "step": (ep + 1) * args.steps_per_epoch, "train_loss": loss,
               "val_drift_mse": quick.drift_mse_vs_true(field, val, sweep.H, DuffingParams())["mean"],
               "wall_s": time.time() - t0}
        if fim:
            with torch.no_grad():
                d = torch.sqrt(sum(((p - q) ** 2).sum() for p, q in zip(_drift_path_params(field.backbone), theta0)))
            row["backbone_rel_disp"] = float(d) / norm0
            row["backbone_drift_change"] = _backbone_change(field.backbone, ref, val)
        log.append(row)

    epochs = args.epochs_cold if arm in ("C1", "C2") else args.epochs
    sweep._train(field, arm, obj, train, epochs, args, on_epoch=on_epoch)
    out = pathlib.Path(args.out) / split / f"{cell}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cell": cell, "split": split, "args": args_dict, "log": log}, indent=1))
    return cell, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="trapped_lobe", choices=sorted(sweep.SPLITS))
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--L-train", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1000, help="FIM cells (x steps-per-epoch steps)")
    ap.add_argument("--epochs-cold", type=int, default=2500)
    ap.add_argument("--steps-per-epoch", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--L-short", type=int, default=6)
    ap.add_argument("--n-times", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="runs/fim_ood/pilot")
    args = ap.parse_args(argv)
    cells = args.cells.split(",")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=torch.multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(run_cell, args.split, c, vars(args)): c for c in cells}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                _, log = fut.result()
                last = log[-1]
                print(f"[pilot] {c}: step {last['step']}  loss {last['train_loss']:.3g}  "
                      f"val drift {last['val_drift_mse']:.3g}  "
                      f"disp {last.get('backbone_rel_disp', float('nan')):.2e}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[pilot] {c} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
