#!/usr/bin/env python3
"""Breadth sweep: retune strategies of a pretrained FIM field on out-of-distribution Duffing splits.

Each split trains on one region and tests on another (see ``ftnode.ood_splits``):

- ``trapped_lobe``  -- train on right-well trapped states, test on left-well trapped states.
- ``crossing_half`` -- train on ``q0 > 0`` (some trajectories cross), test on ``q0 < 0``.
- ``far_field``     -- train in the standard box, test on ``|q0|`` in ``[2, 4]``.

A cell is an arm, a retune scope and an objective:

- arms: ``C1`` cold free MLP, ``C2`` cold ``A(x)(x - g)``, ``Z`` zero-shot FIM drift,
  ``M`` FIM + free MLP head on ``[drift, x, u]``, ``S`` FIM + ``A(x)(x - g)`` head.
  ``M`` and ``S`` use the matched head sizes of the factorial's cells 5 and 4.
- scopes (``M``, ``S``): ``R0`` head only, ``R1`` head + FIM decoder side, ``R2`` head +
  whole FIM drift path (:func:`ftnode.fim.quick.set_retune_scope`).
- objectives: ``roll`` short measured-``q`` rollout from random start windows, ``drift``
  drift matching on finite-difference targets, ``distill`` (``S``, ``R0`` only) fit of the
  head to the frozen zero-shot FIM drift.

Every training input comes from the measured ``q``; the FIM backbone runs with dropout
off.  Each cell trains up to a step cap (the same for every FIM scope and objective) and
is validated every few steps by the drift-matching loss on an in-distribution validation
set (the training region, another data seed; measured ``q`` only).  The best validated
state, including the untrained start, is restored before the test, so selection never
sees the test region.  Each ``(cell, seed)`` is scored on 600-step test rollouts with the
full rollout metric set, plus the drift MSE against the true field on the test and the
training trajectories.  One JSON (and an ``.npz`` of the predicted ``q``) per job.

Usage::

    python scripts/fim_ood_sweep.py --split trapped_lobe --seeds 0-4 --jobs 12 --threads 4
    python scripts/fim_ood_sweep.py --split trapped_lobe --smoke          # minutes, tiny
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from ftnode.experiments.cli import parse_seeds

H = 0.05

SPLITS = {
    "trapped_lobe": ("trapped_right", "trapped_left"),
    "crossing_half": ("half_right", "half_left"),
    "far_field": (None, "far_ring"),  # None: the standard box from make_dataset
}


def _cells():
    cells = {}
    for arm in ("C1", "C2"):
        for obj in ("roll", "drift"):
            cells[f"{arm}-{obj}"] = (arm, None, obj)
    cells["Z"] = ("Z", None, None)
    for arm in ("M", "S"):
        for scope in ("R0", "R1", "R2"):
            for obj in ("roll", "drift"):
                cells[f"{arm}-{scope}-{obj}"] = (arm, scope, obj)
    cells["S-R0-distill"] = ("S", "R0", "distill")
    return cells


CELLS = _cells()


class ZeroShotField(torch.nn.Module):
    """The bare pretrained FIM drift as a field: ``F(x, u) = f_FIM(x)``; ``u`` is in context."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def prepare(self, window, h):
        self.backbone.prepare(window, h)

    def F(self, x, u):
        return self.backbone.drift(x)


def _data(args, region, n, L, seed):
    from ftnode.ood_splits import SplitConfig, make_split
    from ftnode.systems import DuffingDataConfig, make_dataset

    if region is None:
        return make_dataset(DuffingDataConfig(n_traj=n, L=L, seed=seed))
    return make_split(SplitConfig(region=region, n_traj=n, L=L, seed=seed))


def _test_cfgs(args, region):
    from ftnode.ood_splits import SplitConfig

    return [(ts, SplitConfig(region=region, n_traj=args.n_test, L=args.L_eval, seed=ts))
            for ts in parse_seeds(args.test_seeds)]


def _build(arm, scope, seed, backbone_factory):
    from ftnode.fim import quick
    from ftnode.physical.config import (
        PhysicalConfig,
        build_cold_structured,
        build_cold_unstructured,
        build_fim_mlp_head,
        build_fim_structured,
    )

    cfg = PhysicalConfig()
    backbone = backbone_factory() if arm in ("Z", "M", "S") else None
    # Reseed after the backbone load (building FIMODE draws from the RNG) and before the head.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arm == "C1":
        return build_cold_unstructured(cfg).field
    if arm == "C2":
        return build_cold_structured(cfg).field
    if arm == "Z":
        from ftnode.fim.fields import freeze_uncertainty_head

        quick.set_backbone_trainable(backbone, False)
        freeze_uncertainty_head(backbone)
        quick.disable_backbone_dropout(backbone)
        return ZeroShotField(backbone)
    field = (build_fim_mlp_head if arm == "M" else build_fim_structured)(cfg, backbone).field
    quick.set_retune_scope(field, scope)
    return field


def _train(field, arm, obj, train, epochs, args, on_epoch=None):
    from ftnode.fim import quick

    if obj is None:
        return []
    batch = (len(train) + args.steps_per_epoch - 1) // args.steps_per_epoch
    if obj == "roll":
        cfg = quick.RetuneConfig(epochs=epochs, L_short=args.L_short, lr=args.lr,
                                 lr_backbone=args.lr_backbone, batch=batch, random_start=True)
        return quick.retune_short_rollout(field, train, H, cfg, on_epoch=on_epoch)[1]
    if obj == "drift":
        cfg = quick.DriftMatchConfig(epochs=epochs, lr=args.lr, lr_backbone=args.lr_backbone,
                                     batch=batch, n_times=args.n_times)
        return quick.retune_drift_match(field, train, H, cfg, on_epoch=on_epoch)[1]
    if obj == "distill":
        n_points = len(train) * (train.Y.shape[1])
        cfg = quick.DistillConfig(epochs=epochs, lr=args.lr,
                                  batch=(n_points + args.steps_per_epoch - 1) // args.steps_per_epoch)
        return quick.distill_structured(field, field.backbone, train, H, cfg, on_epoch=on_epoch)[1]
    raise ValueError(obj)


@torch.no_grad()
def val_drift_loss(field, val, X, Xdot) -> float:
    """Drift-matching loss on the validation set: measured ``q`` only, no true state.

    ``X, Xdot`` come from :func:`ftnode.fim.quick.drift_targets` on ``val``.  Each time is
    one query per trajectory, so the context and query batches stay aligned.
    """
    from ftnode.fim import quick

    quick.disable_backbone_dropout(field)
    prepare = getattr(field, "prepare", None)
    if prepare is not None:
        prepare(val.W, H)
    se = [((field.F(X[:, t], val.U) - Xdot[:, t]) ** 2).mean() for t in range(X.shape[1])]
    v = float(torch.stack(se).mean())
    return v if np.isfinite(v) else float("inf")


class BestTracker:
    """Validate every ``every`` optimizer steps; keep a copy of the best trainable state.

    Call :meth:`check` at step 0 and from the per-epoch hook; :meth:`restore` loads the
    best state back.  Only parameters with ``requires_grad`` change during a retune, so
    only those are copied.
    """

    def __init__(self, field, val, every, steps_per_epoch):
        from ftnode.fim import quick

        self.field, self.val, self.every, self.spe = field, val, every, steps_per_epoch
        self.X, self.Xdot = quick.drift_targets(val, H)
        self.params = {n: p for n, p in field.named_parameters() if p.requires_grad}
        self.curve, self.best, self.best_step, self.best_state = [], float("inf"), None, None
        self._next = 0

    def check(self, step):
        if step < self._next:
            return
        self._next = (step // self.every + 1) * self.every
        v = val_drift_loss(self.field, self.val, self.X, self.Xdot)
        self.curve.append([step, v])
        if v < self.best:
            self.best, self.best_step = v, step
            self.best_state = {n: p.detach().clone() for n, p in self.params.items()}

    def on_epoch(self, ep, loss):
        self.check((ep + 1) * self.spe)

    @torch.no_grad()
    def restore(self):
        if self.best_state is not None:
            for n, p in self.params.items():
                p.copy_(self.best_state[n])


def _drift_mse(field, datasets):
    from ftnode.fim import quick
    from ftnode.systems import DuffingParams

    vals = [quick.drift_mse_vs_true(field, ds, H, DuffingParams())["mean"] for ds in datasets]
    return float(np.mean(vals))


def run_job(split, cell, seed, args_dict):
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(args.threads)
    from ftnode.fim import quick
    from ftnode.fim.backbone import FIMBackbone
    from ftnode.ood_splits import make_split
    from ftnode.physical.metrics import rollout_metrics, rollout_test_split
    from ftnode.physical.model import PhysicalField

    out_dir = pathlib.Path(args.out) / split / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{seed}.json"
    arm, scope, obj = CELLS[cell]
    train_region, test_region = SPLITS[split]
    t0 = time.time()

    train = _data(args, train_region, args.n_train, args.L_train, args.train_seed)
    val = _data(args, train_region, args.n_val, args.L_train, args.val_seed)
    field = _build(arm, scope, seed, lambda: FIMBackbone.from_pretrained(device="cpu", state_dim=2))
    t1 = time.time()
    cold = arm in ("C1", "C2")
    max_steps = args.max_steps_cold if cold else args.max_steps
    tracker = BestTracker(field, val, args.val_every_cold if cold else args.val_every,
                          args.steps_per_epoch)
    if obj is not None:
        tracker.check(0)
    epochs = max_steps // args.steps_per_epoch
    losses = _train(field, arm, obj, train, epochs, args, on_epoch=tracker.on_epoch)
    tracker.restore()
    t2 = time.time()

    model = PhysicalField(field, H)
    if hasattr(field, "backbone"):
        quick.disable_backbone_dropout(field)
    test_cfgs = _test_cfgs(args, test_region)
    qhat, y, per_seed = rollout_test_split(model, test_cfgs, args.L_eval, H, "cpu", make=make_split)
    metrics = rollout_metrics(qhat, y)
    test_sets = [make_split(c) for _, c in test_cfgs]
    drift_test = _drift_mse(field, test_sets)
    drift_train = _drift_mse(field, [train])
    t3 = time.time()

    record = {
        "split": split, "cell": cell, "seed": seed, "arm": arm, "scope": scope, "objective": obj,
        "train_region": train_region or "standard_box", "test_region": test_region,
        "args": args_dict,
        "train_losses": losses,
        "val_curve": tracker.curve, "best_step": tracker.best_step, "best_val": tracker.best,
        "metrics": metrics, "per_test_seed_mse": per_seed,
        "drift_mse_test": drift_test, "drift_mse_train": drift_train,
        "trainable_params": int(sum(p.numel() for p in field.parameters() if p.requires_grad)),
        "timings_s": {"build": t1 - t0, "train": t2 - t1, "eval": t3 - t2},
    }
    np.savez_compressed(out_dir / f"seed{seed}.npz", qhat=qhat.astype(np.float32), y=y.astype(np.float32))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    return split, cell, seed, metrics.get("mse"), t3 - t0


SUMMARY_KEYS = ("mse", "wrong_wells", "transient_mse", "late_mse_correct_wells",
                "osc_energy_ratio", "dom_freq_match", "well_acc_crossers")


def summarize(root: pathlib.Path, split: str):
    rows = {}
    for f in sorted((root / split).glob("*/seed*.json")):
        r = json.loads(f.read_text())
        vals = {k: r["metrics"].get(k) for k in SUMMARY_KEYS}
        vals["drift_mse_test"] = r["drift_mse_test"]
        vals["drift_mse_train"] = r["drift_mse_train"]
        rows.setdefault(r["cell"], []).append(vals)
    table = {}
    for cell in CELLS:
        if cell not in rows:
            continue
        keys = rows[cell][0].keys()
        table[cell] = {"n_seeds": len(rows[cell])}
        for k in keys:
            v = np.array([row[k] for row in rows[cell] if row[k] is not None], dtype=float)
            table[cell][k] = [float(np.nanmean(v)), float(np.nanstd(v))] if v.size else None
    (root / split / "summary.json").write_text(json.dumps(table, indent=1))
    head = f"{'cell':16s} {'n':>2s} {'mse':>15s} {'wrong':>11s} {'trans':>15s} {'osc':>11s} {'drift_te':>15s} {'drift_tr':>15s}"
    print(head)
    for cell, t in table.items():
        def fmt(k, w=15, p=3):
            v = t.get(k)
            return f"{'-':>{w}s}" if v is None else f"{v[0]:.{p}g}±{v[1]:.2g}".rjust(w)
        print(f"{cell:16s} {t['n_seeds']:2d} {fmt('mse')} {fmt('wrong_wells', 11)} {fmt('transient_mse')} "
              f"{fmt('osc_energy_ratio', 11)} {fmt('drift_mse_test')} {fmt('drift_mse_train')}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--cells", default="all", help="comma list of cell names, or 'all'")
    ap.add_argument("--seeds", nargs="+", default=["0-4"], help="model seeds, e.g. 0-4 or 0 2")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--L-train", type=int, default=200)
    ap.add_argument("--test-seeds", nargs="+", default=["1000-1004"])
    ap.add_argument("--n-test", type=int, default=64, help="trajectories per test seed")
    ap.add_argument("--L-eval", type=int, default=600)
    ap.add_argument("--val-seed", type=int, default=1, help="in-distribution validation data seed")
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=2000, help="FIM cells: optimizer-step cap")
    ap.add_argument("--max-steps-cold", type=int, default=5000, help="cold cells C1, C2")
    ap.add_argument("--val-every", type=int, default=25, help="FIM cells: steps between validations")
    ap.add_argument("--val-every-cold", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=2, help="optimizer steps per epoch")
    ap.add_argument("--L-short", type=int, default=6)
    ap.add_argument("--n-times", type=int, default=8, help="drift-matching times per step")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="runs/fim_ood")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to check the pipeline")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_train, args.L_train, args.n_test, args.L_eval, args.n_val = 8, 30, 8, 400, 4
        args.max_steps, args.max_steps_cold, args.val_every, args.val_every_cold = 20, 20, 5, 5
        args.test_seeds = ["1000"]
        args.out = str(pathlib.Path(args.out) / "smoke")

    root = pathlib.Path(args.out)
    if args.summarize_only:
        summarize(root, args.split)
        return
    cells = list(CELLS) if args.cells == "all" else args.cells.split(",")
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; known: {list(CELLS)}")
    seeds = parse_seeds(args.seeds)
    # Z is deterministic (no training, dropout off): one seed is enough.
    jobs = [(c, s) for c in cells for s in (seeds[:1] if c == "Z" else seeds)]
    if args.skip_existing:
        jobs = [(c, s) for c, s in jobs if not (root / args.split / c / f"seed{s}.json").exists()]
    # Longest first, so the pool does not end on a tail of slow jobs (pilot timings).
    cost = {"roll": 3, "drift": 2, "distill": 1, None: 1}
    jobs.sort(key=lambda j: -(cost[CELLS[j[0]][2]] * (2 if CELLS[j[0]][0] in ("M", "S") else 1)))
    print(f"[sweep] split {args.split}: {len(jobs)} jobs, {args.jobs} workers x {args.threads} threads")
    args_dict = vars(args)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=torch.multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(run_job, args.split, c, s, args_dict): (c, s) for c, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            c, s = futs[fut]
            try:
                _, _, _, mse, dt = fut.result()
                print(f"[sweep] {i}/{len(jobs)} {c} seed{s}  mse {mse:.4g}  {dt:.0f}s  "
                      f"(elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # keep the sweep alive; report the failure
                print(f"[sweep] {i}/{len(jobs)} {c} seed{s}  FAILED: {type(e).__name__}: {e}", flush=True)
    summarize(root, args.split)


if __name__ == "__main__":
    main()
