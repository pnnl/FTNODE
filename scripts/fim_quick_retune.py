#!/usr/bin/env python3
"""Quick, CPU-cheap FIM retune proof-of-concept on Duffing (quick cells 3 and 4).

The fast counterpart to the heavy FIM cells in ``scripts/fim_factorial.py``.  It runs on
**CPU by default** and uses few trajectories, few epochs, and short rollouts, so it never
contends with a live heavy run on the GPUs.  It asks a narrow question:

- **Cell 3 (unstructured, reference):** how well does the pretrained FIM already track the
  Duffing field (zero-shot), and how much does a cheap short-``q``-rollout retune improve it?
- **Cell 4 (structured, the PoC):** can the bounded field ``A(x)(x - g)`` *carry* the
  zero-shot FIM drift by distillation -- in the visited region, and out of it where the
  ``-q**3`` term forces the bounded structure to break?

Primary metric: drift MSE against the true field ``duffing_field_torch`` (we own the plant).

Usage::

    uv run --group fim python scripts/fim_quick_retune.py            # defaults, CPU
    uv run --group fim python scripts/fim_quick_retune.py --n-traj 16 --epochs-retune 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from ftnode.fim import FIMBackbone, FreeFIMField, StructuredFIMField
from ftnode.fim.quick import (
    DistillConfig,
    RetuneConfig,
    distill_structured,
    drift_mse_on_grid,
    drift_mse_vs_reference,
    drift_mse_vs_true,
    retune_short_rollout,
    short_horizon_qmse,
    zero_shot_snapshot,
)
from ftnode.systems import DuffingDataConfig, make_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-traj", type=int, default=12, help="trajectories in the quick dataset")
    ap.add_argument("--L", type=int, default=60, help="steps per trajectory (visited region)")
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--u-range", type=float, default=0.25)
    ap.add_argument("--x-range", type=float, default=1.6)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (default; keeps off the live GPU run)")
    ap.add_argument("--epochs-retune", type=int, default=40, help="cell-3 short-rollout retune")
    ap.add_argument("--epochs-distill", type=int, default=60, help="cell-4 distillation")
    ap.add_argument("--epochs-finetune", type=int, default=20, help="cell-4 optional post-distill polish")
    ap.add_argument("--L-short", type=int, default=6, help="short rollout horizon (4-8)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grid-n", type=int, default=15, help="per-axis grid resolution for OOD eval")
    ap.add_argument("--grid-scale", type=float, default=2.5, help="grid extent as a multiple of the visited box")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/fim_quick/report.json"))
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DuffingDataConfig(
        n_traj=args.n_traj, L=args.L, tau=args.tau, h=args.h,
        u_range=args.u_range, x_range=args.x_range, delta=args.delta, seed=args.seed,
    )
    ds = make_dataset(cfg).to(device)
    params = cfg.params
    print(f"[fim_quick] device {device}  n_traj {args.n_traj}  L {args.L}  "
          f"window {tuple(ds.W.shape)}  visited states {args.n_traj * (args.L + 1)}")

    print("[fim_quick] loading zero-shot FIM base_model (cpu)...")
    backbone = FIMBackbone.from_pretrained(device=args.device, state_dim=2)
    backbone.to(device)
    backbone_c4 = zero_shot_snapshot(backbone)  # cell-4 owns the zero-shot field; retune must not touch it

    report: dict = {"config": vars(args) | {"out": str(args.out)}, "timings_s": {}}

    # --- Cell 3: unstructured (reference number) ---------------------------------------
    t0 = time.time()
    report["cell3_zero_shot"] = drift_mse_vs_true(backbone, ds, args.h, params)
    report["timings_s"]["cell3_zero_shot"] = time.time() - t0
    print(f"[cell3] zero-shot drift MSE vs true  {report['cell3_zero_shot']['mean']:.3e} "
          f"+/- {report['cell3_zero_shot']['std']:.3e}")

    free = FreeFIMField(backbone, d=2, hidden=args.hidden, depth=args.depth).to(device)
    rcfg = RetuneConfig(epochs=args.epochs_retune, L_short=args.L_short, lr=args.lr,
                        lr_backbone=args.lr_backbone, batch=args.batch)
    t0 = time.time()
    free, retune_losses = retune_short_rollout(free, ds, args.h, rcfg, verbose=True)
    report["timings_s"]["cell3_retune"] = time.time() - t0
    report["cell3_retuned"] = drift_mse_vs_true(free, ds, args.h, params)
    report["cell3_retuned"]["short_horizon_qmse"] = short_horizon_qmse(free, ds, args.h, args.L_short)
    report["cell3_retuned"]["final_train_qmse"] = retune_losses[-1] if retune_losses else None
    print(f"[cell3] retuned  drift MSE vs true  {report['cell3_retuned']['mean']:.3e}  "
          f"(short-horizon q MSE {report['cell3_retuned']['short_horizon_qmse']:.3e})")

    # --- Cell 4: structured (distillation + optional polish) ---------------------------
    struct = StructuredFIMField(backbone_c4, d=2, q=1, hidden=args.hidden, depth=args.depth).to(device)
    dcfg = DistillConfig(epochs=args.epochs_distill, lr=args.lr, batch=args.batch)
    t0 = time.time()
    struct, distill_losses = distill_structured(struct, backbone_c4, ds, args.h, dcfg, verbose=True)
    report["timings_s"]["cell4_distill"] = time.time() - t0

    def fim_ref(x, u):
        return backbone_c4.drift(x)  # context is bound by the caller's field.prepare(ds.W, h)

    report["cell4_distilled"] = drift_mse_vs_true(struct, ds, args.h, params)
    report["cell4_distilled"]["grid"] = drift_mse_on_grid(
        struct, ds, args.h, params, n=args.grid_n, scale=args.grid_scale)
    report["cell4_distilled"]["residual_to_fim"] = drift_mse_vs_reference(struct, fim_ref, ds, args.h)["mean"]
    report["cell4_distilled"]["final_distill_mse"] = distill_losses[-1] if distill_losses else None
    g = report["cell4_distilled"]["grid"]
    print(f"[cell4] distilled  drift MSE vs true  {report['cell4_distilled']['mean']:.3e}  "
          f"(residual to FIM {report['cell4_distilled']['residual_to_fim']:.3e})")
    print(f"[cell4] grid  in-region {g['in_region_mse']:.3e}  out-region {g['out_region_mse']:.3e}  "
          f"({g['in_points']} in / {g['out_points']} out)")

    if args.epochs_finetune > 0:
        fcfg = RetuneConfig(epochs=args.epochs_finetune, L_short=args.L_short, lr=args.lr,
                            lr_backbone=args.lr_backbone, batch=args.batch)  # backbone frozen -> head-only
        t0 = time.time()
        struct, finetune_losses = retune_short_rollout(struct, ds, args.h, fcfg, verbose=True)
        report["timings_s"]["cell4_finetune"] = time.time() - t0
        report["cell4_finetuned"] = drift_mse_vs_true(struct, ds, args.h, params)
        report["cell4_finetuned"]["grid"] = drift_mse_on_grid(
            struct, ds, args.h, params, n=args.grid_n, scale=args.grid_scale)
        report["cell4_finetuned"]["short_horizon_qmse"] = short_horizon_qmse(struct, ds, args.h, args.L_short)
        print(f"[cell4] finetuned  drift MSE vs true  {report['cell4_finetuned']['mean']:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\n[fim_quick] wrote {args.out}")
    print("[fim_quick] summary (drift MSE vs true field):")
    print(f"  cell3 zero-shot : {report['cell3_zero_shot']['mean']:.3e}")
    print(f"  cell3 retuned   : {report['cell3_retuned']['mean']:.3e}")
    print(f"  cell4 distilled : {report['cell4_distilled']['mean']:.3e}  "
          f"(out-region {report['cell4_distilled']['grid']['out_region_mse']:.3e})")
    if args.epochs_finetune > 0:
        print(f"  cell4 finetuned : {report['cell4_finetuned']['mean']:.3e}")


if __name__ == "__main__":
    main()
