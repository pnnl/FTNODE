#!/usr/bin/env python3
"""Equilibria of the fields saved by ``fim_ood_sweep.py --save-ckpt``, per test context.

A FIM field depends on its context (the trajectory's own window), so each test trajectory
has its own field.  For every test trajectory this script:

1. binds the trajectory's context and rolls the restored field out for ``L_eval`` steps;
2. runs Newton's method from the final predicted state, to find the equilibrium the
   prediction settles at;
3. runs Newton's method from the true saddle ``(q_s(u), 0)``, to find the model's
   equilibrium nearest the saddle;
4. classifies each equilibrium found by the eigenvalues of the field Jacobian ``D_x F``
   (finite differences, one batch-aligned query per trajectory): sink (no eigenvalue with a
   positive real part), saddle (one), or source (two).

The true plant has a saddle at ``(q_s(u), 0)``.  A model equilibrium there that is a sink is
a spurious attractor.  One JSON per checkpoint, next to it, plus a summary table.

Usage::

    python scripts/fim_ood_equilibria.py --root runs/fim_ood_diag/sink --split crossing_half
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

SADDLE_TOL = 0.3   # an equilibrium within this of q_s(u) counts as "at the saddle"
RES_TOL = 1e-3     # |F(x*)| below this counts as converged


def fd_jacobian(F, x, u, eps=1e-3):
    """``J[b, i, j] = dF_i/dx_j`` by central differences, one query per trajectory."""
    cols = []
    for j in range(x.shape[1]):
        e = torch.zeros_like(x)
        e[:, j] = eps
        cols.append((F(x + e, u) - F(x - e, u)) / (2 * eps))
    return torch.stack(cols, -1)


@torch.no_grad()
def newton(F, x, u, iters=40, max_step=0.5):
    """Damped Newton iterations for ``F(x, u) = 0``, batched; returns the root and |F|."""
    for _ in range(iters):
        f = F(x, u)
        J = fd_jacobian(F, x, u)
        step = torch.linalg.lstsq(J, f.unsqueeze(-1)).solution.squeeze(-1)
        x = x - torch.nan_to_num(step).clamp(-max_step, max_step)
    return x, F(x, u).norm(dim=-1)


def eig_class(J):
    """0 sink, 1 saddle, 2 source, from the number of eigenvalues with a positive real part."""
    return (torch.linalg.eigvals(J).real > 0).sum(-1)


@torch.no_grad()
def analyze(json_path: str, threads: int):
    torch.set_num_threads(threads)
    from ftnode.fim import quick
    from ftnode.fim.backbone import FIMBackbone
    from ftnode.ood_splits import make_split, saddle
    from ftnode.physical.model import PhysicalField
    from ftnode.train import rk4_step

    jp = pathlib.Path(json_path)
    r = json.loads(jp.read_text())
    args = argparse.Namespace(**r["args"])
    field = sweep._build(r["arm"], r["scope"], r["seed"],
                         lambda: FIMBackbone.from_pretrained(device="cpu", state_dim=2),
                         R_g=getattr(args, "R_g", 2.0))
    field.load_state_dict(torch.load(jp.with_suffix(".pth"), map_location="cpu"))
    field.eval()
    if hasattr(field, "backbone"):
        quick.disable_backbone_dropout(field)
    model = PhysicalField(field, sweep.H)

    tests = [make_split(c) for _, c in sweep._test_cfgs(args, r["test_region"])]
    W = torch.cat([d.W for d in tests]); U = torch.cat([d.U for d in tests])
    model.prepare(W)
    x = model.estimate_x0(W)
    late_q = []
    for k in range(args.L_eval):
        x = rk4_step(model.F, x, U, sweep.H)
        if k >= args.L_eval - 200:
            late_q.append(x[:, 0])
    late_std = torch.stack(late_q, 1).std(1)

    F = model.F
    qs = torch.tensor([saddle(float(u)) for u in U], dtype=x.dtype)
    x_end, res_end = newton(F, x, U)
    x_sad, res_sad = newton(F, torch.stack([qs, torch.zeros_like(qs)], -1), U)
    cls_end = eig_class(fd_jacobian(F, x_end, U))
    J_sad = fd_jacobian(F, x_sad, U)
    cls_sad = eig_class(J_sad)
    max_re_sad = torch.linalg.eigvals(J_sad).real.max(-1).values

    finite = torch.isfinite(x).all(-1)
    conv_end = (res_end < RES_TOL) & finite
    at_saddle_end = conv_end & ((x_end[:, 0] - qs).abs() < SADDLE_TOL)
    conv_sad = (res_sad < RES_TOL) & ((x_sad[:, 0] - qs).abs() < SADDLE_TOL)
    names = ["sink", "saddle", "source"]
    out = {
        "split": r["split"], "cell": r["cell"], "seed": r["seed"], "R_g": getattr(args, "R_g", 2.0),
        "n": int(len(U)),
        # Where predictions settle: the equilibrium reached from the final state.
        "settle_converged": int(conv_end.sum()),
        "settle_at_saddle": int(at_saddle_end.sum()),
        "settle_at_saddle_class": {n: int((at_saddle_end & (cls_end == k)).sum()) for k, n in enumerate(names)},
        "settle_q_median": float(x_end[conv_end, 0].median()) if conv_end.any() else float("nan"),
        # The model's equilibrium nearest the true saddle.
        "near_saddle_found": int(conv_sad.sum()),
        "near_saddle_class": {n: int((conv_sad & (cls_sad == k)).sum()) for k, n in enumerate(names)},
        # Largest real part of the eigenvalues there (the true saddle has +0.905 at u = 0).
        "near_saddle_max_re_pct": (np.percentile(max_re_sad[conv_sad].numpy(), [10, 50, 90]).tolist()
                                   if conv_sad.any() else None),
        # Late motion of the predictions that end at an equilibrium near the saddle.
        "settle_at_saddle_late_std_median": {
            n: (float(late_std[at_saddle_end & (cls_end == k)].median())
                if (at_saddle_end & (cls_end == k)).any() else None) for k, n in enumerate(names)},
    }
    (jp.parent / f"seed{r['seed']}.equilibria.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the sweep --out directory")
    ap.add_argument("--split", required=True, choices=sorted(sweep.SPLITS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    jobs = sorted(str(j.with_suffix(".json")) for j in (pathlib.Path(args.root) / args.split).glob("*/seed*.pth"))
    print(f"[equilibria] {args.split}: {len(jobs)} checkpoints")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=torch.multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(analyze, j, args.threads): j for j in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[equilibria] {futs[fut]} FAILED: {type(e).__name__}: {e}", flush=True)
    rows.sort(key=lambda o: (o["cell"], o["seed"]))
    print(f"{'cell':16s} {'seed':>4s} {'settle@saddle':>13s} {'(sink/saddle/source)':>21s} "
          f"{'near-saddle eq':>15s} {'(sink/saddle/source)':>21s}  ({time.time() - t0:.0f}s)")
    for o in rows:
        a, b = o["settle_at_saddle_class"], o["near_saddle_class"]
        print(f"{o['cell']:16s} {o['seed']:4d} {o['settle_at_saddle']:13d} "
              f"{a['sink']:>7d}/{a['saddle']:>3d}/{a['source']:>3d}      "
              f"{o['near_saddle_found']:15d} {b['sink']:>7d}/{b['saddle']:>3d}/{b['source']:>3d}")
    (pathlib.Path(args.root) / args.split / "equilibria_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
