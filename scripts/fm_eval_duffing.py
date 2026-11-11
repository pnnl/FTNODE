#!/usr/bin/env python3
"""Zero-shot evaluation of a pretrained context-conditioned field on the Duffing OOD splits.

No gradient step touches the model.  For each split of the out-of-distribution study:

1. the context is the split's training data (64 trajectories, data seed 0, measured ``q``
   only), cut to its first ``K`` trajectories for ``K`` in ``--context-sizes``;
2. the encoder turns the context into one code ``c``, which stays fixed;
3. the field ``F(x, u; c)`` is rolled out from the finite-difference ``x0`` of every test
   trajectory (test seeds 1000-1004, 64 trajectories each, 600 steps), exactly as the
   OOD sweep scores its cells, and scored with the same rollout metrics;
4. each prediction is sorted into a settling bin (true well, other well, rest at ``q = 0``,
   box edge ``|q| = 2``, still oscillating, other bounded, diverged);
5. at the largest context, the equilibria near ``q = -1, q_s(u), +1`` are found by Newton's
   method for ``u`` in ``{-0.2, 0, 0.2}`` and classified by the eigenvalues of ``D_x F``;
6. the drift error against the true Duffing field is measured on the test states.

The training split of ``far_field`` is the standard box (``make_dataset``).  Writes
``<out>/eval_<ckpt>.json`` and one ``.npz`` of predictions per split and context size.

Usage::

    python scripts/fm_eval_duffing.py --run runs/fm_v0lite/structured_seed0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from ftnode.foundation import build_context_field, tokens_from_q
from ftnode.foundation.generator import batch_generator
from ftnode.ood_splits import SplitConfig, make_split, saddle
from ftnode.physical import PhysicalField
from ftnode.physical.metrics import rollout_metrics, rollout_test_split
from ftnode.systems import DuffingDataConfig, DuffingParams, duffing_field_torch, make_dataset

SPLITS = {
    "trapped_lobe": ("trapped_right", "trapped_left"),
    "crossing_half": ("half_right", "half_left"),
    "far_field": (None, "far_ring"),
}
H = 0.05


def load(run, ckpt, device):
    blob = torch.load(pathlib.Path(run) / ckpt, map_location=device, weights_only=False)
    a = blob["config"]["args"]
    field = build_context_field(a["head"], code_dim=a["code_dim"], R_g=a["R_g"]).to(device)
    field.load_state_dict(blob["model"])
    field.eval()
    return field, blob


def train_data(region, n, seed=0, L=200):
    if region is None:
        return make_dataset(DuffingDataConfig(n_traj=n, L=L, seed=seed))
    return make_split(SplitConfig(region=region, n_traj=n, L=L, seed=seed))


def settle_bins(qhat, y, late=400):
    """Counts per settling bin; the true and the other well take priority over oscillation."""
    P, Y = np.asarray(qhat, float), np.asarray(y, float)
    finite = np.isfinite(P).all(1)
    Pl = np.where(np.isfinite(P), P, 0.0)[:, late:]
    m, s, tm = Pl.mean(1), Pl.std(1), Y[:, late:].mean(1)
    bins = {k: 0 for k in ("true_well", "other_well", "rest_q0", "edge_q2", "oscillating", "other_bounded", "diverged")}
    for i in range(len(Y)):
        if not finite[i] or np.abs(Pl[i]).max() > 5.0:
            bins["diverged"] += 1
        elif abs(m[i] - tm[i]) < 0.3:
            bins["true_well"] += 1
        elif abs(m[i] + tm[i]) < 0.3 and abs(tm[i]) > 0.5:
            bins["other_well"] += 1
        elif s[i] > 0.2:
            bins["oscillating"] += 1
        elif abs(m[i]) < 0.3:
            bins["rest_q0"] += 1
        elif abs(abs(m[i]) - 2.0) < 0.2:
            bins["edge_q2"] += 1
        else:
            bins["other_bounded"] += 1
    return bins


def classify_equilibria(field, device):
    """Newton from ``(-1, 0)``, ``(q_s(u), 0)``, ``(1, 0)`` at fixed ``u``; eigen-classify each."""
    out = []
    for u in (-0.2, 0.0, 0.2):
        uu = torch.tensor([u], device=device)
        for name, q0 in (("left", -1.0), ("saddle", saddle(u)), ("right", 1.0)):
            x = torch.tensor([[q0, 0.0]], device=device)
            f = lambda z: field.F(z.unsqueeze(0), uu).squeeze(0)
            for _ in range(60):
                J = torch.func.jacrev(f)(x[0])
                step = torch.linalg.lstsq(J, -f(x[0]).unsqueeze(-1)).solution.squeeze(-1)
                x = x + 0.5 * step.clamp(-0.5, 0.5)
            res = float(f(x[0]).norm())
            ev = torch.linalg.eigvals(torch.func.jacrev(f)(x[0]))
            n_unst = int((ev.real > 1e-6).sum())
            kind = {0: "sink", 1: "saddle", 2: "source"}[n_unst]
            out.append({"u": u, "start": name, "x": x[0].tolist(), "residual": res,
                        "kind": kind if res < 1e-4 else "not_converged",
                        "max_re": float(ev.real.max()), "eig": [[float(e.real), float(e.imag)] for e in ev]})
    return out


@torch.no_grad()
def drift_mse_on_states(field, ds, device):
    X = ds.Xfull.reshape(-1, 2).to(device)
    U = ds.U.unsqueeze(1).expand(-1, ds.Xfull.shape[1]).reshape(-1).to(device)
    pred = field.F(X, U)
    return float(((pred - duffing_field_torch(X, U, DuffingParams())) ** 2).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--context-sizes", type=int, nargs="+", default=[4, 8, 16, 64])
    ap.add_argument("--n-tok", type=int, default=1024, help="context tokens drawn (0 = all)")
    ap.add_argument("--token-seed", type=int, default=7)
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--L-eval", type=int, default=600)
    args = ap.parse_args()
    device = torch.device(args.device)
    field, blob = load(args.run, args.ckpt, device)
    model = PhysicalField(field, H)
    out_dir = pathlib.Path(args.run)
    result = {"run": args.run, "ckpt": args.ckpt, "ckpt_step": blob["step"], "splits": {}}

    for split in args.splits:
        train_region, test_region = SPLITS[split]
        train = train_data(train_region, max(args.context_sizes))
        q_all = torch.cat([train.W, train.Y], 1).to(device)
        u_all = train.U.to(device)
        test_cfgs = [(ts, SplitConfig(region=test_region, n_traj=args.n_test, L=args.L_eval, seed=ts))
                     for ts in args.test_seeds]
        res_split = {}
        for K in sorted(args.context_sizes):
            gen = batch_generator(args.token_seed, 2, K)
            tokens = tokens_from_q(q_all[:K], u_all[:K], H, n_tok=args.n_tok or None, generator=gen)
            field.bind_context(tokens)
            qhat, y, per_seed = rollout_test_split(model, test_cfgs, args.L_eval, H, device, make=make_split)
            met = rollout_metrics(qhat, y)
            met["bins"] = settle_bins(qhat, y)
            test_ds = make_split(test_cfgs[0][1])
            met["drift_mse_test_states"] = drift_mse_on_states(field, test_ds, device)
            met["drift_mse_train_states"] = drift_mse_on_states(field, train, device)
            if K == max(args.context_sizes):
                met["equilibria"] = classify_equilibria(field, device)
            res_split[f"K{K}"] = met
            np.savez_compressed(out_dir / f"pred_{split}_K{K}.npz", qhat=qhat, y=y)
            b = met["bins"]
            print(f"{split:14s} K={K:3d}  true {b['true_well']:3d}  other {b['other_well']:3d}  "
                  f"q0 {b['rest_q0']:3d}  edge {b['edge_q2']:3d}  osc {b['oscillating']:3d}  "
                  f"div {b['diverged']:3d}  drift(test) {met['drift_mse_test_states']:.3f}", flush=True)
        result["splits"][split] = res_split
    path = out_dir / f"eval_{pathlib.Path(args.ckpt).stem}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
