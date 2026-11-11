#!/usr/bin/env python3
"""Pretrain the context-conditioned field on the damped mechanical family (Duffing held out).

Drift matching with exact labels: for each system of a batch, the encoder reads the context
tokens (built from measured ``q`` only), returns one code ``c``, and the head is fit to the
true field at the query states,

    loss = mean_systems  mean_queries ||F(x, u; c) - f*(x, u)||^2 / (s_b + 0.1),

where ``s_b`` is the mean squared label at the system's own trajectory queries, so stiff
and soft systems weigh alike.

**Schedule and stop.**  Linear warmup, then cosine decay to 5% of the peak over exactly
``--steps`` optimizer steps.  The run stops when the budget is reached (``stop_reason =
"budget"``).  Two guards can stop it earlier: a non-finite validation loss (``"diverged"``),
and the wall-clock ``--deadline`` (``"deadline"``, the run is then truncated mid-schedule).

**Validation.**  A fixed set, the same for every run: batches ``0 .. n_val_batches-1`` of
``VAL_STREAM`` under ``--val-seed``.  The
budget-decision metric is ``val_traj_rel``: the per-system normalized error at the
trajectory queries only, ``mean_q ||F - f*||^2 / (s_b + 0.1)``, averaged over the validation
systems, with its standard error over systems (``val_traj_rel_se``).  The box queries are
reported separately (``val_box_rel``): a structured head cannot fit cubic forces there.  A
Duffing probe (the held-out plant) is logged for monitoring only and never decides anything.

**Reproducibility.**  Batch ``i`` is ``sample_batch(cfg, B, seed, TRAIN_STREAM, i)``: a pure
function of the seed and the step.  The model init is seeded with the same seed.  GPU
arithmetic is not bitwise deterministic; the data stream is.

Writes into ``--out``: ``config.json``; ``log.jsonl`` (one line per validation);
``summary.json`` at the end (stop reason, steps, skipped steps, throughput, the step-1 and
final validation metrics); checkpoints ``last.pt``, ``best.pt`` (lowest ``val_traj_rel``),
and ``step<N>.pt`` every ``--ckpt-every`` steps.

Usage::

    python scripts/fm_pretrain.py --head structured --device cuda:0 --steps 5000 --out runs/fm_v0lite/pilot/structured_b5000_s0
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import pathlib
import subprocess
import time

import torch

from ftnode.foundation import (
    TRAIN_STREAM,
    VAL_STREAM,
    MechanicalFamilyConfig,
    build_context_field,
    sample_batch,
    tokens_from_q,
)
from ftnode.foundation.generator import batch_generator
from ftnode.systems import DuffingDataConfig, DuffingParams, duffing_field_torch, make_dataset


def drift_terms(field, batch):
    """Per-query squared error and the per-system normalization of one batch."""
    c = field.encode(batch.tokens).unsqueeze(1)                       # (B, 1, k)
    pred = field(batch.xq, batch.uq, c)                               # (B, nq, 2)
    se = ((pred - batch.fq) ** 2).sum(-1)                             # (B, nq)
    scale = (batch.fq[:, batch.on_traj] ** 2).sum(-1).mean(1, keepdim=True) + 0.1
    return se, scale


def drift_loss(field, batch):
    se, scale = drift_terms(field, batch)
    return (se / scale).mean(1).mean()


@torch.no_grad()
def validate(field, val_batches):
    """Validation metrics; the ``*_rel`` ones are per-system normalized, then averaged."""
    field.eval()
    tot, traj, box, mse_on, mse_box = [], [], [], [], []
    for b in val_batches:
        se, scale = drift_terms(field, b)
        rel = se / scale
        tot.append(rel.mean(1))
        traj.append(rel[:, b.on_traj].mean(1))
        box.append(rel[:, ~b.on_traj].mean(1))
        mse_on.append(se[:, b.on_traj].mean(1))
        mse_box.append(se[:, ~b.on_traj].mean(1))
    field.train()
    tot, traj, box = torch.cat(tot), torch.cat(traj), torch.cat(box)
    n = traj.numel()
    return {
        "val_loss": tot.mean().item(),
        "val_traj_rel": traj.mean().item(),
        "val_traj_rel_se": (traj.std() / math.sqrt(n)).item(),
        "val_box_rel": box.mean().item(),
        "val_mse_traj": torch.cat(mse_on).mean().item(),
        "val_mse_box": torch.cat(mse_box).mean().item(),
        "n_val_systems": n,
    }


def duffing_probe(n_traj, device, h):
    """The held-out Duffing plant: a context from the standard box, and query states."""
    ds = make_dataset(DuffingDataConfig(n_traj=n_traj, seed=4242))
    q = torch.cat([ds.W, ds.Y], 1).to(device)
    u = ds.U.to(device)
    gen = batch_generator(4242, 9, 0)
    tokens = tokens_from_q(q, u, h, n_tok=1024, generator=gen)
    x = ds.Xfull.reshape(-1, 2).to(device)
    uu = ds.U.unsqueeze(1).expand(-1, ds.Xfull.shape[1]).reshape(-1).to(device)
    f = duffing_field_torch(x, uu, DuffingParams())
    return tokens, x, uu, f


@torch.no_grad()
def probe_mse(field, probe):
    tokens, x, uu, f = probe
    field.eval()
    c = field.encode(tokens)                                          # (1, k)
    pred = field(x, uu, c)
    field.train()
    return ((pred - f) ** 2).sum(-1).mean().item()


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def code_digest():
    """SHA-256 of the pretraining code, so a run records its code even from a dirty tree."""
    root = pathlib.Path(__file__).resolve().parent.parent
    files = sorted((root / "ftnode" / "foundation").glob("*.py")) + [pathlib.Path(__file__).resolve()]
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", choices=["structured", "free"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="model init and training data stream")
    ap.add_argument("--val-seed", type=int, default=0, help="the fixed validation set; the same for every run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True, help="the step budget; the cosine schedule spans it")
    ap.add_argument("--deadline", default=None,
                    help="local wall-clock hard stop, ISO format (e.g. 2026-10-15T12:00); the run is then truncated")
    ap.add_argument("--batch", type=int, default=64, help="systems per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--n-val-batches", type=int, default=8)
    ap.add_argument("--code-dim", type=int, default=64)
    ap.add_argument("--R-g", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    deadline = dt.datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    fam = MechanicalFamilyConfig()

    torch.manual_seed(args.seed)
    field = build_context_field(args.head, code_dim=args.code_dim, R_g=args.R_g).to(device)
    n_enc = sum(p.numel() for p in field.encoder.parameters())
    n_head = sum(p.numel() for p in field.head.parameters())
    config = {
        "args": vars(args), "family": dataclasses.asdict(fam),
        "n_params_encoder": n_enc, "n_params_head": n_head, "git": git_rev(), "code_sha256": code_digest(),
        "torch": torch.__version__, "data_stream": "sample_batch(family, batch, seed, TRAIN_STREAM, step)",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[{args.head} seed {args.seed} budget {args.steps}] encoder {n_enc:,}, head {n_head:,} params, "
          f"device {device}", flush=True)

    opt = torch.optim.AdamW(field.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))

    val_batches = [sample_batch(fam, args.batch, args.val_seed, VAL_STREAM, i, device=device)
                   for i in range(args.n_val_batches)]
    probe = duffing_probe(64, device, fam.h)
    log = open(out / "log.jsonl", "a")
    best = float("inf")
    first = None
    t0 = time.time()
    step, skipped, stop_reason = 0, 0, "budget"
    run_loss, run_n = 0.0, 0

    def checkpoint(name, rec=None):
        torch.save({"step": step, "model": field.state_dict(), "config": config, "val": rec}, out / name)

    def do_validation():
        nonlocal best, first, run_loss, run_n
        rec = {"step": step, "time_s": round(time.time() - t0, 1), "lr": lr_at(max(step - 1, 0)),
               "train_loss": run_loss / max(run_n, 1), **validate(field, val_batches),
               "duffing_probe_mse": probe_mse(field, probe), "skipped_steps": skipped}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        run_loss, run_n = 0.0, 0
        if first is None:
            first = rec
        if math.isfinite(rec["val_traj_rel"]) and rec["val_traj_rel"] < best:
            best = rec["val_traj_rel"]
            checkpoint("best.pt", rec)
        return rec

    rec = do_validation()                                             # step 0: the untrained model
    while step < args.steps:
        if deadline is not None and time.time() >= deadline:
            stop_reason = "deadline"
            break
        batch = sample_batch(fam, args.batch, args.seed, TRAIN_STREAM, step, device=device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = drift_loss(field, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(field.parameters(), args.clip)
        if torch.isfinite(gn) and torch.isfinite(loss):
            opt.step()
        else:
            skipped += 1                                              # dropped, never clipped into Adam
        run_loss += loss.item() if torch.isfinite(loss) else 0.0
        run_n += 1
        step += 1
        if step % args.val_every == 0 or step == args.steps:
            rec = do_validation()
            if not math.isfinite(rec["val_loss"]):
                stop_reason = "diverged"
                break
        if step % args.ckpt_every == 0:
            checkpoint(f"step{step}.pt", rec)
    if rec["step"] != step:
        rec = do_validation()
    checkpoint("last.pt", rec)
    wall = time.time() - t0
    summary = {
        "head": args.head, "seed": args.seed, "budget": args.steps, "steps": step,
        "stop_reason": stop_reason, "skipped_steps": skipped, "skipped_frac": skipped / max(step, 1),
        "wall_s": round(wall, 1), "steps_per_s": step / max(wall, 1e-9),
        "first": first, "final": rec, "best_val_traj_rel": best,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {stop_reason}, {step}/{args.steps} steps in {wall:.0f} s, "
          f"final val_traj_rel {rec['val_traj_rel']:.5f}", flush=True)


if __name__ == "__main__":
    main()
