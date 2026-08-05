"""Rollout and training loop for the latent FT-NODE models.

Promoted from the duffing kappa notebooks.  The only structural change is that
``train_one`` no longer closes over module-level dataset globals
(``Wtr_d``/``Utr_d``/``Ytr_d``/``Wva_d``/... , ``L``, ``L_eval``, ``h_dt``,
``device``); they arrive as a :class:`ftnode.systems.DuffingDataset` pair plus a
:class:`TrainConfig`.
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from .systems import DuffingDataset

__all__ = ["rk4_step", "rollout_y", "TrainConfig", "train_one", "restore_best"]


def rk4_step(F_fn, z, u, h):
    """One classical RK4 step of ``z' = F_fn(z, u)`` with ``u`` held over the step."""
    k1 = F_fn(z, u)
    k2 = F_fn(z + 0.5 * h * k1, u)
    k3 = F_fn(z + 0.5 * h * k2, u)
    k4 = F_fn(z + h * k3, u)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_y(model, window, u, L, h):
    """Encode a measurement window, integrate ``L`` steps, decode at every node.

    Returns ``(ys, zs)`` of shapes ``(b, L+1)`` and ``(b, L+1, m)``.
    """
    z = model.encode(window)
    ys = [model.decode(z)]
    zs = [z]
    for _ in range(L):
        z = rk4_step(model.F, z, u, h)
        zs.append(z)
        ys.append(model.decode(z))
    return torch.stack(ys, 1), torch.stack(zs, 1)


@dataclass(frozen=True)
class TrainConfig:
    """Identification-stage training settings.

    Defaults are the duffing notebooks' values.  ``lam_res`` weights the residual
    regularizer ``E||z - g(z, u)||^2``, which pulls the trajectory toward the
    equilibrium manifold; it is ``1e-2`` for the FT variants and ``0.0`` for the
    unstructured :class:`~ftnode.latent.LatentNODE` baseline (where it is inert
    anyway -- that model has no ``g``).

    ``L`` is the training rollout length and ``L_eval`` the longer validation
    horizon, so validation measures extrapolation beyond what was trained on.
    """

    n_epochs: int = 200
    lr: float = 3e-3
    batch: int = 64
    clip: float = 1.0
    lam_res: float = 0.0
    L: int = 200
    L_eval: int = 600
    h: float = 0.05


def train_one(
    model,
    train: DuffingDataset,
    val: DuffingDataset,
    cfg: TrainConfig,
    *,
    ckpt_path=None,
    label: str = "model",
    device=None,
    verbose: bool = True,
):
    """Train one model, checkpointing on best extrapolation loss.

    Loss is measured-output MSE over the rollout, plus ``cfg.lam_res`` times the
    residual penalty when the dynamics expose a ``g`` (duck-typed, so the
    unstructured baseline silently skips it).  Training aborts on the first
    non-finite loss and records the epoch in ``hist['diverged_at']`` rather than
    poisoning the optimizer state.

    Returns ``(model, hist)``.  ``hist`` carries per-epoch ``train``,
    ``val_extrap``, ``zmax``, ``res`` series plus ``diverged_at``, ``best_val``,
    ``best_epoch`` and ``ckpt_path``; feed it to :func:`restore_best`.
    """
    device = device or next(model.parameters()).device
    model = model.to(device)
    train = train.to(device)
    val = val.to(device)
    # Keep the notebooks' fallback.  examples/duffing/_proto_id.py dropped this line
    # while leaving ckpt_path=None in the signature, so the first improving epoch
    # reached torch.save(state_dict, None).
    if ckpt_path is None:
        ckpt_path = f"best-{label}.pth"

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    hist = {
        "train": [],
        "val_extrap": [],
        "zmax": [],
        "res": [],
        "diverged_at": None,
        "best_val": float("inf"),
        "best_epoch": None,
        "ckpt_path": str(ckpt_path),
    }
    diverged = False
    t0 = time.time()

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(train.W.shape[0], device=device)
        ep_losses, ep_res, ep_zmax = [], [], 0.0

        for i in range(0, len(perm), cfg.batch):
            idx = perm[i : i + cfg.batch]
            w, u, y = train.W[idx], train.U[idx], train.Y[idx]
            yhat, zs = rollout_y(model, w, u, cfg.L, cfg.h)
            loss = ((yhat - y) ** 2).mean()

            res_pen = zs.new_zeros(())
            if cfg.lam_res > 0.0 and hasattr(model.dynamics, "g"):
                u_steps = u.unsqueeze(1).expand(zs.shape[0], zs.shape[1])
                res_pen = ((zs - model.dynamics.g(zs, u_steps)) ** 2).sum(-1).mean()
                loss = loss + cfg.lam_res * res_pen

            if not torch.isfinite(loss):
                diverged = True
                hist["diverged_at"] = epoch
                break

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()

            ep_losses.append(loss.item())
            ep_res.append(res_pen.item())
            with torch.no_grad():
                ep_zmax = max(ep_zmax, zs.norm(dim=-1).max().item())

        if diverged:
            if verbose:
                print(f"[{label}] diverged at epoch {epoch}")
            break

        sched.step()
        model.eval()
        with torch.no_grad():
            yhat_v, _ = rollout_y(model, val.W, val.U, cfg.L_eval, cfg.h)
            val_mse = ((yhat_v - val.Y) ** 2).mean().item()

        hist["train"].append(float(np.mean(ep_losses)))
        hist["val_extrap"].append(val_mse)
        hist["zmax"].append(ep_zmax)
        hist["res"].append(float(np.mean(ep_res)) if ep_res else 0.0)

        if np.isfinite(val_mse) and val_mse < hist["best_val"]:
            hist["best_val"] = val_mse
            hist["best_epoch"] = epoch
            torch.save(model.state_dict(), ckpt_path)

        if verbose and (epoch % 20 == 0 or epoch == cfg.n_epochs - 1):
            print(
                f'[{label}] ep {epoch:3d}  train {hist["train"][-1]:.3e}  val {val_mse:.3e}  '
                f'res {hist["res"][-1]:.3e}  zmax {ep_zmax:.2f}  ({time.time() - t0:.0f}s)'
            )

    if verbose and hist["best_epoch"] is not None:
        print(f'[{label}] best val {hist["best_val"]:.3e} @ ep {hist["best_epoch"]}')
    return model, hist


def restore_best(model, hist, device=None, label: str = "", verbose: bool = True):
    """Reload the best-validation weights recorded by :func:`train_one`, then ``eval()``."""
    device = device or next(model.parameters()).device
    cp = hist.get("ckpt_path")
    if hist.get("best_epoch") is not None and cp and pathlib.Path(cp).exists():
        model.load_state_dict(torch.load(cp, map_location=device))
        if verbose:
            tag = f"[{label}] " if label else ""
            print(f'{tag}restored best @ ep {hist["best_epoch"]} from {cp}')
    model.eval()
    return model
