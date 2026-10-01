"""Rollout and training for the physical-state cells.

Separate from ``ftnode.train`` on purpose: that loop encodes a window into a latent
``z`` and carries a single learning rate, while the physical cells start from a
finite-difference ``x0`` and the FIM cells need a low backbone learning rate.  This
module reuses ``rk4_step`` and ``restore_best`` from ``ftnode.train`` unchanged, and
reuses its divergence-guard / best-validation-checkpoint pattern.

The measured-output loss is ``q = x[0]`` MSE over the rollout.  There is no residual
regularizer: the factorial sets ``lam_res = 0`` for every cell, so a residual term
(which fires only for the structured cells) cannot confound the structure effect.
"""
from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..systems import DuffingDataset
from ..train import rk4_step

__all__ = ["rollout_q", "PhysTrainConfig", "train_physical"]


def rollout_q(model, window, u, L, h):
    """Estimate ``x0`` from the window, integrate ``L`` steps, read ``q`` at each node.

    Returns ``qs`` of shape ``(b, L+1)``.  ``model`` exposes ``estimate_x0``, ``F``,
    and ``readout`` (see :class:`ftnode.physical.PhysicalField`).
    """
    x = model.estimate_x0(window)
    qs = [model.readout(x)]
    for _ in range(L):
        x = rk4_step(model.F, x, u, h)
        qs.append(model.readout(x))
    return torch.stack(qs, 1)


@dataclass(frozen=True)
class PhysTrainConfig:
    """Physical-cell training settings.

    ``lr`` applies to every parameter by default.  ``lr_backbone`` overrides the rate
    for a FIM backbone when the model exposes named parameter groups; it is ``None``
    for the cold cells, where there is no backbone.  ``L`` is the training rollout
    length and ``L_eval`` the longer validation horizon.
    """

    n_epochs: int = 600
    lr: float = 3e-3
    lr_backbone: float | None = None
    batch: int = 64
    clip: float = 1.0
    L: int = 200
    L_eval: int = 600
    h: float = 0.05
    patience: int | None = None  # early-stop after this many epochs with no val improvement; None disables


def _param_groups(model, cfg: PhysTrainConfig):
    """Parameter groups for the optimizer.

    If the model defines ``param_groups(lr, lr_backbone)`` -- the FIM container does,
    to give its backbone a low rate -- use it.  Otherwise one group at ``cfg.lr``.
    """
    if hasattr(model, "param_groups"):
        return model.param_groups(cfg.lr, cfg.lr_backbone)
    return [{"params": list(model.parameters()), "lr": cfg.lr}]


def train_physical(
    model,
    train: DuffingDataset,
    val: DuffingDataset,
    cfg: PhysTrainConfig,
    *,
    ckpt_path=None,
    label: str = "model",
    device=None,
    verbose: bool = True,
):
    """Train one physical-state cell, checkpointing on best validation loss.

    Returns ``(model, hist)``.  ``hist`` carries per-epoch ``train`` and
    ``val_extrap`` series plus ``diverged_at``, ``best_val``, ``best_epoch`` and
    ``ckpt_path``.  Feed it to :func:`ftnode.train.restore_best`.
    """
    device = device or next(model.parameters()).device
    model = model.to(device)
    train = train.to(device)
    val = val.to(device)
    if ckpt_path is None:
        ckpt_path = f"best-{label}.pth"

    opt = torch.optim.Adam(_param_groups(model, cfg))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    hist = {
        "train": [],
        "val_extrap": [],
        "diverged_at": None,
        "best_val": float("inf"),
        "best_epoch": None,
        "ckpt_path": str(ckpt_path),
    }
    diverged = False
    since_best = 0
    t0 = time.time()

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(train.W.shape[0], device=device)
        ep_losses = []

        for i in range(0, len(perm), cfg.batch):
            idx = perm[i : i + cfg.batch]
            w, u, y = train.W[idx], train.U[idx], train.Y[idx]
            if hasattr(model, "prepare"):
                model.prepare(w)  # bind FIM context to this batch's windows; no-op for cold cells
            qhat = rollout_q(model, w, u, cfg.L, cfg.h)
            loss = ((qhat - y) ** 2).mean()

            if not torch.isfinite(loss):
                diverged = True
                hist["diverged_at"] = epoch
                break

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            ep_losses.append(loss.item())

        if diverged:
            if verbose:
                print(f"[{label}] diverged at epoch {epoch}")
            break

        sched.step()
        model.eval()
        with torch.no_grad():
            if hasattr(model, "prepare"):
                model.prepare(val.W)
            qhat_v = rollout_q(model, val.W, val.U, cfg.L_eval, cfg.h)
            val_mse = ((qhat_v - val.Y) ** 2).mean().item()

        hist["train"].append(float(np.mean(ep_losses)))
        hist["val_extrap"].append(val_mse)

        if np.isfinite(val_mse) and val_mse < hist["best_val"]:
            hist["best_val"] = val_mse
            hist["best_epoch"] = epoch
            # Write then rename, so a reader or an interrupt never sees a partial file.
            tmp = f"{ckpt_path}.tmp"
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, ckpt_path)
            since_best = 0
        else:
            since_best += 1

        if verbose and (epoch % 20 == 0 or epoch == cfg.n_epochs - 1):
            print(
                f'[{label}] ep {epoch:3d}  train {hist["train"][-1]:.3e}  '
                f'val {val_mse:.3e}  ({time.time() - t0:.0f}s)'
            )

        if cfg.patience is not None and since_best >= cfg.patience:
            hist["stopped_early_at"] = epoch
            if verbose:
                print(f"[{label}] early stop at ep {epoch} (no val gain in {cfg.patience})")
            break

    if verbose and hist["best_epoch"] is not None:
        print(f'[{label}] best val {hist["best_val"]:.3e} @ ep {hist["best_epoch"]}')
    return model, hist
