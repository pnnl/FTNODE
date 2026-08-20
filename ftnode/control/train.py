"""Training the control splitting ``psi`` against frozen-model closed-loop rollouts."""
from __future__ import annotations

import time

import numpy as np
import torch

from .operator import ControlConfig
from .policy import cost_T, g_range_penalty

__all__ = ["train_psi"]


def train_psi(
    fpsi,
    plant,
    z_star,
    u_star,
    cfg: ControlConfig,
    *,
    Qz=None,
    c_z=None,
    W=None,
    seed: int = 0,
    verbose: bool = True,
):
    """The constrained offline problem: train ``psi``, and only ``psi``.

    ``F_theta`` stays frozen, so the represented plant is untouched by
    construction -- see :func:`~ftnode.control.policy.g_psi`.  The horizon
    follows an increasing schedule over ``cfg.T_list`` to blunt truncation error.

    Two behaviours here are load-bearing rather than defensive:

    * Non-finite **losses** skip the batch.
    * Non-finite **gradients** are DROPPED, not clipped.  ``clip_grad_norm_``
      propagates ``inf``/``nan`` straight into the parameters *and* into Adam's
      first and second moments, which corrupts every subsequent step -- the run
      never recovers.  Rescaling cannot fix a gradient that is already ``inf``.

    Returns ``(fpsi, hist)``; ``hist`` carries per-epoch ``cost``, ``gpen``,
    ``gmax``, ``T``, ``gradnorm`` and a cumulative ``n_skip``.
    """
    device = plant.device
    m = plant.m
    Qz = torch.eye(m, device=device) if Qz is None else Qz
    c_z = z_star if c_z is None else c_z

    g = torch.Generator().manual_seed(seed)
    if cfg.z0_spread is None:
        Z0 = ((2 * torch.rand(cfg.n_ic, m, generator=g) - 1) * cfg.z_scale).to(device)
    else:
        Z0 = (z_star.cpu() + cfg.z0_spread * torch.randn(cfg.n_ic, m, generator=g)).to(device)

    opt = torch.optim.Adam(fpsi.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    hist = {"cost": [], "gpen": [], "gmax": [], "T": [], "gradnorm": [], "n_skip": 0}
    t0 = time.time()

    for ep in range(cfg.n_epochs):
        T_steps = cfg.T_list[
            min(len(cfg.T_list) - 1, int(ep / max(1, cfg.n_epochs // len(cfg.T_list))))
        ]
        perm = torch.randperm(cfg.n_ic, device=device)
        ep_c, ep_g, ep_gm, ep_gn = [], [], 0.0, []

        for i in range(0, cfg.n_ic, cfg.batch):
            z0 = Z0[perm[i : i + cfg.batch]]
            C, zs, us = cost_T(
                fpsi,
                plant,
                z0,
                z_star,
                u_star,
                T_steps,
                cfg.h,
                cfg.eta,
                Qz,
                cfg.R_u,
                cfg.u_lo,
                cfg.u_hi,
                W,
                k_trunc=cfg.k_trunc,
            )
            # L_g sampled on the visited latents at the admissible u applied there.
            # zs has T+1 nodes, us has T -- pair each state with the input held over its step.
            Zs = zs[:, :-1].reshape(-1, m).detach()
            Us = us.reshape(-1, 1).detach()
            Lg, d = g_range_penalty(fpsi, plant, Zs, Us, c_z, cfg.r_in_target)
            loss = C + cfg.lam_g * Lg

            if not torch.isfinite(loss):
                hist["n_skip"] += 1
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(fpsi.parameters(), 1.0)
            if not torch.isfinite(gn):
                hist["n_skip"] += 1
                opt.zero_grad()
                continue
            opt.step()

            ep_c.append(C.item())
            ep_g.append(Lg.item())
            ep_gm = max(ep_gm, d.max().item())
            ep_gn.append(gn.item())

        sched.step()
        hist["cost"].append(float(np.mean(ep_c)) if ep_c else float("nan"))
        hist["gpen"].append(float(np.mean(ep_g)) if ep_g else float("nan"))
        hist["gmax"].append(ep_gm)
        hist["T"].append(T_steps)
        hist["gradnorm"].append(float(np.max(ep_gn)) if ep_gn else float("nan"))

        if verbose and (ep % 10 == 0 or ep == cfg.n_epochs - 1):
            print(
                f'  [psi] ep {ep:3d} T={T_steps:3d}  C_T {hist["cost"][-1]:.4e}  '
                f'L_g {hist["gpen"][-1]:.3e}  max||g-c|| {ep_gm:.3f}  '
                f'|grad| {hist["gradnorm"][-1]:.2e}  ({time.time() - t0:.0f}s)'
            )

    return fpsi, hist
