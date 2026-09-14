"""Quick, CPU-cheap retune of a pretrained FIM field to Duffing (quick cells 3 and 4).

This is the fast proof-of-concept counterpart to the heavy FIM cells in
``scripts/fim_factorial.py``: no ``L=200`` rollout, no backprop-through-time over a long
horizon.  It asks a narrow question -- can a pretrained ODE foundation model be *retuned
into* (cell 3, unstructured) or *carried by* (cell 4, structured) the bounded field
``A(x)(x - g)`` -- as a step toward a foundation model whose backbone learner is that
structured field rather than FIM's unbounded-polynomial prior.

The primary metric is drift MSE against the **true** Duffing field
(:func:`ftnode.systems.duffing_field_torch`; we own the plant), evaluated over the visited
region and on an out-of-region grid where the ``-q**3`` term forces the bounded structure
to break (see the ``structural-prior-limits`` note).

No true state is used to train a field or to build a FIM context: contexts come from the
measured ``q``-window only (``FIMBackbone.prepare``).  ``Xfull`` is used solely to define
evaluation query states and the true-field targets -- evaluation-oracle querying, never a
model input in training.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch

from ..physical.model import estimate_x0
from ..systems import DuffingDataset, DuffingParams, duffing_field_torch
from ..train import rk4_step

__all__ = [
    "field_drift",
    "set_backbone_trainable",
    "drift_mse_vs_true",
    "visited_box",
    "drift_mse_on_grid",
    "drift_mse_vs_reference",
    "retune_short_rollout",
    "distill_structured",
    "zero_shot_snapshot",
    "short_horizon_qmse",
    "RetuneConfig",
    "DistillConfig",
]


def field_drift(field, x, u):
    """The drift a field predicts at states ``x`` with input ``u``.

    A structured / free field exposes ``F(x, u)``; a bare :class:`FIMBackbone` exposes
    ``drift(x)`` and folds ``u`` into its bound context.  Bind the context with
    ``field.prepare(window, h)`` first.
    """
    F = getattr(field, "F", None)
    if F is not None:
        return F(x, u)
    return field.drift(x)


def set_backbone_trainable(field, flag: bool) -> None:
    """Set ``requires_grad`` on every backbone parameter except the uncertainty head.

    ``FreeFIMField`` / ``StructuredFIMField`` freeze only ``fim.u_model`` at construction;
    distillation needs the whole backbone frozen so only the structured head trains.  The
    uncertainty head stays frozen (it is off the drift path).
    """
    backbone = getattr(field, "backbone", field)
    u_ids = set()
    u_model = getattr(getattr(backbone, "fim", None), "u_model", None)
    if u_model is not None:
        u_ids = {id(p) for p in u_model.parameters()}
    for p in backbone.parameters():
        if id(p) not in u_ids:
            p.requires_grad_(flag)


def zero_shot_snapshot(backbone):
    """A deep copy of the (zero-shot) backbone, so a later retune cannot mutate it.

    Cell-3 retune steps the shared backbone (``lr_backbone > 0``); cell 4 distills the
    zero-shot field, so it must own an independent copy.
    """
    return copy.deepcopy(backbone)


@torch.no_grad()
def drift_mse_vs_true(field, ds: DuffingDataset, h: float, params: DuffingParams) -> dict:
    """Per-trajectory drift MSE against the true Duffing field over the visited states.

    Query states are the trajectory's own true states ``Xfull[:, t, :]`` (the visited
    region).  The context is the ``q``-window, bound once.  Each time slice is one query
    per trajectory, so the context and query batch dimensions stay aligned (``B=n_traj``),
    which is the batch contract ``FIMBackbone.prepare`` / ``drift`` expect.
    """
    prepare = getattr(field, "prepare", None)
    if prepare is not None:
        prepare(ds.W, h)
    device = ds.W.device
    B, T, d = ds.Xfull.shape
    u = ds.U
    se = torch.zeros(B, device=device)
    for t in range(T):
        x = ds.Xfull[:, t, :].to(device)
        pred = field_drift(field, x, u)
        true = duffing_field_torch(x, u, params)
        se = se + ((pred - true) ** 2).sum(-1)
    per = se / (T * d)
    return {
        "mean": float(per.mean()),
        "std": float(per.std()),
        "per_traj": per.detach().cpu().tolist(),
    }


@torch.no_grad()
def drift_mse_vs_reference(field, reference_drift, ds: DuffingDataset, h: float) -> dict:
    """Drift MSE of ``field`` against a fixed reference drift (e.g. the FIM target).

    ``reference_drift(x, u) -> (B, d)`` is any callable returning the target drift; used to
    score how well the distilled structured field reproduced the FIM drift it was fit to,
    separately from its error against the true field.
    """
    prepare = getattr(field, "prepare", None)
    if prepare is not None:
        prepare(ds.W, h)
    device = ds.W.device
    B, T, d = ds.Xfull.shape
    u = ds.U
    se = torch.zeros(B, device=device)
    for t in range(T):
        x = ds.Xfull[:, t, :].to(device)
        se = se + ((field_drift(field, x, u) - reference_drift(x, u)) ** 2).sum(-1)
    per = se / (T * d)
    return {"mean": float(per.mean()), "std": float(per.std())}


def visited_box(ds: DuffingDataset):
    """Axis-aligned ``(lo, hi)`` bounds of the true states the dataset visits."""
    X = ds.Xfull.reshape(-1, 2)
    return X.min(0).values, X.max(0).values


@torch.no_grad()
def drift_mse_on_grid(
    field, ds: DuffingDataset, h: float, params: DuffingParams, *, n: int = 15, scale: float = 2.5
) -> dict:
    """Drift MSE against the true field on a grid, split into in-region and out-of-region.

    The grid spans ``scale x`` the visited box, so it reaches states the trajectories never
    entered.  A grid point is *in-region* if it lies inside the visited box and *out-region*
    otherwise.  Each grid point is queried under every trajectory's context (one B-aligned
    batch per point), because the FIM drift is context-conditioned.

    This exposes the structural limit: ``A(x - g)`` grows at most linearly in ``||x||``
    while the true field carries ``-q**3``, so the out-of-region error is the honest signal.
    """
    prepare = getattr(field, "prepare", None)
    if prepare is not None:
        prepare(ds.W, h)
    device = ds.W.device
    lo, hi = visited_box(ds)
    lo, hi = lo.to(device), hi.to(device)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * scale
    qs = torch.linspace(float(center[0] - half[0]), float(center[0] + half[0]), n, device=device)
    qds = torch.linspace(float(center[1] - half[1]), float(center[1] + half[1]), n, device=device)
    B = ds.W.shape[0]
    u = ds.U
    in_se, in_n, out_se, out_n = 0.0, 0, 0.0, 0
    for qv in qs:
        for qdv in qds:
            pt = torch.stack([qv, qdv])
            inside = bool((pt >= lo).all() and (pt <= hi).all())
            x = pt.unsqueeze(0).expand(B, 2).contiguous()
            err = ((field_drift(field, x, u) - duffing_field_torch(x, u, params)) ** 2).sum(-1)
            s = float(err.sum())
            if inside:
                in_se += s
                in_n += B
            else:
                out_se += s
                out_n += B
    return {
        "in_region_mse": in_se / in_n / 2 if in_n else float("nan"),
        "out_region_mse": out_se / out_n / 2 if out_n else float("nan"),
        "in_points": in_n // B,
        "out_points": out_n // B,
        "grid_box": [[float(qs[0]), float(qs[-1])], [float(qds[0]), float(qds[-1])]],
    }


@dataclass(frozen=True)
class RetuneConfig:
    """Settings for the cheap cell-3 short-rollout retune."""

    epochs: int = 40
    L_short: int = 6
    lr: float = 3e-3
    lr_backbone: float = 5e-6
    batch: int = 32
    clip: float = 1.0


def _short_rollout_q(field, w, u, L_short, h):
    """Estimate ``x0`` from the window, integrate ``L_short`` steps, stack the ``q`` read."""
    x = estimate_x0(w, h)
    qs = [x[..., 0]]
    for _ in range(L_short):
        x = rk4_step(field.F, x, u, h)
        qs.append(x[..., 0])
    return torch.stack(qs, 1)


@torch.no_grad()
def short_horizon_qmse(field, ds: DuffingDataset, h: float, L_short: int) -> float:
    """Measured-``q`` MSE over a short ``L_short``-step rollout -- the secondary metric."""
    field.prepare(ds.W, h)
    qhat = _short_rollout_q(field, ds.W, ds.U, L_short, h)
    return float(((qhat - ds.Y[:, : L_short + 1]) ** 2).mean())


def retune_short_rollout(field, ds: DuffingDataset, h: float, cfg: RetuneConfig, *, verbose=False):
    """Retune a field on a short measured-``q`` rollout (no long horizon, minimal BPTT).

    Reuses the repository's real observation model -- ``estimate_x0`` then RK4 then compare
    only the measured ``q`` -- so it needs no finite-difference acceleration target.  Works
    for the free field (backbone trainable) and, with the backbone frozen, as the optional
    post-distillation polish of the structured head.  The FIM context is rebound every
    minibatch (:meth:`FIMBackbone.prepare`'s graph must not be reused after a backward).
    """
    opt = torch.optim.Adam(field.param_groups(cfg.lr, cfg.lr_backbone))
    N = ds.W.shape[0]
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=ds.W.device)
        batch_losses = []
        for i in range(0, N, cfg.batch):
            idx = perm[i : i + cfg.batch]
            w, u, y = ds.W[idx], ds.U[idx], ds.Y[idx]
            field.prepare(w, h)
            qhat = _short_rollout_q(field, w, u, cfg.L_short, h)
            loss = ((qhat - y[:, : cfg.L_short + 1]) ** 2).mean()
            if not torch.isfinite(loss):
                if verbose:
                    print(f"[retune] non-finite loss at epoch {ep}; stopping")
                return field, losses
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in field.parameters() if p.requires_grad), cfg.clip)
            opt.step()
            batch_losses.append(loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[retune] ep {ep:3d}  q-rollout mse {losses[-1]:.3e}")
    return field, losses


@dataclass(frozen=True)
class DistillConfig:
    """Settings for the cell-4 distillation of the FIM drift into the structured head."""

    epochs: int = 60
    lr: float = 3e-3
    batch: int = 64
    clip: float = 1.0


@torch.no_grad()
def _fim_drift_points(backbone, ds: DuffingDataset, h: float):
    """Detached zero-shot FIM drift at every (trajectory, time) state, B-aligned per slice.

    Precomputing the target once means the frozen 13M backbone is not re-run on every
    distillation step.  Returns flattened ``x (N, 2)``, ``u (N,)``, ``f_fim (N, 2)``.
    """
    backbone.prepare(ds.W, h)
    device = ds.W.device
    B, T, d = ds.Xfull.shape
    xs, us, fs = [], [], []
    for t in range(T):
        x = ds.Xfull[:, t, :].to(device)
        fs.append(backbone.drift(x))
        xs.append(x)
        us.append(ds.U)
    return torch.cat(xs, 0), torch.cat(us, 0), torch.cat(fs, 0).detach()


def distill_structured(struct_field, backbone, ds: DuffingDataset, h: float, cfg: DistillConfig, *, verbose=False):
    """Fit the structured head so ``A(x)(x - g) ~= f_FIM(x)`` on the visited region.

    ``backbone`` is the zero-shot backbone whose drift is the distillation target; it is the
    same instance ``struct_field`` reads its ``[drift, x]`` feature from.  The whole backbone
    is frozen, so only ``L_net`` / ``S_net`` / ``g_net`` train and ``A`` stays negative
    definite by construction.  The target drift is precomputed and detached, so the fit uses
    the field's feature-level builders (``_A_from_feat`` / ``_g_from_feat``) directly and
    never re-runs the backbone per step.
    """
    set_backbone_trainable(struct_field, False)
    x_all, u_all, f_all = _fim_drift_points(backbone, ds, h)
    feat_all = torch.cat([f_all, x_all], dim=-1)  # matches StructuredFIMField feat = [drift, x]
    head_params = [p for p in struct_field.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=cfg.lr)
    N = x_all.shape[0]
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=x_all.device)
        batch_losses = []
        for i in range(0, N, cfg.batch):
            idx = perm[i : i + cfg.batch]
            feat, x, u, tgt = feat_all[idx], x_all[idx], u_all[idx], f_all[idx]
            A = struct_field._A_from_feat(feat)
            g = struct_field._g_from_feat(feat, u)
            pred = torch.einsum("bij,bj->bi", A, x - g)
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, cfg.clip)
            opt.step()
            batch_losses.append(loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[distill] ep {ep:3d}  drift mse {losses[-1]:.3e}")
    return struct_field, losses
