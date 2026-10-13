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
measured ``q``-window only (``FIMBackbone.prepare``), and training query states are rebuilt
from the measured ``q`` by finite differences (:func:`reconstruct_states`), because
``q_dot`` is never measured.  ``Xfull`` is used solely to define evaluation query states
and the true-field targets -- evaluation-oracle querying, never a model input in training.
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
    "reconstruct_states",
    "drift_targets",
    "retune_drift_match",
    "DriftMatchConfig",
    "set_backbone_trainable",
    "set_retune_scope",
    "disable_backbone_dropout",
    "SCOPES",
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


SCOPES = ("R0", "R1", "R2")


def set_retune_scope(field, scope: str) -> None:
    """Set which FIM backbone parameters train; the head always trains.

    - ``R0`` -- head only; the whole backbone is frozen.
    - ``R1`` -- head and the decoder side (``location_proj`` and ``functional_decoder``);
      the context encoder ``trajectory_encoder`` is frozen.
    - ``R2`` -- head and the whole drift path.

    The uncertainty head ``u_model`` is off the drift path and stays frozen in every scope.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
    set_backbone_trainable(field, scope == "R2")
    if scope == "R1":
        fim = field.backbone.fim
        for module in (fim.location_proj, fim.functional_decoder):
            for p in module.parameters():
                p.requires_grad_(True)


def disable_backbone_dropout(field) -> None:
    """Put the FIM backbone in eval mode, so its dropout layers are off.

    The pretrained FIM loads in training mode with dropout ``p = 0.1``.  Left on, it adds
    random noise to every drift query: to a frozen backbone's output, to distillation
    targets, and to metrics.  It would also regularize only the scopes that train the
    backbone.  The heads have no dropout, so their mode does not matter.  Gradient
    checkpointing keys on ``torch.is_grad_enabled()``, not on the mode, so it is unaffected.
    """
    getattr(field, "backbone", field).eval()


def zero_shot_snapshot(backbone):
    """A deep copy of the (zero-shot) backbone, so a later retune cannot mutate it.

    Cell-3 retune steps the shared backbone (``lr_backbone > 0``); cell 4 distills the
    zero-shot field, so it must own an independent copy.
    """
    return copy.deepcopy(backbone)


def reconstruct_states(ds: DuffingDataset, h: float) -> torch.Tensor:
    """States ``(q, q_dot_hat)`` at every target time, from the measured ``q`` only.

    The window and the targets are consecutive samples, so ``[W, Y]`` is one measured
    record.  ``q_dot_hat`` is its second-order central difference (one-sided at the ends).
    Returns ``(n_traj, L+1, 2)``, aligned with ``ds.Y`` and with ``ds.Xfull``.
    """
    q = torch.cat([ds.W, ds.Y], dim=1)
    qd = torch.gradient(q, spacing=h, dim=1)[0]
    tau = ds.W.shape[1]
    return torch.stack([q[:, tau:], qd[:, tau:]], dim=-1)


def drift_targets(ds: DuffingDataset, h: float):
    """Reconstructed states and finite-difference drift targets, from the measured ``q`` only.

    ``x_hat = (q, q_dot_hat)`` as in :func:`reconstruct_states`, and
    ``xdot_hat = (q_dot_hat, q_ddot_hat)`` with ``q_ddot_hat`` the second difference
    ``(q[k+1] - 2 q[k] + q[k-1]) / h^2``.  The last target time has no forward neighbour,
    so it is dropped.  Returns two ``(n_traj, L, 2)`` tensors.
    """
    X = reconstruct_states(ds, h)[:, :-1]
    q = torch.cat([ds.W, ds.Y], dim=1)
    tau, N = ds.W.shape[1], q.shape[1]
    qdd = (q[:, tau + 1 : N] - 2 * q[:, tau : N - 1] + q[:, tau - 1 : N - 2]) / h**2
    return X, torch.stack([X[..., 1], qdd], dim=-1)


@torch.no_grad()
def drift_mse_vs_true(field, ds: DuffingDataset, h: float, params: DuffingParams) -> dict:
    """Per-trajectory drift MSE against the true Duffing field over the visited states.

    Query states are the trajectory's own true states ``Xfull[:, t, :]`` (the visited
    region).  The context is the ``q``-window, bound once.  Each time slice is one query
    per trajectory, so the context and query batch dimensions stay aligned (``B=n_traj``),
    which is the batch contract ``FIMBackbone.prepare`` / ``drift`` expect.
    """
    disable_backbone_dropout(field)
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
    disable_backbone_dropout(field)
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
    disable_backbone_dropout(field)
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
    """Settings for the cheap short-rollout retune.

    ``random_start`` draws each trajectory's start time anywhere in its measured record,
    so the objective sees the whole trajectory.  When ``False`` every rollout starts right
    after the context window, so only its first ``L_short + 1`` targets are ever used.
    """

    epochs: int = 40
    L_short: int = 6
    lr: float = 3e-3
    lr_backbone: float = 5e-6
    batch: int = 32
    clip: float = 1.0
    random_start: bool = False


def _optimizer(field, lr, lr_backbone):
    """Adam over the trainable parameters; a FIM field's ``param_groups`` set the rates."""
    if hasattr(field, "param_groups"):
        return torch.optim.Adam(field.param_groups(lr, lr_backbone))
    return torch.optim.Adam([p for p in field.parameters() if p.requires_grad], lr=lr)


def _start_windows(ds: DuffingDataset, idx, L_short: int, random_start: bool):
    """Context windows and ``q`` targets for the trajectories ``idx``.

    A start index ``k`` is the first target time in the record ``[W, Y]``.  The window is
    the ``tau`` samples before ``k``, the targets are ``q[k : k + L_short + 1]``.
    ``k = tau`` reproduces ``(W, Y[:, :L_short+1])``.
    """
    if not random_start:
        return ds.W[idx], ds.Y[idx, : L_short + 1]
    q = torch.cat([ds.W[idx], ds.Y[idx]], dim=1)
    tau = ds.W.shape[1]
    k = torch.randint(tau, q.shape[1] - L_short, (len(idx), 1), device=q.device)
    w = torch.gather(q, 1, k - tau + torch.arange(tau, device=q.device))
    y = torch.gather(q, 1, k + torch.arange(L_short + 1, device=q.device))
    return w, y


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
    disable_backbone_dropout(field)
    field.prepare(ds.W, h)
    qhat = _short_rollout_q(field, ds.W, ds.U, L_short, h)
    return float(((qhat - ds.Y[:, : L_short + 1]) ** 2).mean())


def retune_short_rollout(field, ds: DuffingDataset, h: float, cfg: RetuneConfig, *, verbose=False,
                         on_epoch=None):
    """Retune a field on a short measured-``q`` rollout (no long horizon, minimal BPTT).

    Reuses the repository's real observation model -- ``estimate_x0`` then RK4 then compare
    only the measured ``q`` -- so it needs no finite-difference acceleration target.  Works
    for the free field (backbone trainable) and, with the backbone frozen, as the optional
    post-distillation polish of the structured head.  The FIM context is rebound every
    minibatch (:meth:`FIMBackbone.prepare`'s graph must not be reused after a backward).
    A field without ``prepare`` (a cold field) skips the context binding.
    ``on_epoch(ep, loss)``, when given, runs after each epoch (for example, to log
    diagnostics); it must leave the field's parameters unchanged.
    """
    disable_backbone_dropout(field)
    opt = _optimizer(field, cfg.lr, cfg.lr_backbone)
    prepare = getattr(field, "prepare", None)
    N = ds.W.shape[0]
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=ds.W.device)
        batch_losses = []
        for i in range(0, N, cfg.batch):
            idx = perm[i : i + cfg.batch]
            w, y = _start_windows(ds, idx, cfg.L_short, cfg.random_start)
            u = ds.U[idx]
            if prepare is not None:
                prepare(w, h)
            qhat = _short_rollout_q(field, w, u, cfg.L_short, h)
            loss = ((qhat - y) ** 2).mean()
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
        if on_epoch is not None:
            on_epoch(ep, losses[-1])
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[retune] ep {ep:3d}  q-rollout mse {losses[-1]:.3e}")
    return field, losses


@dataclass(frozen=True)
class DriftMatchConfig:
    """Settings for the drift-matching retune (regression on finite-difference drift).

    ``n_times`` is the number of target times drawn per minibatch; each time is one FIM
    query per trajectory, so it sets the per-step cost.
    """

    epochs: int = 40
    lr: float = 3e-3
    lr_backbone: float = 5e-6
    batch: int = 32
    n_times: int = 8
    clip: float = 1.0


def retune_drift_match(field, ds: DuffingDataset, h: float, cfg: DriftMatchConfig, *, verbose=False,
                       on_epoch=None):
    """Fit a field's drift to finite-difference targets: ``F(x_hat, u) ~= xdot_hat``.

    A regression on the vector field with no rollout, the estimated-label analogue of FIM's
    pretraining objective.  States and targets come from :func:`drift_targets` (measured
    ``q`` only).  ``field`` exposes ``F(x, u)`` and, for a FIM field, ``prepare(window, h)``
    and ``param_groups``; only parameters with ``requires_grad`` train, so the caller sets
    the retune scope.  The FIM context is rebound every minibatch, and each drawn time is
    queried as one state per trajectory, which keeps the context and query batches aligned.
    ``on_epoch(ep, loss)``, when given, runs after each epoch.
    """
    disable_backbone_dropout(field)
    X, Xdot = drift_targets(ds, h)
    T = X.shape[1]
    opt = _optimizer(field, cfg.lr, cfg.lr_backbone)
    prepare = getattr(field, "prepare", None)
    N = X.shape[0]
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=X.device)
        batch_losses = []
        for i in range(0, N, cfg.batch):
            idx = perm[i : i + cfg.batch]
            if prepare is not None:
                prepare(ds.W[idx], h)
            times = torch.randperm(T)[: min(cfg.n_times, T)]
            u = ds.U[idx]
            loss = torch.stack(
                [((field.F(X[idx, t], u) - Xdot[idx, t]) ** 2).mean() for t in times.tolist()]
            ).mean()
            if not torch.isfinite(loss):
                if verbose:
                    print(f"[drift-match] non-finite loss at epoch {ep}; stopping")
                return field, losses
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in field.parameters() if p.requires_grad), cfg.clip)
            opt.step()
            batch_losses.append(loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
        if on_epoch is not None:
            on_epoch(ep, losses[-1])
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[drift-match] ep {ep:3d}  drift mse {losses[-1]:.3e}")
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

    The query states come from :func:`reconstruct_states` (measured ``q`` only), never from
    the true state ``Xfull``.  Precomputing the target once means the frozen 13M backbone is
    not re-run on every distillation step.  Returns flattened ``x (N, 2)``, ``u (N,)``,
    ``f_fim (N, 2)``.
    """
    disable_backbone_dropout(backbone)
    backbone.prepare(ds.W, h)
    X = reconstruct_states(ds, h)
    T = X.shape[1]
    xs, us, fs = [], [], []
    for t in range(T):
        x = X[:, t, :]
        fs.append(backbone.drift(x))
        xs.append(x)
        us.append(ds.U)
    return torch.cat(xs, 0), torch.cat(us, 0), torch.cat(fs, 0).detach()


def distill_structured(struct_field, backbone, ds: DuffingDataset, h: float, cfg: DistillConfig, *, verbose=False,
                       on_epoch=None):
    """Fit the structured head so ``A(x)(x - g) ~= f_FIM(x)`` on the visited region.

    ``backbone`` is the zero-shot backbone whose drift is the distillation target; it is the
    same instance ``struct_field`` reads its ``[drift, x]`` feature from.  The whole backbone
    is frozen, so only ``L_net`` / ``S_net`` / ``g_net`` train and ``A`` stays negative
    definite by construction.  The target drift is precomputed and detached, so the fit uses
    the field's feature-level builders (``_A_from_feat`` / ``_g_from_feat``) directly and
    never re-runs the backbone per step.  ``on_epoch(ep, loss)``, when given, runs after
    each epoch.
    """
    set_backbone_trainable(struct_field, False)
    disable_backbone_dropout(struct_field)
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
        if on_epoch is not None:
            on_epoch(ep, losses[-1])
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[distill] ep {ep:3d}  drift mse {losses[-1]:.3e}")
    return struct_field, losses
