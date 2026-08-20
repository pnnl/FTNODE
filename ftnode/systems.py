"""Duffing plant and the partial-observation dataset used by the latent FT-NODE work.

Promoted verbatim from the duffing example notebooks (see
``examples/duffing/duffing_kappa_svdclamp_vs_ln_2variant_10seed.ipynb`` and
``..._youla_skew_3variant_10seed.ipynb``), with the module-level globals those
notebooks close over (``params``, ``x_range``, ``u_range``, ``tau``, ``L``,
``h_dt``) turned into explicit config fields.

The plant is the forced Duffing oscillator

    q_ddot + delta q_dot + q^3 - q = u,

written as the first-order field ``[q_dot, -delta q_dot - q^3 + q + u]``.  Only
``y = q`` is measured; the second state ``q_dot`` is never observed, which is
what makes the identification problem a latent one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "DuffingParams",
    "duffing_field",
    "duffing_field_torch",
    "equilibria",
    "sinks",
    "simulate",
    "DuffingDataConfig",
    "DuffingDataset",
    "make_dataset",
]


@dataclass(frozen=True)
class DuffingParams:
    """Physical parameters of the plant.  ``delta`` is the damping coefficient."""

    delta: float = 0.2


def duffing_field(x, u, params: DuffingParams):
    """True Duffing field, numpy, batched over leading dims.

    ``x`` is ``(..., 2)`` and ``u`` broadcasts against ``x[..., 0]``.  Named
    ``duffing_field`` rather than the notebooks' bare ``F`` because a module-level
    ``F`` is unusable at package scope.
    """
    x1 = x[..., 0]
    x2 = x[..., 1]
    return np.stack([x2, -params.delta * x2 - x1**3 + x1 + np.asarray(u)], axis=-1)


def duffing_field_torch(x, u, params: DuffingParams):
    """True Duffing field, torch, batched.  ``x`` is ``(b, 2)``, ``u`` is ``(b, 1)`` or ``(b,)``.

    The torch twin of :func:`duffing_field`, needed by the control stage where the
    deployment loop is integrated inside an autograd graph.
    """
    q, qd = x[..., 0], x[..., 1]
    uu = u.squeeze(-1) if u.dim() == x.dim() else u
    return torch.stack([qd, -params.delta * qd - q**3 + q + uu], dim=-1)


def equilibria(u):
    """Real equilibria of the unforced-in-velocity system: roots of ``q^3 - q - u``.

    Returns them as ``[q, 0]`` pairs sorted by ``q``: three of them (sink, saddle,
    sink) inside the pitchfork region ``|u| < 2/(3 sqrt(3))``, one outside.
    """
    roots = np.roots([1.0, 0.0, -1.0, -float(u)])
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    return [np.array([r, 0.0]) for r in real]


def sinks(u):
    """The stable equilibria at input ``u``, keyed ``left``/``right`` (or ``only``)."""
    eqs = equilibria(u)
    return {"left": eqs[0], "right": eqs[2]} if len(eqs) == 3 else {"only": eqs[0]}


def simulate(x0, u_of_t, params: DuffingParams, t_grid):
    """Fixed-step RK4 on the true plant with a time-varying input.  Returns ``(T, 2)``."""
    xs = np.empty((len(t_grid), 2))
    xs[0] = x0
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        h = t_grid[i + 1] - t
        x = xs[i]
        u1 = u_of_t(t)
        u2 = u_of_t(t + 0.5 * h)
        u4 = u_of_t(t + h)
        k1 = duffing_field(x, u1, params)
        k2 = duffing_field(x + 0.5 * h * k1, u2, params)
        k3 = duffing_field(x + 0.5 * h * k2, u2, params)
        k4 = duffing_field(x + h * k3, u4, params)
        xs[i + 1] = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xs


@dataclass(frozen=True)
class DuffingDataConfig:
    """Dataset generation settings.

    Defaults are the values every duffing notebook uses, so
    ``make_dataset(DuffingDataConfig())`` reproduces their training set exactly.
    The validation set in those notebooks is ``DuffingDataConfig(n_traj=64,
    L=600, seed=1)``.
    """

    n_traj: int = 512
    L: int = 200
    tau: int = 8
    h: float = 0.05
    u_range: float = 0.25
    x_range: float = 1.6
    delta: float = 0.2
    seed: int = 0

    @property
    def params(self) -> DuffingParams:
        return DuffingParams(delta=self.delta)


@dataclass
class DuffingDataset:
    """One split of the partial-observation dataset.

    ``W`` is the ``(n, tau)`` encoder window of past measurements, ``U`` the
    ``(n,)`` constant input held over the trajectory, ``Y`` the ``(n, L+1)``
    measured output ``q``, and ``Xfull`` the ``(n, L+1, 2)`` true state (kept for
    diagnostics only -- no model ever sees it).

    Replaces the six loose ``Wtr_d/Utr_d/Ytr_d/Wva_d/...`` globals that the
    notebooks' ``train_one`` closes over.
    """

    W: torch.Tensor
    U: torch.Tensor
    Y: torch.Tensor
    Xfull: torch.Tensor

    def to(self, device) -> "DuffingDataset":
        """Move the tensors the trainer touches onto ``device``.

        ``Xfull`` stays on the CPU: it is diagnostics-only and can be large.
        """
        return DuffingDataset(
            W=self.W.to(device),
            U=self.U.to(device),
            Y=self.Y.to(device),
            Xfull=self.Xfull,
        )

    def __len__(self) -> int:
        return self.W.shape[0]


def make_dataset(cfg: DuffingDataConfig) -> DuffingDataset:
    """Simulate ``cfg.n_traj`` trajectories under constant inputs and window them.

    Initial conditions are drawn uniformly from ``[-x_range, x_range] x [-1, 1]``
    and the input uniformly from ``[-u_range, u_range]``, held constant over the
    trajectory.  The RNG draw order (``x0`` then ``u``, per trajectory) is
    preserved from the notebooks so a given seed reproduces the same data.
    """
    rng = np.random.default_rng(cfg.seed)
    params = cfg.params
    tau, L = cfg.tau, cfg.L
    total = tau + L + 1
    t_grid = np.arange(total) * cfg.h

    Xfull = np.zeros((cfg.n_traj, total, 2), dtype=np.float32)
    U = np.zeros((cfg.n_traj,), dtype=np.float32)
    for i in range(cfg.n_traj):
        x0 = np.array([rng.uniform(-cfg.x_range, cfg.x_range), rng.uniform(-1.0, 1.0)])
        u = float(rng.uniform(-cfg.u_range, cfg.u_range))
        Xfull[i] = simulate(x0, lambda t, _u=u: _u, params, t_grid).astype(np.float32)
        U[i] = u

    Y = Xfull[..., 0]
    return DuffingDataset(
        W=torch.from_numpy(Y[:, :tau]),
        U=torch.from_numpy(U),
        Y=torch.from_numpy(Y[:, tau : tau + L + 1]),
        Xfull=torch.from_numpy(Xfull[:, tau : tau + L + 1]),
    )
