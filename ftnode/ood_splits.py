"""Out-of-distribution data splits for the Duffing plant.

The standard generator :func:`ftnode.systems.make_dataset` draws every split from one
state box, so train and test differ only in seed and horizon.  The splits here move the
test outside the training data.  They are new functions on purpose: ``make_dataset``
keeps its random draw order, which the frozen notebooks and the factorial reproduce.

Regions (``SplitConfig.region``):

- ``trapped_right`` / ``trapped_left`` -- initial states that can never leave one well.
  For constant ``u`` the energy ``E = q_dot^2/2 + q^4/4 - q^2/2 - u q`` satisfies
  ``dE/dt = -delta q_dot^2 <= 0``, and on the line ``q = q_s(u)`` (the saddle)
  ``E >= E_s(u)``.  So ``E(x0) < E_s(u)`` together with ``q0 > q_s(u)`` keeps
  ``q(t) > q_s(u)`` for all ``t >= 0``, the context window included.
- ``half_right`` / ``half_left`` -- ``q0 > 0`` (or ``< 0``) with no energy filter, so a
  fraction of the trajectories crosses into the other well.
- ``far_ring`` -- ``|q0|`` in ``far_q``, outside the standard box, both signs.

Every ``*_left`` region is the exact mirror of its ``*_right`` region at the same seed:
the plant is odd under ``(q, q_dot, u) -> (-q, -q_dot, -u)``, so the left split is built
by negating the right one.  The proposal box is the standard ``[-x_range, x_range] x
[-v_range, v_range]``; the trapped regions are its intersection with the trap set.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .systems import DuffingDataset, DuffingParams, equilibria, simulate

__all__ = [
    "REGIONS",
    "energy",
    "saddle",
    "barrier_energy",
    "is_trapped",
    "SplitConfig",
    "make_split",
    "mirror",
]

REGIONS = ("trapped_right", "trapped_left", "half_right", "half_left", "far_ring")


def energy(q, qd, u):
    """Duffing energy ``q_dot^2/2 + q^4/4 - q^2/2 - u q`` (numpy or torch, broadcasting)."""
    return 0.5 * qd**2 + 0.25 * q**4 - 0.5 * q**2 - u * q


def saddle(u) -> float:
    """The saddle ``q_s(u)``: the middle real root of ``q^3 - q - u``.

    Defined only inside the pitchfork region ``|u| < 2/(3 sqrt 3)``, where the plant has
    two wells.
    """
    eqs = equilibria(u)
    if len(eqs) != 3:
        raise ValueError(f"u={u} is outside the two-well region; no saddle")
    return float(eqs[1][0])


def barrier_energy(u) -> float:
    """``E_s(u)``, the energy at the saddle ``(q_s(u), 0)``."""
    return float(energy(saddle(u), 0.0, u))


def is_trapped(q0, qd0, u, side=+1) -> bool:
    """True when ``(q0, qd0)`` can never leave the well on ``side`` (+1 right, -1 left)."""
    qs = saddle(u)
    return bool(energy(q0, qd0, u) < barrier_energy(u) and side * (q0 - qs) > 0)


@dataclass(frozen=True)
class SplitConfig:
    """One out-of-distribution split.

    ``tau``, ``L``, ``h``, ``u_range``, ``x_range`` and ``delta`` mean the same as in
    :class:`ftnode.systems.DuffingDataConfig`.  ``v_range`` is the ``q_dot`` half-width
    of the proposal box (``make_dataset`` fixes it at 1).  ``far_q`` is the ``|q0|``
    interval of the far ring.  ``max_draws`` caps the rejection sampler.
    """

    region: str
    n_traj: int = 512
    L: int = 200
    tau: int = 8
    h: float = 0.05
    u_range: float = 0.25
    x_range: float = 1.6
    v_range: float = 1.0
    far_q: tuple[float, float] = (2.0, 4.0)
    delta: float = 0.2
    seed: int = 0
    max_draws: int = 1_000_000

    @property
    def params(self) -> DuffingParams:
        return DuffingParams(delta=self.delta)


_MIRROR_OF = {"trapped_left": "trapped_right", "half_left": "half_right"}


def _accept(region, q, qd, u) -> bool:
    if region == "trapped_right":
        return is_trapped(q, qd, u, side=+1)
    if region == "half_right":
        return q > 0.0
    raise AssertionError(region)


def _simulate_accepted(cfg: SplitConfig, rng):
    """Draw and simulate ``n_traj`` accepted trajectories for a right-side or far region.

    Returns ``Xfull`` of shape ``(n_traj, tau+L+1, 2)`` (float64) and ``U`` of shape
    ``(n_traj,)``.  A trapped draw is also checked after simulation: the energy test is
    exact for the plant, but RK4 does not dissipate energy exactly, so a draw just under
    the barrier could cross numerically.  Such a draw is rejected.
    """
    t_grid = np.arange(cfg.tau + cfg.L + 1) * cfg.h
    xs, us, draws = [], [], 0
    while len(us) < cfg.n_traj:
        if draws >= cfg.max_draws:
            raise RuntimeError(
                f"{cfg.region}: accepted {len(us)}/{cfg.n_traj} in {draws} draws; "
                "raise max_draws or widen the proposal box"
            )
        draws += 1
        if cfg.region == "far_ring":
            lo, hi = cfg.far_q
            q = rng.uniform(lo, hi) * (1.0 if rng.uniform() < 0.5 else -1.0)
            qd = rng.uniform(-cfg.v_range, cfg.v_range)
            u = float(rng.uniform(-cfg.u_range, cfg.u_range))
        else:
            q = rng.uniform(-cfg.x_range, cfg.x_range)
            qd = rng.uniform(-cfg.v_range, cfg.v_range)
            u = float(rng.uniform(-cfg.u_range, cfg.u_range))
            if not _accept(cfg.region, q, qd, u):
                continue
        x = simulate(np.array([q, qd]), lambda t, _u=u: _u, cfg.params, t_grid)
        if cfg.region == "trapped_right" and not (x[:, 0] > saddle(u)).all():
            continue
        xs.append(x)
        us.append(u)
    return np.stack(xs), np.asarray(us)


def mirror(ds: DuffingDataset) -> DuffingDataset:
    """The exact mirror image ``(q, q_dot, u) -> (-q, -q_dot, -u)`` of a dataset."""
    return DuffingDataset(W=-ds.W, U=-ds.U, Y=-ds.Y, Xfull=-ds.Xfull)


def make_split(cfg: SplitConfig) -> DuffingDataset:
    """Simulate one out-of-distribution split; same layout as ``make_dataset``.

    ``W`` is the ``(n, tau)`` window, ``U`` the ``(n,)`` constant input, ``Y`` the
    ``(n, L+1)`` measured ``q`` after the window, and ``Xfull`` the matching true state.
    """
    if cfg.region not in REGIONS:
        raise ValueError(f"unknown region {cfg.region!r}; expected one of {REGIONS}")
    # The trapping proof needs dE/dt = -delta q_dot^2 <= 0.
    if not (np.isfinite(cfg.delta) and cfg.delta >= 0):
        raise ValueError(f"delta must be finite and >= 0; got {cfg.delta}")
    if not (np.isfinite(cfg.h) and cfg.h > 0):
        raise ValueError(f"h must be finite and > 0; got {cfg.h}")
    if cfg.region in _MIRROR_OF:
        return mirror(make_split(replace(cfg, region=_MIRROR_OF[cfg.region])))

    rng = np.random.default_rng(cfg.seed)
    X, us = _simulate_accepted(cfg, rng)
    Xfull = X.astype(np.float32)
    tau, L = cfg.tau, cfg.L
    Y = Xfull[..., 0]
    return DuffingDataset(
        W=torch.from_numpy(Y[:, :tau].copy()),
        U=torch.from_numpy(us.astype(np.float32)),
        Y=torch.from_numpy(Y[:, tau : tau + L + 1].copy()),
        Xfull=torch.from_numpy(Xfull[:, tau : tau + L + 1].copy()),
    )
