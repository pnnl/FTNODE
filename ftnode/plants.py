"""A registry of small plants for the multi-system latent tests.

:mod:`ftnode.systems` holds Duffing alone, with the draw order the frozen notebooks
reproduce.  This module adds the other plants of the latent pH head test without touching
it: each plant is a first-order field ``x' = f(x, u)`` with a scalar constant input, one
scalar output ``y = c^T x``, a proposal box for initial states, and the input ranges and
out-of-distribution regions its test splits use.

Every plant settles to an equilibrium at every constant input in its ranges (no limit
cycles, no chaos) -- the target class of the latent work.  The mechanical plants settle by
energy decrease and LaSalle; ``toggle`` by Bendixson (its divergence is -2).

Fields are written once, in torch, so the same function serves batched RK4 in float64 and
``torch.func`` Jacobians for the critical points.  ``duffing`` and ``duffing_light`` call
:func:`ftnode.systems.duffing_field_torch`, so they agree with the notebooks' plant exactly.

Critical points are found numerically (damped Newton from a grid of starts) rather than
written per plant, so a new plant needs only its field and its boxes.
"""
from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .systems import DuffingParams, duffing_field_torch

__all__ = [
    "Plant",
    "PLANTS",
    "PLANT_IDS",
    "get_plant",
    "output",
    "simulate",
    "CriticalPoint",
    "critical_points",
    "sinks",
    "separator_output",
]


@dataclass(frozen=True)
class Plant:
    """One plant and the regions its splits draw from.

    ``box`` is the proposal box of initial states, one ``(lo, hi)`` per state coordinate.
    ``u_train`` is the training input interval; ``u_near`` and ``u_fold`` are unions of
    intervals beyond it -- the first before any fold of the training-range equilibrium
    structure, the second beyond a fold.  ``far_coords`` lists the coordinates the far
    split moves outside the box (one at a time) into ``far_band``, mirrored to both signs
    when ``far_signed``.  ``sep_index`` is the Morse index of the critical point whose
    output separates the two sides of the one-side split (``None`` for a monostable
    plant).  ``u_scan`` bounds the input range the fold table covers.
    """

    name: str
    n: int
    c: tuple
    field: Callable
    box: tuple
    u_train: tuple
    u_near: tuple = ()
    u_fold: tuple = ()
    far_coords: tuple = (0,)
    far_band: tuple = (2.0, 4.0)
    far_signed: bool = True
    sep_index: int | None = 1
    u_scan: tuple = (-0.8, 0.8)

    @property
    def c_vec(self) -> np.ndarray:
        return np.asarray(self.c, dtype=np.float64)


def _duffing(delta):
    params = DuffingParams(delta=delta)
    return lambda x, u: duffing_field_torch(x, u, params)


def _asym_well(x, u):
    q, v = x[..., 0], x[..., 1]
    return torch.stack([v, -(q**3 + 0.5 * q**2 - q) - (0.2 + 0.1 * q**2) * v + u], dim=-1)


def _hardening(x, u):
    q, v = x[..., 0], x[..., 1]
    return torch.stack([v, -(q + q**3) - 0.2 * v + u], dim=-1)


def _toggle(x, u):
    # The input enters both production terms with opposite signs, so the swap
    # (x1, x2, u) -> (x2, x1, -u) is a symmetry and the bistable interval is centred at 0.
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack([(4.0 + u) / (1.0 + x2**2) - x1, (4.0 - u) / (1.0 + x1**2) - x2], dim=-1)


def _coupled(x, u):
    q1, q2, v1, v2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    k = 0.2
    a1 = -(q1**3 - q1 + k * (q1 - q2)) - 0.2 * v1 + u
    a2 = -(q2**3 - q2 + k * (q2 - q1)) - 0.2 * v2
    return torch.stack([v1, v2, a1, a2], dim=-1)


_MECH_BOX = ((-1.6, 1.6), (-1.0, 1.0))
_DUFFING_BANDS = dict(u_near=((-0.35, -0.27), (0.27, 0.35)), u_fold=((-0.60, -0.45), (0.45, 0.60)))

PLANTS: dict[str, Plant] = {
    p.name: p
    for p in (
        Plant("duffing", 2, (1.0, 0.0), _duffing(0.2), _MECH_BOX, (-0.25, 0.25), **_DUFFING_BANDS),
        Plant("duffing_light", 2, (1.0, 0.0), _duffing(0.05), _MECH_BOX, (-0.25, 0.25),
              **_DUFFING_BANDS),
        # Folds at u = -0.258 and u = 0.610: train inside, probe one band on each side.
        Plant("asym_well", 2, (1.0, 0.0), _asym_well, _MECH_BOX, (-0.20, 0.20),
              u_near=((0.25, 0.40),), u_fold=((-0.45, -0.30),)),
        Plant("hardening", 2, (1.0, 0.0), _hardening, _MECH_BOX, (-0.25, 0.25),
              u_near=((-0.35, -0.27), (0.27, 0.35)), sep_index=None),
        Plant("toggle", 2, (1.0, 0.0), _toggle, ((0.0, 5.0), (0.0, 5.0)), (-0.25, 0.25),
              u_near=((-0.45, -0.30), (0.30, 0.45)), u_fold=((-0.80, -0.60), (0.60, 0.80)),
              far_coords=(0, 1), far_band=(5.0, 10.0), far_signed=False, u_scan=(-1.0, 1.0)),
        # Four sinks for |u| <= 0.115; the index-2 point separates the two sides.
        Plant("coupled", 4, (1.0, 0.5, 0.0, 0.0), _coupled,
              ((-1.6, 1.6), (-1.6, 1.6), (-1.0, 1.0), (-1.0, 1.0)), (-0.10, 0.10),
              u_fold=((-0.40, -0.15), (0.15, 0.40)), far_coords=(0, 1), sep_index=2),
    )
}

#: Fixed integer ids that enter every ``SeedSequence``; never reorder.
PLANT_IDS = {"duffing": 0, "duffing_light": 1, "asym_well": 2, "hardening": 3, "toggle": 4,
             "coupled": 5}


def get_plant(name: str) -> Plant:
    try:
        return PLANTS[name]
    except KeyError:
        raise ValueError(f"unknown plant {name!r}; known: {sorted(PLANTS)}") from None


def output(plant: Plant, x):
    """``y = c^T x`` over the last axis (numpy or torch)."""
    if isinstance(x, torch.Tensor):
        return x @ torch.as_tensor(plant.c_vec, dtype=x.dtype)
    return np.asarray(x) @ plant.c_vec


@torch.no_grad()
def simulate(plant: Plant, x0, u, steps: int, h: float = 0.05, keep: int | None = None):
    """Batched fixed-step RK4 in float64 under constant inputs.

    ``x0`` is ``(b, n)``, ``u`` is ``(b,)``.  Returns ``(Y, X, x_end)``: the output at every
    one of the ``steps + 1`` nodes, the states at the first ``keep`` nodes (all of them when
    ``keep`` is ``None``), and the final state.  Keeping only a prefix of the states lets a
    long continuation run without storing it.
    """
    x = torch.as_tensor(x0, dtype=torch.float64).clone()
    uu = torch.as_tensor(u, dtype=torch.float64)
    keep = steps + 1 if keep is None else min(keep, steps + 1)
    f = plant.field
    c = torch.as_tensor(plant.c_vec)
    Y = torch.empty(x.shape[0], steps + 1, dtype=torch.float64)
    X = torch.empty(x.shape[0], keep, plant.n, dtype=torch.float64)
    Y[:, 0] = x @ c
    X[:, 0] = x
    for k in range(steps):
        k1 = f(x, uu)
        k2 = f(x + 0.5 * h * k1, uu)
        k3 = f(x + 0.5 * h * k2, uu)
        k4 = f(x + h * k3, uu)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[:, k + 1] = x @ c
        if k + 1 < keep:
            X[:, k + 1] = x
    return Y.numpy(), X.numpy(), x.numpy()


@dataclass(frozen=True)
class CriticalPoint:
    """An equilibrium of the plant at one input: state, output, Morse-type index, spectrum.

    ``index`` is the number of eigenvalues of ``D_x f`` with positive real part: 0 for a
    sink, 1 for a saddle, 2 for the index-2 point of ``coupled``.
    """

    x: np.ndarray
    y: float
    index: int
    eig: np.ndarray


def _search_starts(plant: Plant) -> torch.Tensor:
    # 5 per axis in 4-D finds every critical point of `coupled` over its scan range.
    per = 25 if plant.n <= 2 else 5
    axes = []
    for lo, hi in plant.box:
        mid, half = 0.5 * (lo + hi), 0.75 * (hi - lo)
        axes.append(np.linspace(mid - half, mid + half, per))
    return torch.tensor(list(itertools.product(*axes)), dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def _critical_points(name: str, u: float) -> tuple:
    plant = get_plant(name)
    f = plant.field
    z = _search_starts(plant)
    uu = torch.full((z.shape[0],), u, dtype=torch.float64)
    jac = torch.func.vmap(torch.func.jacrev(f, argnums=0))
    for _ in range(60):
        J = jac(z, uu)
        step = torch.linalg.lstsq(J, f(z, uu).unsqueeze(-1)).solution.squeeze(-1)
        z = z - torch.nan_to_num(step).clamp(-0.5, 0.5)
    res = f(z, uu).norm(dim=-1)
    roots: list[np.ndarray] = []
    for i in torch.nonzero(torch.isfinite(res) & (res < 1e-10)).flatten().tolist():
        r = z[i].numpy()
        if all(np.linalg.norm(r - s) > 1e-6 for s in roots):
            roots.append(r)
    out = []
    for r in roots:
        J = jac(torch.tensor(r).unsqueeze(0), torch.tensor([u], dtype=torch.float64))[0]
        ev = np.linalg.eigvals(J.numpy())
        out.append(CriticalPoint(r, float(r @ plant.c_vec), int((ev.real > 1e-9).sum()), ev))
    return tuple(sorted(out, key=lambda p: p.y))


def critical_points(plant: Plant, u: float) -> tuple:
    """Every equilibrium of ``plant`` at constant input ``u``, sorted by output.  Cached."""
    return _critical_points(plant.name, float(u))


def sinks(plant: Plant, u: float) -> tuple:
    return tuple(p for p in critical_points(plant, u) if p.index == 0)


def separator_output(plant: Plant, u: float) -> float:
    """Output of the critical point that separates the two sides of the one-side split.

    It is the unique critical point of index ``plant.sep_index`` at ``u``: the saddle of a
    2-D double well, the index-2 point of ``coupled``.  Raises when the plant is monostable
    or when that point is absent or not unique at ``u`` -- the one-side split is undefined
    there.
    """
    if plant.sep_index is None:
        raise ValueError(f"{plant.name} has no separating critical point")
    cand = [p for p in critical_points(plant, u) if p.index == plant.sep_index]
    if len(cand) != 1:
        raise ValueError(f"{plant.name}: {len(cand)} critical points of index "
                         f"{plant.sep_index} at u={u}")
    return cand[0].y
