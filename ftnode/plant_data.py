"""Partial-observation datasets for the plants of :mod:`ftnode.plants`.

The layout is that of :class:`ftnode.systems.DuffingDataset` -- an encoder window ``W`` of
past outputs, a constant input ``U``, the output targets ``Y``, and the true states
``Xfull`` -- so :func:`ftnode.train.train_one` trains on it unchanged.  Three things are
added:

* **Measurement noise.**  ``W`` and ``Y`` may carry i.i.d. Gaussian noise; ``W_clean`` and
  ``Y_clean`` keep the noise-free outputs for scoring.
* **Regions.**  Besides the proposal box, a dataset can come from one side of the plant's
  separating critical point (``upper`` / ``lower``), from outside the box (``far``), or from
  input bands beyond training (``input_near`` / ``input_fold``).  All are drawn by rejection
  sampling in float64.
* **The rest output.**  Every trajectory is continued noise-free to ``L_REST`` steps, and
  ``rest_y`` records the output it settles at.

Seeding is ``numpy.random.SeedSequence([plant, region, role, index, stream, ...])``: no two
draws share a stream, the noise of each level is drawn independently, and the clean
trajectories are identical across noise levels.  ``index`` is the replicate for training
and validation sets and the block for test sets.

:func:`ftnode.systems.make_dataset` and :mod:`ftnode.ood_splits` are not used, so the
frozen notebooks keep their draw order.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, replace

import numpy as np
import torch

from .plants import PLANT_IDS, Plant, critical_points, get_plant, separator_output, simulate

__all__ = [
    "REGIONS",
    "L_REST",
    "PlantDataConfig",
    "PlantDataset",
    "make_clean",
    "add_noise",
    "output_scale",
    "dataset_hash",
]

REGIONS = ("box", "upper", "lower", "far", "input_near", "input_fold")
_REGION_IDS = {r: i for i, r in enumerate(REGIONS)}
_ROLE_IDS = {"train": 0, "val": 1, "test": 2}
_STREAM_SAMPLE, _STREAM_NOISE = 0, 1

#: Length of the noise-free continuation that gives each trajectory's rest output.
L_REST = 4000
_MAX_CRIT = 9


@dataclass(frozen=True)
class PlantDataConfig:
    """One dataset: plant, region, role, replicate or block, size, and horizon."""

    plant: str
    region: str = "box"
    role: str = "train"
    index: int = 0
    n_traj: int = 512
    L: int = 200
    tau: int = 8
    h: float = 0.05

    def __post_init__(self):
        get_plant(self.plant)
        if self.region not in _REGION_IDS:
            raise ValueError(f"unknown region {self.region!r}; known: {REGIONS}")
        if self.role not in _ROLE_IDS:
            raise ValueError(f"unknown role {self.role!r}; known: {sorted(_ROLE_IDS)}")

    def entropy(self, stream: int, *extra: int) -> list:
        return [PLANT_IDS[self.plant], _REGION_IDS[self.region], _ROLE_IDS[self.role],
                self.index, stream, *extra]


@dataclass
class PlantDataset:
    """One split.  ``W`` ``(n, tau)``, ``U`` ``(n,)``, ``Y`` ``(n, L+1)`` are what a model
    sees (noisy when noise was added); ``W_clean`` and ``Y_clean`` are noise-free;
    ``Xfull`` ``(n, L+1, n_state)`` holds the true states aligned with ``Y``; ``rest_y``
    ``(n,)`` is the output at the end of the continuation.  For test sets, ``crit_y`` and
    ``crit_index`` ``(n, 9)`` list the outputs and indices of the plant's critical points at
    each trajectory's input (NaN and -1 padded); they are empty otherwise.  ``accept_rate``
    is the fraction of drawn candidates the region accepted.
    """

    W: torch.Tensor
    U: torch.Tensor
    Y: torch.Tensor
    W_clean: torch.Tensor
    Y_clean: torch.Tensor
    Xfull: torch.Tensor
    rest_y: torch.Tensor
    crit_y: torch.Tensor
    crit_index: torch.Tensor
    accept_rate: float = 1.0

    def to(self, device) -> "PlantDataset":
        """Move what the trainer touches; the scoring-only fields stay on the CPU."""
        return replace(self, W=self.W.to(device), U=self.U.to(device), Y=self.Y.to(device))

    def __len__(self) -> int:
        return self.W.shape[0]


def _uniform_union(rng, intervals, size):
    lens = np.array([hi - lo for lo, hi in intervals])
    pick = rng.choice(len(intervals), size=size, p=lens / lens.sum())
    lo = np.array([intervals[i][0] for i in pick])
    return lo + rng.uniform(0.0, 1.0, size) * lens[pick]


def _draw(plant: Plant, region: str, rng, size):
    lo = np.array([b[0] for b in plant.box])
    hi = np.array([b[1] for b in plant.box])
    x0 = lo + rng.uniform(0.0, 1.0, (size, plant.n)) * (hi - lo)
    if region == "far":
        coord = rng.choice(np.asarray(plant.far_coords), size=size)
        mag = rng.uniform(*plant.far_band, size)
        sign = rng.choice([-1.0, 1.0], size=size) if plant.far_signed else np.ones(size)
        x0[np.arange(size), coord] = sign * mag
    if region == "input_near":
        u = _uniform_union(rng, plant.u_near, size)
    elif region == "input_fold":
        u = _uniform_union(rng, plant.u_fold, size)
    else:
        u = rng.uniform(*plant.u_train, size)
    return x0, u


@functools.lru_cache(maxsize=None)
def _separator_table(name: str):
    plant = get_plant(name)
    grid = np.linspace(*plant.u_train, 21)
    return grid, np.array([separator_output(plant, u) for u in grid])


def _accept(plant: Plant, region: str, Y, u, n_keep):
    if region not in ("upper", "lower"):
        return np.ones(len(u), dtype=bool)
    grid, ysep = _separator_table(plant.name)
    sep = np.interp(u, grid, ysep)[:, None]
    head, rest = Y[:, :n_keep], Y[:, -1:]
    if region == "upper":
        return np.all(head > sep, axis=1) & (rest[:, 0] > sep[:, 0])
    return np.all(head < sep, axis=1) & (rest[:, 0] < sep[:, 0])


def _crit_table(plant: Plant, U):
    cy = np.full((len(U), _MAX_CRIT), np.nan)
    ci = np.full((len(U), _MAX_CRIT), -1, dtype=np.int64)
    for i, u in enumerate(U):
        cps = critical_points(plant, float(u))
        cy[i, : len(cps)] = [p.y for p in cps]
        ci[i, : len(cps)] = [p.index for p in cps]
    return torch.from_numpy(cy), torch.from_numpy(ci)


def make_clean(cfg: PlantDataConfig, max_rounds: int = 200) -> PlantDataset:
    """Draw ``cfg.n_traj`` accepted trajectories, noise-free.

    Candidates are drawn in batches, simulated over the window, the horizon, and the
    continuation, and filtered by the region's acceptance rule; accepted trajectories keep
    their draw order, so a configuration always yields the same dataset.
    """
    plant = get_plant(cfg.plant)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.entropy(_STREAM_SAMPLE)))
    n_keep = cfg.tau + cfg.L + 1
    steps = cfg.tau + max(cfg.L, L_REST)
    parts, got, drawn, hits = [], 0, 0, 0
    for _ in range(max_rounds):
        size = max(256, 4 * (cfg.n_traj - got))
        x0, u = _draw(plant, cfg.region, rng, size)
        Y, X, _ = simulate(plant, x0, u, steps, cfg.h, keep=n_keep)
        ok = _accept(plant, cfg.region, Y, u, n_keep)
        drawn, hits = drawn + size, hits + int(ok.sum())
        idx = np.nonzero(ok)[0][: cfg.n_traj - got]
        parts.append((Y[idx], X[idx], u[idx]))
        got += len(idx)
        if got == cfg.n_traj:
            break
    else:
        raise RuntimeError(f"{cfg}: only {got} of {cfg.n_traj} trajectories accepted")
    Y = np.concatenate([p[0] for p in parts])
    X = np.concatenate([p[1] for p in parts])
    U = np.concatenate([p[2] for p in parts])
    tau = cfg.tau
    W = torch.from_numpy(Y[:, :tau].astype(np.float32))
    Yt = torch.from_numpy(Y[:, tau:n_keep].astype(np.float32))
    if cfg.role == "test":
        crit_y, crit_index = _crit_table(plant, U)
    else:
        crit_y, crit_index = torch.empty(0), torch.empty(0, dtype=torch.int64)
    return PlantDataset(
        W=W, U=torch.from_numpy(U.astype(np.float32)), Y=Yt, W_clean=W.clone(),
        Y_clean=Yt.clone(), Xfull=torch.from_numpy(X[:, tau:].astype(np.float32)),
        rest_y=torch.from_numpy(Y[:, -1].astype(np.float32)), crit_y=crit_y,
        crit_index=crit_index, accept_rate=hits / drawn,
    )


def add_noise(ds: PlantDataset, cfg: PlantDataConfig, r: float, s_y: float) -> PlantDataset:
    """``ds`` with ``N(0, (r s_y)^2)`` noise on ``W`` and ``Y``; the clean fields unchanged.

    The draw depends on the dataset and on the level ``r`` only, so every head and every
    model seed sees the same noise, and different levels are independent.
    """
    if r == 0.0:
        return ds
    level = int(round(r * 10000))
    rng = np.random.default_rng(np.random.SeedSequence(cfg.entropy(_STREAM_NOISE, level)))
    sigma = r * s_y
    nw = rng.standard_normal(tuple(ds.W_clean.shape)).astype(np.float32) * sigma
    ny = rng.standard_normal(tuple(ds.Y_clean.shape)).astype(np.float32) * sigma
    return replace(ds, W=ds.W_clean + torch.from_numpy(nw), Y=ds.Y_clean + torch.from_numpy(ny))


@functools.lru_cache(maxsize=None)
def output_scale(plant: str) -> float:
    """``s_y``: the standard deviation of the clean outputs of the plant's T0 training set
    (replicate 0, window and targets together).  The noise levels are relative to it."""
    ds = make_clean(PlantDataConfig(plant))
    return float(torch.cat([ds.W_clean, ds.Y_clean], dim=1).std())


def dataset_hash(ds: PlantDataset) -> str:
    """SHA-256 over every tensor of ``ds``, in field order (first 16 hex digits)."""
    h = hashlib.sha256()
    for t in (ds.W, ds.U, ds.Y, ds.W_clean, ds.Y_clean, ds.Xfull, ds.rest_y, ds.crit_y,
              ds.crit_index):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
