"""Pretraining systems for the context-conditioned field: a damped mechanical family.

Every system is a one-degree-of-freedom oscillator with a polynomial potential and a
nonnegative damping coefficient,

    q'' = -U'(q) - r(q) q' + u,     U(q) = a4 q^4/4 + a3 q^3/3 + a2 q^2/2,
                                    r(q) = d0 + d1 q^2 >= 0,

with ``u`` constant over a trajectory.  With ``a4 > 0`` (or ``a4 = a3 = 0`` and ``a2 > 0``)
the potential is coercive and the damping is positive, so every trajectory settles to an
equilibrium: the family sits inside the steady-state target class.  Duffing is the member
``(a4, a3, a2, d0, d1) = (1, 0, -1, 0.2, 0)``; it and a neighbourhood of it are held out
(:data:`DUFFING_HOLDOUT`), so a Duffing evaluation tests identification of an unseen member.

**Reproducibility.**  Every random draw of a batch -- the system parameters, the initial
states, the inputs, the context sizes, the token and query subsamples -- comes from one CPU
``torch.Generator`` seeded by ``(seed, stream, index)`` through :func:`batch_generator`.  So
any batch is a pure function of those three integers and the config, independent of the
device and of every batch drawn before it.  The simulation itself is deterministic RK4; it
runs on the requested device, so only floating-point rounding can differ between devices.

Only ``q`` is "measured".  Context tokens are built from ``q`` alone with the same central
differences the Duffing pipeline uses, so pretraining and evaluation see identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "MechanicalFamilyConfig",
    "DUFFING_PARAMS",
    "DUFFING_HOLDOUT",
    "in_duffing_holdout",
    "batch_generator",
    "sample_params",
    "mechanical_field",
    "simulate_q",
    "context_tokens",
    "SystemBatch",
    "sample_batch",
    "TRAIN_STREAM",
    "VAL_STREAM",
]

#: Stream ids for :func:`batch_generator`: training batches and the fixed validation set
#: draw from disjoint streams of the same seed.
TRAIN_STREAM = 0
VAL_STREAM = 1

#: Duffing in this family's coordinates: ``(a4, a3, a2, d0, d1)``.
DUFFING_PARAMS = (1.0, 0.0, -1.0, 0.2, 0.0)

#: The held-out neighbourhood of Duffing.  A draw is rejected when **all** hold:
#: ``|a4 - 1| < 0.25``, ``|a3| < 0.15``, ``|a2 + 1| < 0.25``, ``|d0 - 0.2| < 0.06``,
#: ``d1 < 0.05``.  Draws outside it may still be double wells with wells near +-1, but
#: never this plant's combination of curvature and damping.
DUFFING_HOLDOUT = {"a4": 0.25, "a3": 0.15, "a2": 0.25, "d0": 0.06, "d1": 0.05}


def in_duffing_holdout(a4, a3, a2, d0, d1):
    """Boolean mask: which parameter draws fall inside :data:`DUFFING_HOLDOUT`."""
    H = DUFFING_HOLDOUT
    return (
        ((a4 - 1.0).abs() < H["a4"])
        & (a3.abs() < H["a3"])
        & ((a2 + 1.0).abs() < H["a2"])
        & ((d0 - 0.2).abs() < H["d0"])
        & (d1 < H["d1"])
    )


@dataclass(frozen=True)
class MechanicalFamilyConfig:
    """The system family, the observation grid, and the per-batch sampling sizes.

    Kinds (probabilities ``p_double``, ``p_single``, the rest linear):

    - double well: well position ``w ~ U[w_lo, w_hi]``, ``a4 ~ logU[a4_lo, a4_hi]``,
      ``a2 = -a4 w^2``, ``a3 = a4 w s`` with the asymmetry ``s ~ U[-asym, asym]``;
    - single well: ``a2 ~ logU[a2_lo, a2_hi]``, ``a4 ~ U[0, a4_hi]``, ``a3 = 0``;
    - linear: ``a2 ~ logU[a2_lo, a2_hi]``, ``a4 = a3 = 0``.

    Damping ``d0 ~ logU[d0_lo, d0_hi]``; with probability ``p_d1`` also
    ``d1 ~ U[0, d1_hi]``, else ``d1 = 0``.

    Each trajectory starts uniformly in ``[-x_range, x_range] x [-v_range, v_range]`` (the
    standard Duffing box) with ``u ~ U[-u_range, u_range]``.  A trajectory has ``tau + L + 1``
    samples at step ``h`` (the Duffing training length).  Contexts hold ``K ~ U{k_min..k_max}``
    trajectories, of which ``n_tok`` tokens are drawn.  Queries: ``n_query`` per system, half
    at true trajectory states, half uniform in the query box ``[-q_box, q_box] x
    [-v_box, v_box]`` with their own random ``u``.
    """

    h: float = 0.05
    tau: int = 8
    L: int = 200
    u_range: float = 0.25
    x_range: float = 1.6
    v_range: float = 1.0
    p_double: float = 0.5
    p_single: float = 0.35
    w_lo: float = 0.6
    w_hi: float = 1.5
    a4_lo: float = 0.3
    a4_hi: float = 2.0
    asym: float = 0.3
    a2_lo: float = 0.25
    a2_hi: float = 4.0
    d0_lo: float = 0.03
    d0_hi: float = 1.0
    p_d1: float = 0.3
    d1_hi: float = 0.5
    k_min: int = 4
    k_max: int = 64
    n_tok: int = 1024
    n_query: int = 256
    q_box: float = 2.0
    v_box: float = 1.6
    max_resample: int = 50

    @property
    def T(self) -> int:
        """Samples per trajectory."""
        return self.tau + self.L + 1


def batch_generator(seed: int, stream: int, index: int) -> torch.Generator:
    """The CPU generator for batch ``index`` of ``stream`` under ``seed``.

    ``numpy.random.SeedSequence`` mixes the three integers, so neighbouring indices give
    unrelated streams and the mapping is stable across platforms and versions.
    """
    state = np.random.SeedSequence([int(seed), int(stream), int(index)]).generate_state(2)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(state[0]) << 32 | int(state[1]))
    return g


def _u(gen, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64)


def _logu(gen, n, lo, hi):
    return torch.exp(_u(gen, n, np.log(lo), np.log(hi)))


def sample_params(cfg: MechanicalFamilyConfig, n: int, gen: torch.Generator) -> torch.Tensor:
    """``n`` parameter draws ``(a4, a3, a2, d0, d1)``, shape ``(n, 5)``, float64, on CPU.

    Draws inside :data:`DUFFING_HOLDOUT` are redrawn (up to ``max_resample`` rounds), so
    the result never contains Duffing's neighbourhood.  The redraws come from the same
    generator, so the output stays a pure function of the generator's seed.
    """
    out = torch.empty(n, 5, dtype=torch.float64)
    todo = torch.arange(n)
    for _ in range(cfg.max_resample):
        m = len(todo)
        kind = torch.rand(m, generator=gen, dtype=torch.float64)
        double = kind < cfg.p_double
        single = (kind >= cfg.p_double) & (kind < cfg.p_double + cfg.p_single)
        w = _u(gen, m, cfg.w_lo, cfg.w_hi)
        a4_dbl = _logu(gen, m, cfg.a4_lo, cfg.a4_hi)
        s = _u(gen, m, -cfg.asym, cfg.asym)
        a2_pos = _logu(gen, m, cfg.a2_lo, cfg.a2_hi)
        a4_sgl = _u(gen, m, 0.0, cfg.a4_hi)
        d0 = _logu(gen, m, cfg.d0_lo, cfg.d0_hi)
        has_d1 = torch.rand(m, generator=gen, dtype=torch.float64) < cfg.p_d1
        d1 = torch.where(has_d1, _u(gen, m, 0.0, cfg.d1_hi), torch.zeros(m, dtype=torch.float64))
        a4 = torch.where(double, a4_dbl, torch.where(single, a4_sgl, torch.zeros_like(a4_sgl)))
        a2 = torch.where(double, -a4_dbl * w**2, a2_pos)
        a3 = torch.where(double, a4_dbl * w * s, torch.zeros_like(s))
        p = torch.stack([a4, a3, a2, d0, d1], dim=-1)
        out[todo] = p
        bad = in_duffing_holdout(a4, a3, a2, d0, d1)
        todo = todo[bad]
        if len(todo) == 0:
            return out
    raise RuntimeError("could not leave the Duffing hold-out region; check DUFFING_HOLDOUT")


def mechanical_field(x, u, p):
    """``(q', v')`` of the family.  ``x`` is ``(..., 2)``, ``u`` is ``(...)``, ``p`` is ``(..., 5)``."""
    q, v = x[..., 0], x[..., 1]
    a4, a3, a2, d0, d1 = p.unbind(-1)
    dU = a4 * q**3 + a3 * q**2 + a2 * q
    return torch.stack([v, -dU - (d0 + d1 * q**2) * v + u], dim=-1)


def simulate_q(p, x0, u, h, T):
    """RK4 trajectories of the family.  Returns the full state ``(..., T, 2)``.

    ``p`` is ``(..., 5)``, ``x0`` is ``(..., 2)``, ``u`` is ``(...)``, all on one device.
    """
    xs = [x0]
    x = x0
    for _ in range(T - 1):
        k1 = mechanical_field(x, u, p)
        k2 = mechanical_field(x + 0.5 * h * k1, u, p)
        k3 = mechanical_field(x + 0.5 * h * k2, u, p)
        k4 = mechanical_field(x + h * k3, u, p)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs, dim=-2)


#: Fixed feature scales for the context tokens ``(q, q_dot_hat, q_ddot_hat, u)``.  Constants,
#: not data statistics, so the model works in raw coordinates (no per-context normalization).
TOKEN_SCALE = (1.5, 1.5, 5.0, 0.25)


def context_tokens(q, u, h):
    """Tokens ``(q_i, q_dot_hat_i, q_ddot_hat_i, u)`` from measured ``q`` only.

    ``q`` is ``(..., T)`` and ``u`` is ``(...)``.  Central differences at the interior samples
    ``i = 1 .. T-2``; returns ``(..., T-2, 4)``, scaled by :data:`TOKEN_SCALE`.
    """
    qd = (q[..., 2:] - q[..., :-2]) / (2.0 * h)
    qdd = (q[..., 2:] - 2.0 * q[..., 1:-1] + q[..., :-2]) / h**2
    uu = u.unsqueeze(-1).expand_as(qd)
    tok = torch.stack([q[..., 1:-1], qd, qdd, uu], dim=-1)
    return tok / tok.new_tensor(TOKEN_SCALE)


@dataclass
class SystemBatch:
    """One batch of ``B`` systems.

    - ``params`` ``(B, 5)``: ``(a4, a3, a2, d0, d1)``.
    - ``tokens`` ``(B, n_tok, 4)``: the context, from measured ``q`` only.
    - ``k`` ``(B,)``: how many trajectories each context holds.
    - ``xq`` ``(B, n_query, 2)``, ``uq`` ``(B, n_query)``: query states and inputs.
    - ``fq`` ``(B, n_query, 2)``: the exact field at the queries.
    - ``on_traj`` ``(n_query,)``: which query columns are trajectory states.
    """

    params: torch.Tensor
    tokens: torch.Tensor
    k: torch.Tensor
    xq: torch.Tensor
    uq: torch.Tensor
    fq: torch.Tensor
    on_traj: torch.Tensor = field(default=None)


def sample_batch(cfg: MechanicalFamilyConfig, B: int, seed: int, stream: int, index: int,
                 device="cpu", dtype=torch.float32) -> SystemBatch:
    """Draw, simulate and tokenize ``B`` systems -- a pure function of ``(seed, stream, index)``.

    All random draws happen on the CPU generator first, in a fixed order; only then does the
    simulation run on ``device``.  Simulation is in float64 for accuracy, and the outputs are
    cast to ``dtype``.
    """
    gen = batch_generator(seed, stream, index)
    K, T, h = cfg.k_max, cfg.T, cfg.h
    p = sample_params(cfg, B, gen)                                            # (B, 5)
    q0 = _u(gen, B * K, -cfg.x_range, cfg.x_range).view(B, K)
    v0 = _u(gen, B * K, -cfg.v_range, cfg.v_range).view(B, K)
    u = _u(gen, B * K, -cfg.u_range, cfg.u_range).view(B, K)
    k = torch.randint(cfg.k_min, cfg.k_max + 1, (B,), generator=gen)
    n_tok_avail = T - 2
    # Token draws: trajectory index below k[b], time index over the interior samples.
    tok_traj = (torch.rand(B, cfg.n_tok, generator=gen, dtype=torch.float64) * k.unsqueeze(1)).long()
    tok_time = torch.randint(0, n_tok_avail, (B, cfg.n_tok), generator=gen)
    n_on = cfg.n_query // 2
    n_box = cfg.n_query - n_on
    qry_traj = (torch.rand(B, n_on, generator=gen, dtype=torch.float64) * k.unsqueeze(1)).long()
    qry_time = torch.randint(0, T, (B, n_on), generator=gen)
    box_q = _u(gen, B * n_box, -cfg.q_box, cfg.q_box).view(B, n_box)
    box_v = _u(gen, B * n_box, -cfg.v_box, cfg.v_box).view(B, n_box)
    box_u = _u(gen, B * n_box, -cfg.u_range, cfg.u_range).view(B, n_box)

    # Deterministic part, on the device.
    dev = torch.device(device)
    p_d = p.to(dev)
    X = simulate_q(p_d.unsqueeze(1).expand(B, K, 5), torch.stack([q0, v0], -1).to(dev), u.to(dev), h, T)
    q = X[..., 0]                                                             # (B, K, T)
    tok_all = context_tokens(q, u.to(dev), h)                                 # (B, K, T-2, 4)
    bi = torch.arange(B, device=dev).unsqueeze(1)
    tokens = tok_all[bi, tok_traj.to(dev), tok_time.to(dev)]                  # (B, n_tok, 4)
    x_on = X[bi, qry_traj.to(dev), qry_time.to(dev)]                          # (B, n_on, 2)
    u_on = u.to(dev)[bi, qry_traj.to(dev)]                                    # (B, n_on)
    x_box = torch.stack([box_q, box_v], -1).to(dev)
    xq = torch.cat([x_on, x_box], dim=1)
    uq = torch.cat([u_on, box_u.to(dev)], dim=1)
    fq = mechanical_field(xq, uq, p_d.unsqueeze(1))
    on_traj = torch.cat([torch.ones(n_on, dtype=torch.bool), torch.zeros(n_box, dtype=torch.bool)]).to(dev)
    return SystemBatch(
        params=p_d.to(dtype), tokens=tokens.to(dtype), k=k.to(dev),
        xq=xq.to(dtype), uq=uq.to(dtype), fq=fq.to(dtype), on_traj=on_traj,
    )
