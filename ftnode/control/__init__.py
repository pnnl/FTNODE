"""Learned-splitting control for latent FT-NODE models.

Promoted from ``examples/duffing/duffing_learned_splitting_control.ipynb`` and the
``_proto_*.py`` prototypes beside it.

The idea: identification fixes the plant as a latent field ``F_theta``, but the
splitting ``F_theta = f(z)(z - g(z, u))`` is not unique -- for any invertible
``f``, setting ``g = z - f^{-1} F_theta`` reproduces ``F_theta`` exactly.  The
control stage freezes ``F_theta`` and trains only that gauge choice, so the
represented plant provably cannot move while the control cost landscape is
reshaped.

Typical use::

    from ftnode.control import (ControlConfig, FrozenLatentPlant, SplitOperator,
                                closed_loop, latent_target, train_psi)
    from ftnode.latent import KappaBudget

    budget = KappaBudget()
    plant  = FrozenLatentPlant.from_checkpoint('best-ctrl-id-svdclamp-seed0.pth')
    cfg    = ControlConfig()

    z_star, res, q = latent_target(plant, cfg.u_star, cfg.q_star, tau=8)
    fpsi = SplitOperator.from_budget(budget)
    fpsi, hist = train_psi(fpsi, plant, z_star, cfg.u_star, cfg)
    out = closed_loop(fpsi, plant, x0, w0, z_star, T, cfg.h, cfg.eta, cfg.u_lo, cfg.u_hi)

Unlike the prototype scripts, importing this package has no side effects: it
loads no checkpoint, generates no data and seeds no RNG.
"""

from .operator import ControlConfig, SplitOperator
from .plant import FrozenLatentPlant
from .policy import (
    closed_loop,
    cost_T,
    g_psi,
    g_range_penalty,
    grad_u_J,
    model_closed_loop,
    sat_u,
    udot_dir,
)
from .target import latent_lqr, latent_target, latent_target_encode, lqr_closed_loop_true
from .train import train_psi

__all__ = [
    "ControlConfig",
    "FrozenLatentPlant",
    "SplitOperator",
    "closed_loop",
    "cost_T",
    "g_psi",
    "g_range_penalty",
    "grad_u_J",
    "latent_lqr",
    "latent_target",
    "latent_target_encode",
    "lqr_closed_loop_true",
    "model_closed_loop",
    "sat_u",
    "train_psi",
    "udot_dir",
]
