"""The trainable control splitting operator, and the control-stage config.

The identified field ``F_theta`` admits infinitely many splittings

    F_theta(z, u) = f(z) @ (z - g(z, u)),

because for any invertible ``f`` the choice ``g = z - f^{-1} F_theta`` reproduces
``F_theta`` exactly.  That gauge freedom is what the control stage exploits: the
plant is fixed by identification, and only the *representation* is redesigned.
:class:`SplitOperator` is the sole trainable object in that stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..latent import MLP, KappaBudget, spectral_clamp_safe

__all__ = ["SplitOperator", "ControlConfig"]


class SplitOperator(nn.Module):
    """Redesigned control splitting ``f_psi(z)``, in the same kappa-bounded class as the ID operator.

    ``f = -(sigma_min I + P) + K`` with ``||P||_2 <= c_P`` and ``||K||_2 <= c_K``,
    so ``sym f <= -sigma_min I``, ``||f||_2 <= sigma_min + c_P + c_K = sigma_max``
    and ``kappa <= sigma_max / sigma_min`` -- all **by construction, not by
    penalty**, and therefore preserved exactly at every point of training.

    Structurally identical to :class:`~ftnode.latent.LatentFTNODEClamp`'s ``A(z)``,
    with two differences: it *is* the operator (no ``g``, no ``F`` -- those come
    from the gauge relation in :mod:`ftnode.control.policy`), and it projects with
    :func:`~ftnode.latent.spectral_clamp_safe`, because under psi-training the
    SVD in the original clamp fails to converge on the skew argument.

    Attribute names ``L_net``/``S_net``/``_eye`` are fixed by the committed
    ``best-ctrl-psi-seed0.pth`` checkpoint.
    """

    def __init__(self, c_P, c_K, m=4, hidden=64, depth=3, sigma_min=0.1, activation="silu"):
        super().__init__()
        self.m, self.sigma_min = m, sigma_min
        self.c_P, self.c_K = float(c_P), float(c_K)
        self.activation = activation
        self.L_net = MLP(m, m * m, hidden, depth, activation=activation)
        self.S_net = MLP(m, m * m, hidden, depth, last_zero=True, activation=activation)
        self.register_buffer("_eye", torch.eye(m))

    @classmethod
    def from_budget(
        cls, budget: KappaBudget, hidden=64, depth=3, activation="silu"
    ) -> "SplitOperator":
        """Build from a :class:`~ftnode.latent.KappaBudget`, matching the ID operator's class."""
        return cls(
            c_P=budget.c_P,
            c_K=budget.c_K,
            m=budget.m,
            hidden=hidden,
            depth=depth,
            sigma_min=budget.sigma_min,
            activation=activation,
        )

    def forward(self, z):
        Lc = spectral_clamp_safe(self.L_net(z).view(-1, self.m, self.m), self.c_P**0.5)
        P = Lc @ Lc.transpose(1, 2)
        Mr = self.S_net(z).view(-1, self.m, self.m)
        K = spectral_clamp_safe(Mr - Mr.transpose(1, 2), self.c_K)
        return -(self.sigma_min * self._eye + P) + K

    @property
    def sigma_max(self) -> float:
        return self.sigma_min + self.c_P + self.c_K

    @property
    def kappa_bound(self) -> float:
        return self.sigma_max / self.sigma_min


@dataclass(frozen=True)
class ControlConfig:
    """Control-stage settings.

    Defaults are the values ``duffing_learned_splitting_control.ipynb`` settled
    on, each for a documented reason:

    ``u_bound = 0.5``
        The admissible input set is ``[-u_bound, u_bound]``.  Deliberately
        **wider** than the identification excitation range (``0.25``): the
        notebook's reachability study finds a saturated LQR at the saddle
        stabilizes only 19 of 39 initial conditions at ``|u| <= 0.25`` but all 39
        at ``|u| <= 0.5``.  0.25 is not enough authority for the task.
    ``k_trunc = 10``
        Truncated-BPTT window.  Backpropagating through the full 120-step loop
        produces gradient norms spiking to 1e17-1e19, which poison Adam's moments
        beyond rescue (``clip_grad_norm_`` cannot clip an ``inf``).  At
        ``k_trunc=10`` the max gradient norm stays around 7e2.
    ``z0_spread = 0.35``
        Initial conditions drawn Gaussian around ``z*`` rather than uniform over
        the latent box: this trains for regulation, not global capture.
    """

    u_star: float = 0.0
    q_star: float = 0.0
    u_bound: float = 0.5
    eta: float = 20.0
    k_trunc: int = 10
    T_list: tuple = (120,)
    n_epochs: int = 40
    n_ic: int = 48
    batch: int = 24
    lr: float = 2e-3
    R_u: float = 1e-2
    lam_g: float = 1.0
    r_in_target: float = 1.0
    z0_spread: float | None = 0.35
    h: float = 0.05
    z_scale: float = 2.0

    @property
    def u_lo(self) -> float:
        return -self.u_bound

    @property
    def u_hi(self) -> float:
        return self.u_bound
