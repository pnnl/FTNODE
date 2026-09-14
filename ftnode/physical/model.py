"""The physical-state field, its two parameterizations, and the container.

``x = (q, q_dot)`` is the physical state.  ``u`` is the scalar input, held constant
over a trajectory.  The measured output is ``q = x[0]``.

Two field parameterizations, matching the 2x2 factorial's structure axis:

- :class:`UnstructuredField` -- a free field ``F(x,u) = MLP([x,u])``.
- :class:`StructuredField` -- ``F(x,u) = A(x)(x - g(x,u))`` with ``A`` negative
  definite by construction, composed from the generic ``ClampOperator`` and
  ``BoundedTanhG``.

:class:`PhysicalField` ties a field to the finite-difference initial-state estimate
and the ``q = x[0]`` readout.  It is the observed-state analogue of
``ftnode.latent.LatentSysID``, with no encoder and no decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..latent.equilibrium import BoundedTanhG
from ..latent.nets import MLP
from ..latent.operator import ClampOperator, KappaBudget, spectral_clamp

__all__ = [
    "estimate_x0",
    "UnstructuredField",
    "StructuredField",
    "PhysicalField",
]


def estimate_x0(window, h):
    """Finite-difference estimate of ``x0 = (q0, q_dot0)`` at the first rollout time.

    ``window`` is ``(..., tau)`` of past ``q`` samples at times ``0 .. (tau-1)h``.
    The first rollout target is one step past the window, at time ``tau*h``, so the
    estimate extrapolates one step rather than reading the last sample:

    - ``q_dot0`` is the backward difference at the window end.
    - ``q0`` is a one-step linear extrapolation, ``q0 = q[-1] + h*q_dot0``.

    The estimate is parameter-free and identical for every cell, so initial-state
    error is never a confound.  Requires ``tau >= 2``.

    Returns ``(..., 2)``.
    """
    if window.shape[-1] < 2:
        raise ValueError(f"estimate_x0 needs a window of length >= 2; got {window.shape[-1]}")
    q_last = window[..., -1]
    q_prev = window[..., -2]
    qd0 = (q_last - q_prev) / h
    q0 = q_last + h * qd0
    return torch.stack([q0, qd0], dim=-1)


class UnstructuredField(nn.Module):
    """Free field ``F(x,u) = MLP([x,u]) -> R^m``, no structural prior.

    ``hidden=95, depth=4`` matches the ``ftnode.latent.LatentNODE`` sizing, which was
    itself chosen to match a ``ClampOperator`` model's parameter count, so the cold
    row's unstructured and structured cells compare structure rather than capacity.
    """

    def __init__(self, m=2, q=1, hidden=95, depth=4, activation="silu"):
        super().__init__()
        self.m, self.q = m, q
        self.f_net = MLP(m + q, m, hidden=hidden, depth=depth, activation=activation)

    def F(self, x, u):
        if u.dim() == x.dim() - 1:
            u = u.unsqueeze(-1)
        return self.f_net(torch.cat([x, u], dim=-1))


class StructuredField(nn.Module):
    """Structured field ``F(x,u) = A(x)(x - g(x,u))``, ``A`` negative definite.

    Composes the generic ``ClampOperator`` (for ``A``) and ``BoundedTanhG`` (for
    ``g``).  ``A = -(sigma_min I + P) + K`` with ``P`` symmetric PSD and ``K`` skew,
    so ``sym(A) <= -sigma_min I`` by construction at every ``x``.

    The equilibrium map ``g`` is built **before** the operator ``A``, matching the
    construction order the latent builders use, so a seed reproduces the same draws.
    """

    def __init__(self, m=2, q=1, hidden=64, depth=3, sigma_min=0.1, R_g=2.0,
                 activation="silu", budget=None, clamp_fn=spectral_clamp):
        super().__init__()
        self.m, self.q = m, q
        # g first, then A -- see the class docstring.
        self.equilibrium = BoundedTanhG(m, q, hidden, depth, R_g, activation)
        self.operator = ClampOperator(
            m, sigma_min, hidden, depth, activation, budget, clamp_fn=clamp_fn
        )

    def A(self, x):
        """The operator ``A(x)``, shape ``(b, m, m)``."""
        return self.operator(x)

    def g(self, x, u):
        """The equilibrium map ``g(x, u)``, shape ``(..., m)``."""
        return self.equilibrium(x, u)

    def F(self, x, u):
        return torch.einsum("bij,bj->bi", self.A(x), x - self.g(x, u))


class PhysicalField(nn.Module):
    """Container: a field, the finite-difference initial-state estimate, and the readout.

    The observed-state analogue of ``ftnode.latent.LatentSysID``.  It exposes the
    three things the training rollout needs -- ``estimate_x0``, ``F``, ``readout`` --
    and holds the step ``h`` the estimate uses.  ``field`` is any module exposing
    ``F(x, u)``; a :class:`StructuredField` additionally exposes ``A`` and ``g``.
    """

    def __init__(self, field, h):
        super().__init__()
        self.field = field
        self.h = h

    def estimate_x0(self, window):
        return estimate_x0(window, self.h)

    def prepare(self, window):
        """Bind any per-trajectory context the field needs before a rollout.

        A no-op for the cold cells (their field has no context); the FIM cells encode
        their context from the window here.
        """
        if hasattr(self.field, "prepare"):
            self.field.prepare(window, self.h)

    def F(self, x, u):
        return self.field.F(x, u)

    def param_groups(self, lr, lr_backbone=None):
        """Optimizer parameter groups, delegating to the field when it defines them.

        The FIM cells give their backbone a low ``lr_backbone``; the cold cells put
        every parameter in one group at ``lr``.
        """
        if hasattr(self.field, "param_groups"):
            return self.field.param_groups(lr, lr_backbone)
        return [{"params": list(self.parameters()), "lr": lr}]

    @staticmethod
    def readout(x):
        """The measured output ``q = x[0]``.  No parameters."""
        return x[..., 0]
