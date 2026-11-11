"""Field heads conditioned on a system code ``c``.

- :class:`StructuredCodeHead` -- today's tanh head, ``F = A(x; c)(x - g(x, u; c))``.  It is
  the factorial's cell-4 head with the FIM drift feature replaced by the code: the networks
  that build ``P`` and ``K`` read ``[x, c]``, and the ``g`` network reads ``[x, c, u]``.  The
  bounds hold for **every** ``c``: ``sym A <= -sigma_min I``, ``||A||_2 <= sigma_max`` and
  ``g in [-R_g, R_g]^d``.  So no encoder output can make a rollout at a fixed code blow up.
- :class:`FreeCodeHead` -- the matched unstructured control, ``F = MLP([x, u, c])``, with its
  width chosen so its parameter count matches the structured head's.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..latent.nets import MLP
from ..latent.operator import KappaBudget, spectral_clamp_safe

__all__ = ["StructuredCodeHead", "FreeCodeHead", "mlp_numel", "matched_free_hidden", "spectral_clamp_2x2"]


def mlp_numel(in_dim, out_dim, hidden, depth):
    """Parameter count of ``ftnode.latent.nets.MLP(in_dim, out_dim, hidden, depth)``."""
    return in_dim * hidden + hidden + (depth - 1) * (hidden * hidden + hidden) + hidden * out_dim + out_dim


def _broadcast_code(c, x):
    """Expand a code to the leading shape of ``x`` ``(..., d)``.

    ``c`` must have the same rank as ``x``, with size 1 wherever it is shared: ``(1, k)``
    against ``x`` ``(n, d)`` in a rollout, ``(B, 1, k)`` against ``(B, n_query, d)`` in training.
    """
    if c.dim() != x.dim():
        raise ValueError(f"code rank {c.dim()} must match state rank {x.dim()}; add singleton dims")
    return c.expand(*x.shape[:-1], c.shape[-1])


class StructuredCodeHead(nn.Module):
    """``F(x, u; c) = A(x; c)(x - g(x, u; c))`` with the clamp of ``ClampOperator``.

    ``A = -(sigma_min I + P) + K``, ``P = L L^T`` with ``||L||_2^2 <= c_P`` and ``K`` skew with
    ``||K||_2 <= c_K`` (closed-form 2x2 spectral clamp; eigvalsh for other ``d``), ``g = R_g tanh(.)``.  ``g`` is built before
    ``A``, the order the other structured fields use.
    """

    def __init__(self, d=2, q=1, code_dim=64, hidden=64, depth=3, sigma_min=0.1, kappa_max=25.0,
                 skew_frac=0.6, R_g=2.0, activation="silu"):
        super().__init__()
        self.d, self.q, self.code_dim = d, q, code_dim
        self.sigma_min, self.R_g = sigma_min, R_g
        budget = KappaBudget(sigma_min, kappa_max, skew_frac, d)
        self.c_P, self.c_K = float(budget.c_P), float(budget.c_K)
        self.sigma_max = float(budget.sigma_max)
        feat = d + code_dim
        self.g_net = MLP(feat + q, d, hidden, depth, last_zero=True, activation=activation)
        self.L_net = MLP(feat, d * d, hidden, depth, activation=activation)
        self.S_net = MLP(feat, d * d, hidden, depth, last_zero=True, activation=activation)
        self.register_buffer("_eye", torch.eye(d))

    def A(self, x, c):
        feat = torch.cat([x, _broadcast_code(c, x)], dim=-1)
        lead = feat.shape[:-1]
        clamp = spectral_clamp_2x2 if self.d == 2 else spectral_clamp_safe
        Lc = clamp(self.L_net(feat).reshape(*lead, self.d, self.d), self.c_P**0.5)
        P = Lc @ Lc.transpose(-1, -2)
        Mr = self.S_net(feat).reshape(*lead, self.d, self.d)
        K = clamp(Mr - Mr.transpose(-1, -2), self.c_K)
        return -(self.sigma_min * self._eye + P) + K

    def g(self, x, u, c):
        if u.dim() == x.dim() - 1:
            u = u.unsqueeze(-1)
        feat = torch.cat([x, _broadcast_code(c, x), u], dim=-1)
        return self.R_g * torch.tanh(self.g_net(feat))

    def forward(self, x, u, c):
        A = self.A(x, c)
        return (A @ (x - self.g(x, u, c)).unsqueeze(-1)).squeeze(-1)

    def numel(self) -> int:
        return sum(p.numel() for p in self.parameters())


def spectral_clamp_2x2(B, cap, eps=1e-12):
    """Scale each ``(..., 2, 2)`` matrix so that its spectral norm is at most ``cap``.

    The same projection as :func:`~ftnode.latent.operator.spectral_clamp_safe`, in closed
    form: ``||B||_2^2 = lambda_max(B^T B) = (a + c)/2 + sqrt(((a - c)/2)^2 + b^2)`` for
    ``B^T B = [[a, b], [b, c]]``.  No batched eigensolver, so it works at any batch size
    (cuSOLVER's batched ``eigvalsh`` rejects batches of tens of thousands) and it is cheaper.
    The ``eps`` inside the square root keeps the gradient finite at a repeated singular
    value.  It can only overestimate the norm, so the clamp stays conservative.
    """
    G = B.transpose(-1, -2) @ B
    a, b, c = G[..., 0, 0], G[..., 0, 1], G[..., 1, 1]
    lam = 0.5 * (a + c) + torch.sqrt(0.25 * (a - c) ** 2 + b**2 + eps)
    scale = torch.clamp(cap / torch.sqrt(lam), max=1.0)
    return B * scale.unsqueeze(-1).unsqueeze(-1)


def matched_free_hidden(target, in_dim, out_dim, depth):
    """The MLP width whose parameter count is closest to ``target``."""
    best_h, best_gap = 1, None
    for h in range(1, 4096):
        gap = abs(mlp_numel(in_dim, out_dim, h, depth) - target)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
        elif gap > best_gap:
            break
    return best_h


class FreeCodeHead(nn.Module):
    """``F(x, u; c) = MLP([x, u, c])``, no structure.  Width matched to a structured head."""

    def __init__(self, d=2, q=1, code_dim=64, hidden=None, depth=3, activation="silu", match=None):
        super().__init__()
        self.d, self.q, self.code_dim = d, q, code_dim
        in_dim = d + q + code_dim
        if hidden is None:
            if match is None:
                raise ValueError("FreeCodeHead needs `hidden` or a head to `match`")
            hidden = matched_free_hidden(match.numel(), in_dim, d, depth)
        self.hidden = hidden
        self.net = MLP(in_dim, d, hidden, depth, activation=activation)

    def forward(self, x, u, c):
        if u.dim() == x.dim() - 1:
            u = u.unsqueeze(-1)
        return self.net(torch.cat([x, u, _broadcast_code(c, x)], dim=-1))

    def numel(self) -> int:
        return sum(p.numel() for p in self.parameters())
