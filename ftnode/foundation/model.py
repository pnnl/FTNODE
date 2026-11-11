"""The context-conditioned field: an encoder that reads a context and a head that outputs ``F``.

:class:`ContextField` holds a :class:`~ftnode.foundation.encoder.SetEncoder` and one head.
Training calls ``field(x, u, c)`` with a code per system.  A rollout first binds one code
with :meth:`ContextField.bind_context` and then calls ``F(x, u)``, so the code stays fixed
for the whole rollout and the field is autonomous -- the setting in which the head's bounds
are statements about trajectories.

Wrapped in :class:`ftnode.physical.PhysicalField`, a bound ``ContextField`` runs through the
existing physical rollout and metric code unchanged.  It has no ``prepare`` method on
purpose: its context is bound explicitly, not taken from each test trajectory's window.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import SetEncoder
from .generator import context_tokens
from .heads import FreeCodeHead, StructuredCodeHead

__all__ = ["ContextField", "build_context_field", "tokens_from_q"]


def tokens_from_q(q, u, h, n_tok=None, generator=None):
    """Context tokens from measured trajectories ``q`` ``(K, T)`` with inputs ``u`` ``(K,)``.

    Returns ``(1, J, 4)``.  With ``n_tok`` set, draws that many tokens uniformly (with the
    given generator, for a reproducible subsample); otherwise keeps all ``K (T - 2)`` tokens.
    """
    tok = context_tokens(q, u, h).reshape(-1, 4)
    if n_tok is not None:
        idx = torch.randint(0, tok.shape[0], (n_tok,), generator=generator).to(tok.device)
        tok = tok[idx]
    return tok.unsqueeze(0)


class ContextField(nn.Module):
    """``F(x, u; c)`` with ``c = encoder(tokens)``."""

    def __init__(self, encoder: SetEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.register_buffer("_c", torch.zeros(1, encoder.code_dim), persistent=False)

    def encode(self, tokens):
        return self.encoder(tokens)

    def forward(self, x, u, c):
        return self.head(x, u, c)

    @torch.no_grad()
    def bind_context(self, tokens):
        """Encode one context ``(1, J, 4)`` and hold its code for :meth:`F`."""
        self._c = self.encoder(tokens).detach()
        return self._c

    def bind_code(self, c):
        self._c = c

    def F(self, x, u):
        return self.head(x, u, self._c)


def build_context_field(head="structured", code_dim=64, enc_width=128, n_latents=16, n_heads=4,
                        n_self=2, head_hidden=64, head_depth=3, R_g=2.0, sigma_min=0.1,
                        kappa_max=25.0, skew_frac=0.6):
    """Build an encoder plus a head.  Call ``torch.manual_seed`` first for a reproducible init.

    The encoder is built before the head.  The free head's width is matched to the
    parameter count of a structured head of the same settings.
    """
    enc = SetEncoder(token_dim=4, d=enc_width, n_latents=n_latents, n_heads=n_heads,
                     n_self=n_self, code_dim=code_dim)
    if head == "structured":
        h = StructuredCodeHead(2, 1, code_dim, head_hidden, head_depth, sigma_min, kappa_max,
                               skew_frac, R_g)
    elif head == "free":
        ref = StructuredCodeHead(2, 1, code_dim, head_hidden, head_depth, sigma_min, kappa_max,
                                 skew_frac, R_g)
        h = FreeCodeHead(2, 1, code_dim, depth=head_depth, match=ref)
    else:
        raise ValueError(f"unknown head {head!r}; choose 'structured' or 'free'")
    return ContextField(enc, h)
