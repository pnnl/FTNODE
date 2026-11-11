"""The context encoder: a set of transition tokens to one global system code ``c``.

A Perceiver-style set encoder.  A fixed number of learned latent queries cross-attend to
the tokens, a few self-attention layers mix the latents, and the latents are mean-pooled
into one code.  There is no positional encoding anywhere, so the code is invariant to the
order of the tokens: the field is time-invariant, so the context is a set.  The latent
bottleneck makes the cost linear in the number of tokens, so a context of 4 or of 64
trajectories goes through the same network.

The code is **global**: one vector per system, not a feature per query state.  A per-query
cross-attention decoder (FIM's design) acts as a local interpolator; a global code must
describe the whole system, which is the only route to an equilibrium the context does not
visit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SetEncoder"]


class SetEncoder(nn.Module):
    """``tokens (B, J, token_dim) -> c (B, code_dim)``, invariant to token order.

    Args:
        token_dim: features per token (4: ``q, q_dot_hat, q_ddot_hat, u``).
        d: the model width.
        n_latents: learned latent queries (the bottleneck).
        n_heads: attention heads.
        n_self: self-attention layers among the latents.
        code_dim: the size of the system code ``c``.
    """

    def __init__(self, token_dim=4, d=128, n_latents=16, n_heads=4, n_self=2, code_dim=64):
        super().__init__()
        self.token_dim, self.d, self.code_dim = token_dim, d, code_dim
        self.embed = nn.Sequential(nn.Linear(token_dim, d), nn.GELU(), nn.Linear(d, d))
        self.latents = nn.Parameter(0.02 * torch.randn(n_latents, d))
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, n_self, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, code_dim))

    def forward(self, tokens):
        t = self.embed(tokens)
        kv = self.ln_kv(t)
        z = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        z = z + self.cross(self.ln_q(z), kv, kv, need_weights=False)[0]
        z = z + self.ff(self.ln_ff(z))
        z = self.mix(z)
        return self.out(z.mean(dim=1))
