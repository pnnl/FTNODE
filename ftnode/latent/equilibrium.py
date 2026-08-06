"""Equilibrium maps ``g(z, u)`` -- one half of the split ``F = A(z)(z - g(z,u))``.

Selecting a map by name from :data:`G_KINDS` is what lets any ``g`` compose with
any operator (see :mod:`ftnode.latent.operator`) instead of needing a class per
pair.  The frozen ``duffing_partial_obs_latent_ftnode_superposition_g_*`` notebooks
show the cost of the alternative: they declare ``LatentFTNODESuper`` with ``A``
re-implemented inline, usable with one operator out of three.

.. warning::
   The module attached here is reached under a fixed attribute name on the
   dynamics module, and that name is a checkpoint key.  See :mod:`ftnode.latent`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .nets import MLP

__all__ = ["BoundedTanhG", "G_KINDS", "resolve_g"]


class BoundedTanhG(MLP):
    """Default equilibrium map: ``g(z, u) = R_g * tanh(MLP([z, u]))``.

    The ``tanh`` confines the image of ``g`` -- i.e. the model's equilibria -- to
    the box ``[-R_g, R_g]^m``.  This is the form every duffing notebook and every
    committed checkpoint was produced with.

    It **subclasses** :class:`~ftnode.latent.nets.MLP` rather than wrapping one,
    so its weights sit at ``equilibrium.net.*`` and its parameter draws are
    bitwise identical to the bare ``MLP`` this replaced.  A new kind is under no
    obligation to do the same -- a checkpoint written by one map was never
    loadable into another.
    """

    def __init__(self, m, q, hidden, depth, R_g, activation):
        super().__init__(m + q, m, hidden, depth, last_zero=True, activation=activation)
        self.m, self.q = m, q
        self.R_g = R_g

    def forward(self, z, u):
        """``u`` may be ``(b,)`` or ``(b, q)``; broadcasts over rollout dims too."""
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.R_g * torch.tanh(self.net(torch.cat([z, u], -1)))


#: Selectable equilibrium maps, keyed by the string a config stores.
#:
#: A string rather than a class for the same reason ``activation`` is one: the
#: config has to round-trip through :func:`ftnode.utils.save_config`, and
#: ``yaml.safe_dump`` cannot serialize a class object.
#:
#: Every entry must accept ``(m, q, hidden, depth, R_g, activation)`` positionally
#: plus any variant-specific keywords, and expose ``forward(z, u) -> (..., m)``.
#: That signature is the whole contract -- the model builds whichever map it is
#: told to and never looks inside, so an entry is free to carry as much internal
#: structure as its math needs.
#:
#: .. note::
#:    The equilibrium map is constructed *before* the operator, so a kind with a
#:    different parameter count shifts the RNG stream the operator draws from.  At
#:    a fixed seed, "same ``A``, different ``g``" gives a structurally identical
#:    but numerically different ``A(z)`` at initialization.  That is expected --
#:    the order is fixed for checkpoint compatibility -- but it means a ``g``
#:    comparison is only fair across seeds, not within one.
G_KINDS: dict[str, type] = {
    "tanh_mlp": BoundedTanhG,
}


def resolve_g(spec):
    """Turn an equilibrium-map spec into the class to construct.

    Accepts a name from :data:`G_KINDS`, an ``nn.Module`` subclass, or any
    callable returning a module.  Mirrors
    :func:`~ftnode.latent.nets.resolve_activation`.
    """
    if isinstance(spec, str):
        try:
            return G_KINDS[spec]
        except KeyError:
            raise ValueError(
                f"unknown g_kind {spec!r}; choose from {sorted(G_KINDS)} "
                "or pass an nn.Module subclass"
            ) from None
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    if callable(spec):
        return spec
    raise TypeError(f"g_kind must be a name, nn.Module subclass, or factory; got {spec!r}")
