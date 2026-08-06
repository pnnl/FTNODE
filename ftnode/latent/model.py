"""The assembled latent models, and the state-dict migration for the layout change.

:class:`LatentFTNODE` is the split field ``F = A(z)(z - g(z,u))`` holding two peer
strategy modules -- an operator from :mod:`~ftnode.latent.operator` and an
equilibrium map from :mod:`~ftnode.latent.equilibrium`.  It never looks inside
either, which is what lets both axes vary independently and carry whatever
internal structure their math needs.

.. warning::
   ``encoder``/``dynamics``/``decoder``, ``operator``, ``equilibrium`` and the
   sub-network names beneath them are checkpoint keys.  See :mod:`ftnode.latent`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .nets import MLP

__all__ = [
    "LatentSysID",
    "LatentFTNODE",
    "LatentNODE",
    "migrate_flat_state_dict",
]


class LatentSysID(nn.Module):
    """Container tying an encoder, a latent vector field, and a decoder together."""

    def __init__(self, encoder, dynamics, decoder):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder

    def encode(self, w):
        return self.encoder(w)

    def decode(self, z):
        return self.decoder(z)

    def F(self, z, u):
        return self.dynamics.F(z, u)


class LatentFTNODE(nn.Module):
    """Split latent field ``F(z, u) = A(z) (z - g(z, u))``.

    One concrete class holding two interchangeable modules.  ``operator(z)``
    returns the ``(b, m, m)`` matrix and ``equilibrium(z, u)`` the ``(..., m)``
    equilibrium point; neither is inspected here, so **any registered operator
    pairs with any registered equilibrium map**.

    ``A`` and ``g`` stay *methods* rather than attributes on purpose: two places
    duck-type on them -- :func:`ftnode.train.train_one` checks
    ``hasattr(dynamics, "g")`` to know the residual regularizer applies, and the
    spectrum diagnostics check ``hasattr(dyn, "A")`` to choose ``eigvals_A`` over
    ``eig_field_jac``.  :class:`LatentNODE` has neither, which is how both know to
    take the other branch.

    .. warning::
       The **equilibrium map must be constructed before the operator**.  Both draw
       from the global torch RNG as they initialize, so the order fixes what
       ``torch.manual_seed(s)`` produces; it matches the frozen notebooks, where
       ``g_net`` was built in the base ``__init__`` ahead of the operator
       sub-networks.  Building them the other way round raises no error and gives
       correct kappa values -- it just silently stops reproducing every committed
       result.  :func:`ftnode.latent.config.build_latent_ftnode` owns this order;
       do not construct the pair anywhere else.
    """

    def __init__(self, operator, equilibrium):
        super().__init__()
        self.operator = operator
        self.equilibrium = equilibrium
        self.m = operator.m
        self.q = getattr(equilibrium, "q", 1)

    def A(self, z):
        """The operator ``A(z)``, shape ``(b, m, m)``."""
        return self.operator(z)

    def g(self, z, u):
        """The equilibrium map.  ``u`` may be ``(b,)`` or ``(b, q)``."""
        return self.equilibrium(z, u)

    def F(self, z, u):
        return torch.einsum("bij,bj->bi", self.A(z), z - self.g(z, u))


class LatentNODE(nn.Module):
    """Unstructured latent NODE baseline: ``F(z, u) = MLP([z, u])``, no structural prior.

    ``hidden=95, depth=4`` makes its dynamics parameter count (~28.3k) match a
    :class:`~ftnode.latent.operator.ClampOperator` model, so the comparison
    isolates structure rather than capacity.  Deliberately exposes neither ``.A``
    nor ``.g``, which is how :func:`ftnode.train.train_one` knows the residual
    regularizer is inert for it and how the kappa diagnostics know to skip it.

    It is not a point on either axis -- there is no operator to swap and no
    equilibrium map to swap -- so it stays outside :class:`LatentFTNODE` entirely.
    """

    def __init__(self, m=4, q=1, hidden=95, depth=4, activation="silu"):
        super().__init__()
        self.m = m
        self.q = q
        self.activation = activation
        self.f_net = MLP(m + q, m, hidden=hidden, depth=depth, activation=activation)

    def F(self, z, u):
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.f_net(torch.cat([z, u], dim=-1))


# ------------------------------------------------------------------- migration


#: Prefix the equilibrium map used before the operator/equilibrium split.
_LEGACY_G_PREFIX = "g_net."


def migrate_flat_state_dict(sd, model):
    """Map a pre-split flat state dict onto the nested layout, for loading.

    Before the operator/equilibrium split, the operator's sub-networks hung
    directly off the dynamics module beside ``g_net``::

        dynamics.L_net.*   dynamics.S_net.*   dynamics._eye   dynamics.g_net.*

    They now live under two funnels::

        dynamics.operator.*                    dynamics.equilibrium.*

    Every committed checkpoint under ``examples/duffing/`` predates the split, so
    this runs on load.  The ``.pth`` files are deliberately **not** rewritten in
    place: ``duffing_learned_splitting_control.ipynb`` is frozen and loads
    ``best-ctrl-id-svdclamp-seed0.pth`` with its own inline *flat* class
    definitions, so rewriting the binary would silently break it.

    The routing is derived from ``model`` rather than from a hardcoded per-variant
    name list.  Those lists are exactly what gets missed -- ``W_net``/``b_net``
    exist only on the Youla operator and were overlooked once already.

    Args:
        sd (dict): A state dict in either layout.  Already-nested input is
            returned unchanged, so this is safe to apply unconditionally.
        model: The target :class:`LatentSysID`, already built.

    Returns:
        dict: ``sd`` re-keyed for ``model``.

    Raises:
        KeyError: If the result does not exactly match the target's key set --
            a mis-routed name fails loudly here rather than as a partial load.
    """
    target = set(model.state_dict())
    if set(sd) == target:
        return dict(sd)

    dyn = model.dynamics
    if not (hasattr(dyn, "operator") and hasattr(dyn, "equilibrium")):
        return dict(sd)  # e.g. LatentNODE -- nothing was ever split

    op_keys = set(dyn.operator.state_dict())
    eq_keys = set(dyn.equilibrium.state_dict())

    out = {}
    for key, value in sd.items():
        if key.startswith("dynamics."):
            rest = key[len("dynamics."):]
            if rest.startswith(_LEGACY_G_PREFIX) and rest[len(_LEGACY_G_PREFIX):] in eq_keys:
                key = "dynamics.equilibrium." + rest[len(_LEGACY_G_PREFIX):]
            elif rest in op_keys:
                key = "dynamics.operator." + rest
        out[key] = value

    missing, unexpected = target - set(out), set(out) - target
    if missing or unexpected:
        raise KeyError(
            "state-dict migration did not land on the target layout; "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return out
