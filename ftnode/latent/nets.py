"""Shared building blocks: selectable activations, the MLP, encoder and decoder.

Nothing here knows about operators or equilibrium maps -- this is the bottom of
the dependency order inside :mod:`ftnode.latent`, imported by everything else and
importing nothing from the package itself.

.. warning::
   The ``nn.Module`` attribute names here (``net``, ``c``) are load-bearing: the
   checkpoints committed under ``examples/duffing/`` are state dicts keyed by
   them.  See :mod:`ftnode.latent` for the full serialization contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_LIPSCHITZ",
    "resolve_activation",
    "is_lipschitz_1",
    "lipschitz_bound",
    "MLP",
    "Encoder",
    "LinearDecoder",
]


# -------------------------------------------------------------------- activations

#: Selectable hidden activations, keyed by the string a config stores.
#:
#: The ``lipschitz_1`` flag records whether the activation is 1-Lipschitz, which
#: matters whenever a downstream argument composes per-layer bounds into a bound
#: on the whole network -- contraction and conditioning arguments do exactly
#: that, and they are only valid if every nonlinearity in the path has unit
#: gain.  Note that the default ``silu`` is **not** 1-Lipschitz (its derivative
#: peaks near 1.0998 around x = 2.4), and neither is ``gelu`` (~1.1289).  They
#: remain available because the committed results were produced with ``silu``.
#: Use :func:`ftnode.diagnostics.empirical_lipschitz` to check any of these
#: numerically rather than trusting the table.
ACTIVATIONS: dict[str, tuple[type, bool]] = {
    "silu": (nn.SiLU, False),
    "gelu": (nn.GELU, False),
    "tanh": (nn.Tanh, True),
    "relu": (nn.ReLU, True),
    "leaky_relu": (nn.LeakyReLU, True),
    "elu": (nn.ELU, True),
    "softplus": (nn.Softplus, True),
}


#: Numeric UPPER bounds on ``sup_t |sigma'(t)|``, keyed as :data:`ACTIVATIONS` is.
#:
#: Separate from the ``lipschitz_1`` flag above because a *bound* and a *boolean* answer
#: different questions.  Anything that composes per-layer gains into a bound on a whole
#: network needs the number, not the flag: a spectrally capped potential certifies
#: ``||grad Phi||_2 <= l_sigma**depth * prod_j ||W_j||_2``, and with ``silu`` the
#: ``l_sigma`` factor is 1.10 per activation rather than 1.
#:
#: Values are rounded **up** from :func:`ftnode.diagnostics.empirical_lipschitz`, which is
#: a grid maximum and therefore a *lower* bound on the true supremum -- rounding up is what
#: keeps the certificate conservative in the right direction.  Pinned against that
#: measurement by ``tests/test_grad_potential.py``.
ACTIVATION_LIPSCHITZ: dict[str, float] = {
    "silu": 1.10,        # measured 1.0998
    "gelu": 1.13,        # measured 1.1289
    "tanh": 1.0,
    "relu": 1.0,
    "leaky_relu": 1.0,
    "elu": 1.0,
    "softplus": 1.0,
}


def lipschitz_bound(spec) -> float:
    """Upper bound on the gain of a named activation, for composing a network bound.

    Raises rather than defaulting to ``1.0`` for an unknown spec.  A silent ``1.0`` would
    hand back a certificate that is not merely loose but *wrong* whenever the activation
    exceeds unit gain -- which the two defaults, ``silu`` and ``gelu``, both do.
    """
    if isinstance(spec, str) and spec in ACTIVATION_LIPSCHITZ:
        return ACTIVATION_LIPSCHITZ[spec]
    raise ValueError(
        f"no Lipschitz bound recorded for activation {spec!r}; "
        f"choose from {sorted(ACTIVATION_LIPSCHITZ)}, or measure it with "
        "ftnode.diagnostics.empirical_lipschitz and add it to ACTIVATION_LIPSCHITZ"
    )


def resolve_activation(spec):
    """Turn an activation spec into a zero-argument factory producing fresh modules.

    Accepts a name from :data:`ACTIVATIONS`, an ``nn.Module`` subclass, or any
    callable returning a module.  A *factory* rather than a shared instance so
    every layer gets its own module -- harmless for the parameterless
    activations here, but it keeps a stateful or parameterized activation (say
    ``nn.PReLU``) from being silently tied across layers.
    """
    if isinstance(spec, str):
        try:
            return ACTIVATIONS[spec][0]
        except KeyError:
            raise ValueError(
                f"unknown activation {spec!r}; choose from {sorted(ACTIVATIONS)} "
                "or pass an nn.Module subclass"
            ) from None
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec
    if callable(spec):
        return spec
    raise TypeError(f"activation must be a name, nn.Module subclass, or factory; got {spec!r}")


def is_lipschitz_1(spec) -> bool:
    """Whether a named activation is 1-Lipschitz.  Unknown/custom specs return ``False``."""
    return ACTIVATIONS.get(spec, (None, False))[1] if isinstance(spec, str) else False


# --------------------------------------------------------------------------- nets


class MLP(nn.Module):
    """``Linear -> activation`` repeated ``depth`` times, then a linear readout.

    ``last_zero`` zeroes the final weight and bias, so the module starts at the
    zero map -- used for ``g`` (equilibrium map starts at the origin) and for the
    skew generators (``Q`` starts at the identity).

    ``activation`` takes a name from :data:`ACTIVATIONS`, an ``nn.Module``
    subclass, or a factory.  It defaults to ``"silu"`` because that is what
    produced the committed checkpoints and the frozen notebook results -- **not**
    because it is the best choice.  SiLU is not 1-Lipschitz, so any argument that
    composes per-layer Lipschitz bounds needs one of the ``lipschitz_1``
    activations instead.

    Swapping the activation does not disturb checkpoints: state-dict keys are
    ``net.0.weight``, ``net.2.weight``, ... -- indices into the ``Sequential`` --
    and every activation here is parameterless, so it occupies its slot without
    contributing or shifting a single key.

    Distinct from :class:`ftnode.node.terms.MLP`, which takes a ``dims`` list and
    serves the x-space model.  The two never meet: ``ftnode.node`` deliberately
    does not export its ``MLP``, and this one is reached as ``ftnode.latent.MLP``.
    """

    def __init__(self, in_dim, out_dim, hidden=64, depth=3, last_zero=False, activation="silu"):
        super().__init__()
        act = resolve_activation(activation)
        self.activation = activation
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        if last_zero:
            with torch.no_grad():
                self.net[-1].weight.zero_()
                self.net[-1].bias.zero_()

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    """Information state from a window of past measurements: ``z = z_scale * tanh(MLP(w))``.

    The ``tanh`` confines the latent to the box ``[-z_scale, z_scale]^m``, which
    is what makes the forward-invariance diagnostics well posed.

    .. note::
       **The encoder is not 1-Lipschitz, and choosing a 1-Lipschitz hidden
       activation does not make it one.**  The output map is
       ``z_scale * tanh(.)``, whose gain is ``z_scale`` (2.0 by default), since
       ``tanh`` has unit slope at the origin and the scaling sits outside it.
       So the encoder's Lipschitz constant is ``z_scale * L(MLP)``, bounded below
       by ``z_scale`` no matter what ``activation`` is set to.

       This is separate from the per-layer activation gain that
       :data:`ACTIVATIONS` tracks, and it is *not* currently addressed anywhere in
       the package.  Any end-to-end bound that needs a non-expansive encoder has
       to account for this factor explicitly, or the encoder has to be
       reparameterized (e.g. fold ``z_scale`` into the latent metric, or
       normalize the readout).  Flagged rather than fixed: changing it would
       alter the latent scale and invalidate every committed checkpoint.
    """

    def __init__(self, tau, m, hidden=64, depth=2, z_scale=2.0, activation="silu"):
        super().__init__()
        self.net = MLP(tau, m, hidden, depth, activation=activation)
        self.z_scale = z_scale

    def forward(self, w):
        return self.z_scale * torch.tanh(self.net(w))


class LinearDecoder(nn.Module):
    """Affine readout of the measured output ``y = q`` from the latent state."""

    def __init__(self, m):
        super().__init__()
        self.c = nn.Linear(m, 1, bias=True)

    def forward(self, z):
        return self.c(z).squeeze(-1)
