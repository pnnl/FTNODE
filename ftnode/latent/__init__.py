"""Latent forward-tracking NODEs with spectrally-bounded ``A(z)``.

Promoted from the two kappa-bounded duffing notebooks:

* ``examples/duffing/duffing_kappa_svdclamp_vs_ln_2variant_10seed.ipynb``
* ``examples/duffing/duffing_kappa_bounded_youla_skew_3variant_10seed.ipynb``

The model family here is *not* the x-space :class:`ftnode.node.FTNODE`
(``f(x) * (x - g(x, u))``, elementwise).  It is the latent, matrix-valued form

    F(z, u) = A(z) @ (z - g(z, u)),

where ``z`` is an information state produced by an encoder from a window of past
measurements, ``g`` is the equilibrium map, and ``A(z)`` carries the stability
structure.

**``A`` and ``g`` are peers.**  Each is a standalone module selected by name --
operators from :data:`~ftnode.latent.operator.A_KINDS`, equilibrium maps from
:data:`~ftnode.latent.equilibrium.G_KINDS` -- and :class:`LatentFTNODE` holds one
of each without inspecting either.  So every operator composes with every
equilibrium map, adding either is a module plus a registry line, and neither is
constrained in how much internal structure it carries.  Both axes are declared
the same way in :class:`LatentModelConfig`.

Module layout::

    nets         activations, MLP, Encoder, LinearDecoder
    operator     KappaBudget, spectral clamps, the A(z) strategies, A_KINDS
    equilibrium  the g(z,u) strategies, G_KINDS
    model        LatentSysID, LatentFTNODE, LatentNODE, state-dict migration
    config       the *Config dataclasses and the builders

This module re-exports the full public API flat, so ``from ftnode.latent import
build_clamp`` keeps working and callers need not know the layout.

.. warning::
   The ``nn.Module`` attribute names across this package (``encoder``/
   ``dynamics``/``decoder``, ``operator``, ``equilibrium``, ``net``, ``L_net``,
   ``S_net``, ``W_net``, ``b_net``, ``c``, and the ``_eye`` buffer) are
   load-bearing: the checkpoints under ``examples/duffing/`` are state dicts
   keyed by them.  Renaming any of them silently breaks ``load_state_dict``.
   ``tests/test_checkpoints.py`` guards this.

   Those checkpoints predate the operator/equilibrium split and are **not**
   rewritten -- :func:`~ftnode.latent.model.migrate_flat_state_dict` re-keys them
   on load.  See its docstring for why the binaries are left alone.
"""
from __future__ import annotations

from .config import (
    EncoderConfig,
    EquilibriumConfig,
    LatentModelConfig,
    OperatorConfig,
    build_clamp,
    build_latent_ftnode,
    build_latent_node,
    build_unbounded,
    build_youla,
)
from .equilibrium import G_KINDS, BoundedTanhG, GradPotentialG, resolve_g
from .feasibility import FeasibilityResult, fit_potential
from .model import (
    LatentFTNODE,
    LatentNODE,
    LatentSysID,
    migrate_flat_state_dict,
)
from .nets import (
    ACTIVATION_LIPSCHITZ,
    ACTIVATIONS,
    MLP,
    Encoder,
    LinearDecoder,
    is_lipschitz_1,
    lipschitz_bound,
    resolve_activation,
)
from .operator import (
    A_KINDS,
    ClampOperator,
    KappaBudget,
    OperatorBase,
    UnboundedOperator,
    YoulaOperator,
    resolve_operator,
    spectral_clamp,
    spectral_clamp_safe,
)

__all__ = [
    # nets
    "ACTIVATIONS",
    "ACTIVATION_LIPSCHITZ",
    "resolve_activation",
    "is_lipschitz_1",
    "lipschitz_bound",
    "MLP",
    "Encoder",
    "LinearDecoder",
    # operator axis
    "KappaBudget",
    "spectral_clamp",
    "spectral_clamp_safe",
    "OperatorBase",
    "UnboundedOperator",
    "ClampOperator",
    "YoulaOperator",
    "A_KINDS",
    "resolve_operator",
    # equilibrium axis
    "BoundedTanhG",
    "GradPotentialG",
    "G_KINDS",
    "resolve_g",
    "FeasibilityResult",
    "fit_potential",
    # models
    "LatentSysID",
    "LatentFTNODE",
    "LatentNODE",
    "migrate_flat_state_dict",
    # config + builders
    "EncoderConfig",
    "OperatorConfig",
    "EquilibriumConfig",
    "LatentModelConfig",
    "build_latent_ftnode",
    "build_clamp",
    "build_youla",
    "build_unbounded",
    "build_latent_node",
]
