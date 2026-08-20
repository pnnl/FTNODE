"""Architecture configuration and the builders that turn it into a model.

The config mirrors the model's structure: one section per role, each naming its
strategy and sizing only its own sub-networks.  ``operator`` and ``equilibrium``
are declared the same way because they are peers -- see :mod:`ftnode.latent`.

The builders own the **construction order**, which is load-bearing rather than
stylistic -- see :func:`_assemble` and :func:`build_latent_ftnode`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .equilibrium import resolve_g
from .model import LatentFTNODE, LatentNODE, LatentSysID
from .nets import Encoder, LinearDecoder, is_lipschitz_1, resolve_activation
from .operator import KappaBudget, resolve_operator, spectral_clamp

__all__ = [
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


@dataclass(frozen=True)
class EncoderConfig:
    """Sizing for the :class:`~ftnode.latent.nets.Encoder` only.

    ``tau`` must match the dataset's window length
    (:attr:`ftnode.systems.DuffingDataConfig.tau`).
    """

    hidden: int = 64
    depth: int = 2
    z_scale: float = 2.0
    tau: int = 8


@dataclass(frozen=True)
class OperatorConfig:
    """Which ``A(z)`` to build, and the sizing for its sub-networks only.

    ``kind`` names an entry in :data:`~ftnode.latent.operator.A_KINDS`;
    ``kwargs`` carries whatever that operator needs beyond the shared
    ``(m, sigma_min, hidden, depth, activation, budget)`` -- ``beta_min_frac`` for
    Youla, for instance.

    ``kwargs`` must stay YAML-safe.  ``clamp_fn`` is deliberately *not* reachable
    here: it is a callable, and a config file must not be able to change which
    clamp the frozen results were produced with.  Route it through
    :func:`build_clamp`.

    ``sigma_min`` duplicates :attr:`~ftnode.latent.operator.KappaBudget.sigma_min`;
    the budget stays a separate argument to the builders because the control stage
    shares it (``SplitOperator.from_budget``).  Keep the two in sync, as the
    duffing notebooks do.
    """

    kind: str = "svd_clamp"
    hidden: int = 64
    depth: int = 3
    sigma_min: float = 0.1
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EquilibriumConfig:
    """Which ``g(z, u)`` to build, and the sizing for its sub-networks only.

    ``kind`` names an entry in :data:`~ftnode.latent.equilibrium.G_KINDS`;
    ``kwargs`` carries whatever that map needs beyond the shared
    ``(m, q, hidden, depth, R_g, activation)``.  Must stay YAML-safe.
    """

    kind: str = "tanh_mlp"
    hidden: int = 64
    depth: int = 3
    R_g: float = 2.0
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LatentModelConfig:
    """Full architecture specification: shared dimensions plus one section per role.

    Defaults are the values the duffing notebooks use, so ``LatentModelConfig()``
    describes the SVD-clamp model those results were produced with.

    Only genuinely shared settings live at this level.  ``m``/``q`` are the latent
    and input dimensions every part needs; ``activation`` is threaded into every
    sub-network.  Everything else belongs to exactly one role and is sized inside
    that role's section, so there is no longer a bare ``hidden`` whose scope a
    reader has to look up.

    ``activation``, ``operator.kind`` and ``equilibrium.kind`` are **strings**
    rather than classes so the whole config round-trips through
    :func:`ftnode.utils.save_config` -- ``yaml.safe_dump`` cannot serialize a
    class object.  ``activation`` defaults to ``"silu"`` for reproducibility of
    the committed results, which is not the same as it being the right choice --
    see :data:`~ftnode.latent.nets.ACTIVATIONS` and :attr:`lipschitz_1`.

    The config now fully determines the model, so
    :func:`build_latent_ftnode` needs nothing but this and a budget.

    Does **not** describe :class:`~ftnode.latent.model.LatentNODE`, which has
    neither an operator nor an equilibrium map; :func:`build_latent_node` reads
    only ``m``/``q``/``activation`` and takes its own width arguments.

    .. note::
       ``kwargs`` being a ``dict`` makes this frozen dataclass unhashable.
       Nothing keys on a config, and the alternative (a tuple of pairs) does not
       survive a YAML round-trip as readably.
    """

    m: int = 4
    q: int = 1
    activation: str = "silu"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    equilibrium: EquilibriumConfig = field(default_factory=EquilibriumConfig)

    def __post_init__(self):
        # Fail at construction rather than deep inside a layer build.
        resolve_activation(self.activation)
        resolve_operator(self.operator.kind)
        resolve_g(self.equilibrium.kind)

    @property
    def lipschitz_1(self) -> bool:
        """Whether every hidden nonlinearity in this configuration is 1-Lipschitz.

        Composed per-layer Lipschitz bounds -- the kind contraction and
        conditioning arguments rely on -- are only valid when this is ``True``.
        It is ``False`` under the default ``"silu"``.
        """
        return is_lipschitz_1(self.activation)

    def _encoder(self) -> Encoder:
        e = self.encoder
        return Encoder(e.tau, self.m, e.hidden, e.depth, e.z_scale, activation=self.activation)

    def _equilibrium(self):
        g = self.equilibrium
        return resolve_g(g.kind)(
            self.m, self.q, g.hidden, g.depth, g.R_g, self.activation, **g.kwargs
        )

    def _operator(self, budget: KappaBudget | None, **extra):
        o = self.operator
        return resolve_operator(o.kind)(
            self.m, o.sigma_min, o.hidden, o.depth, self.activation, budget,
            **{**o.kwargs, **extra},
        )


def _assemble(cfg: LatentModelConfig, make_dynamics) -> LatentSysID:
    """Build encoder, then dynamics, then decoder -- in that order.

    The order is load-bearing, not stylistic: these modules draw from the global
    torch RNG as they initialize, so building them in a different sequence gives
    different weights for the same ``torch.manual_seed(s)``.  This is the order
    the notebooks' ``build_*`` closures use (Python evaluates
    ``LatentSysID(Encoder(...), Dynamics(...), Decoder(...))`` left to right), and
    matching it is what lets a seed reproduce a frozen notebook run.
    """
    encoder = cfg._encoder()
    dynamics = make_dynamics() # Initilaize different builds
    decoder = LinearDecoder(cfg.m)
    return LatentSysID(encoder, dynamics, decoder)


def build_latent_ftnode(
    cfg: LatentModelConfig, budget: KappaBudget | None = None, **operator_kwargs
) -> LatentSysID:
    """Build the model ``cfg`` describes: ``F(z,u) = A(z)(z - g(z,u))``.

    ``cfg.operator.kind`` and ``cfg.equilibrium.kind`` select the two halves
    independently, so every pairing is reachable from here.  ``operator_kwargs``
    passes non-serializable extras such as ``clamp_fn`` that a config must not
    carry.

    .. warning::
       The equilibrium map is constructed **before** the operator.  Both draw from
       the global torch RNG, so this order is what ``torch.manual_seed(s)``
       reproduces; it matches the frozen notebooks, where ``g_net`` was built in
       the base ``__init__`` ahead of the operator sub-networks.  Swapping the two
       lines raises no error and yields correct kappa values -- it silently stops
       reproducing every committed result.  Pinned by
       ``tests/test_shapes.py::test_builders_construct_the_equilibrium_map_before_the_operator``
       and, transitively, by the whole notebook-equivalence suite.
    """
    def make_dynamics():
        equilibrium = cfg._equilibrium()          # FIRST -- see the warning above
        operator = cfg._operator(budget, **operator_kwargs)
        return LatentFTNODE(operator, equilibrium)

    return _assemble(cfg, make_dynamics)


def _with_operator(cfg: LatentModelConfig, kind: str, **kwargs) -> LatentModelConfig:
    """``cfg`` with its operator section overridden -- how the named builders work."""
    op = replace(cfg.operator, kind=kind, kwargs={**cfg.operator.kwargs, **kwargs})
    return replace(cfg, operator=op)


def build_clamp(
    cfg: LatentModelConfig, budget: KappaBudget, clamp_fn=spectral_clamp
) -> LatentSysID:
    """The kappa-bounded SVD-clamp model.  This is the ID architecture the control stage builds on."""
    return build_latent_ftnode(_with_operator(cfg, "svd_clamp"), budget, clamp_fn=clamp_fn)


def build_youla(
    cfg: LatentModelConfig, budget: KappaBudget, beta_min_frac: float = 0.0
) -> LatentSysID:
    """The kappa-bounded Youla model: same cap, no SVD anywhere."""
    return build_latent_ftnode(
        _with_operator(cfg, "youla", beta_min_frac=beta_min_frac), budget
    )


def build_unbounded(cfg: LatentModelConfig, budget: KappaBudget | None = None) -> LatentSysID:
    """The unconstrained-skew FT baseline (stable, but kappa is unbounded)."""
    return build_latent_ftnode(_with_operator(cfg, "unbounded"), budget)


def build_latent_node(cfg: LatentModelConfig, hidden: int = 95, depth: int = 4) -> LatentSysID:
    """The unstructured, parameter-matched latent NODE baseline."""
    return _assemble(
        cfg,
        lambda: LatentNODE(
            m=cfg.m, q=cfg.q, hidden=hidden, depth=depth, activation=cfg.activation
        ),
    )
