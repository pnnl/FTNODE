"""`A(z)` and `g(z,u)` are peers: selected the same way, and freely composable.

Both used to be reached differently. The operator was chosen by *subclassing* a
base that also owned the equilibrium map, while the map was hardcoded inside that
base -- so pairing a new `g` with an existing `A` meant cloning the operator too.
`duffing_partial_obs_latent_ftnode_superposition_g_3model_10seed.ipynb` did exactly
that, declaring `LatentFTNODESuper` with `A` re-implemented inline and usable with
one operator out of three.

Two things are under test. First, that the *defaults* are bitwise what they were --
a silent break here invalidates every committed checkpoint, and it would present as
slightly-worse training rather than as an error. Second, that both axes really are
independent: every operator works with every equilibrium map, in either direction.
"""
import pytest
import torch
import torch.nn as nn

from ftnode.latent import (
    A_KINDS,
    G_KINDS,
    MLP,
    BoundedTanhG,
    ClampOperator,
    EquilibriumConfig,
    LatentModelConfig,
    OperatorConfig,
    UnboundedOperator,
    YoulaOperator,
    build_latent_ftnode,
    build_latent_node,
    resolve_g,
    resolve_operator,
)
from ftnode.utils import load_config, save_config

A_NAMES = sorted(A_KINDS)


def _cfg(a_kind="svd_clamp", g_kind="tanh_mlp", **g_kwargs):
    return LatentModelConfig(
        operator=OperatorConfig(kind=a_kind),
        equilibrium=EquilibriumConfig(kind=g_kind, kwargs=dict(g_kwargs)),
    )


class _AffineG(nn.Module):
    """Stand-in for a future `g` variant, deliberately *not* an `MLP` subclass.

    Structurally unlike `BoundedTanhG` (one affine layer, a different sub-attribute
    name, an extra `scale` keyword) so a test asserting it took effect cannot pass
    by accident. It ignores `hidden`/`depth`/`activation`, which a real variant is
    also free to do -- the contract is the signature and `forward(z, u)`, not the
    use made of every argument.
    """

    def __init__(self, m, q, hidden, depth, R_g, activation, scale=1.0):
        super().__init__()
        self.m, self.q = m, q
        self.lin = nn.Linear(m + q, m)
        self.R_g, self.scale = R_g, scale

    def forward(self, z, u):
        if u.dim() == z.dim() - 1:
            u = u.unsqueeze(-1)
        return self.scale * self.R_g * torch.tanh(self.lin(torch.cat([z, u], -1)))


class _DiagonalOperator(nn.Module):
    """Stand-in for a future operator: `A(z) = -(sigma_min + softplus(d(z))) I`.

    Contractive by construction, so the shared invariants still hold, but with a
    single sub-network under a different name -- again, unmistakable if it took
    effect.
    """

    def __init__(self, m, sigma_min, hidden, depth, activation, budget, floor=0.0):
        super().__init__()
        self.m, self.sigma_min, self.floor = m, sigma_min, floor
        self.d_net = MLP(m, m, hidden, depth, activation=activation)
        self.register_buffer("_eye", torch.eye(m))

    def forward(self, z):
        d = self.sigma_min + self.floor + nn.functional.softplus(self.d_net(z))
        return -torch.diag_embed(d)


@pytest.fixture
def stub_kinds():
    """Register both stubs for the duration of a test, then remove them.

    Registration is the path a real variant takes -- a string in the registry is
    what makes the config YAML-serializable -- so the tests exercise that rather
    than passing class objects directly.
    """
    G_KINDS["_affine_test"] = _AffineG
    A_KINDS["_diagonal_test"] = _DiagonalOperator
    try:
        yield "_diagonal_test", "_affine_test"
    finally:
        del G_KINDS["_affine_test"]
        del A_KINDS["_diagonal_test"]


# ------------------------------------------------------- the defaults are unchanged


def test_default_g_is_bitwise_identical_to_the_bare_mlp_it_replaced():
    """`BoundedTanhG` must be a pure relocation of the old inline code.

    Reproduces the pre-refactor path -- `MLP(m+q, m, ...)` plus an explicit
    `R_g * tanh(cat([z, u]))` -- and demands zero difference in both the drawn
    parameters and the output.
    """
    m, q, hidden, depth, R_g, act = 4, 1, 64, 3, 2.0, "silu"

    torch.manual_seed(0)
    new = BoundedTanhG(m, q, hidden, depth, R_g, act)
    torch.manual_seed(0)
    old_net = MLP(m + q, m, hidden, depth, last_zero=True, activation=act)

    sd_new, sd_old = new.state_dict(), old_net.state_dict()
    assert sorted(sd_new) == sorted(sd_old), "state-dict keys diverged"
    for k in sd_old:
        assert torch.equal(sd_new[k], sd_old[k]), f"{k} differs"

    g = torch.Generator().manual_seed(1)
    z = (2 * torch.rand(32, m, generator=g) - 1) * 2.0
    u = (2 * torch.rand(32, generator=g) - 1) * 0.25
    old_out = R_g * torch.tanh(old_net(torch.cat([z, u.unsqueeze(-1)], -1)))
    assert (new(z, u) - old_out).abs().max().item() == 0.0


def test_default_g_handles_the_rollout_shape():
    """`train_one` calls `g` on `(b, L+1, m)` latents with `(b, L+1)` inputs."""
    gm = BoundedTanhG(4, 1, 16, 2, 2.0, "silu")
    z, u = torch.randn(5, 7, 4), torch.randn(5, 7)
    assert gm(z, u).shape == (5, 7, 4)
    # An explicit trailing q-dim must work too -- that is how `plant.DuF_closed`
    # calls it, via torch.func.jvp on a (b, 1) input.
    assert gm(z, u.unsqueeze(-1)).shape == (5, 7, 4)


def test_defaults_are_pinned():
    """Changing either default breaks every committed checkpoint."""
    cfg = LatentModelConfig()
    assert cfg.operator.kind == "svd_clamp"
    assert cfg.equilibrium.kind == "tanh_mlp"
    assert A_KINDS["svd_clamp"] is ClampOperator
    assert G_KINDS["tanh_mlp"] is BoundedTanhG


# ----------------------------------------------------------- the axes are symmetric


@pytest.mark.parametrize("a_kind", A_NAMES)
def test_every_operator_accepts_a_non_default_g(a_kind, budget, stub_kinds):
    """One `g` variant, every operator -- the property the funnels exist to provide."""
    _, g_kind = stub_kinds
    torch.manual_seed(0)
    model = build_latent_ftnode(_cfg(a_kind, g_kind), budget)

    assert isinstance(model.dynamics.equilibrium, _AffineG), f"{a_kind} ignored the g kind"
    assert type(model.dynamics.operator) is A_KINDS[a_kind]
    z, u = torch.randn(8, 4), torch.randn(8)
    assert model.dynamics.g(z, u).shape == (8, 4)
    assert torch.isfinite(model.F(z, u)).all(), f"{a_kind}: F broke under a new g"


@pytest.mark.parametrize("g_kind", sorted(G_KINDS) + ["_affine_test"])
def test_a_new_operator_accepts_every_g(g_kind, budget, stub_kinds):
    """And the mirror image: one operator variant, every equilibrium map."""
    a_kind, _ = stub_kinds
    torch.manual_seed(0)
    model = build_latent_ftnode(_cfg(a_kind, g_kind), budget)

    assert isinstance(model.dynamics.operator, _DiagonalOperator), "the operator kind was ignored"
    assert type(model.dynamics.equilibrium) is resolve_g(g_kind)
    z, u = torch.randn(8, 4), torch.randn(8)
    assert model.dynamics.A(z).shape == (8, 4, 4)
    assert torch.isfinite(model.F(z, u)).all(), f"{g_kind}: F broke under a new operator"


def test_kwargs_reach_both_axes(budget, stub_kinds):
    """Variant-specific settings survive the trip through the config sections."""
    a_kind, g_kind = stub_kinds
    cfg = LatentModelConfig(
        operator=OperatorConfig(kind=a_kind, kwargs={"floor": 0.25}),
        equilibrium=EquilibriumConfig(kind=g_kind, kwargs={"scale": 0.5}),
    )
    torch.manual_seed(0)
    dyn = build_latent_ftnode(cfg, budget).dynamics
    assert dyn.operator.floor == 0.25
    assert dyn.equilibrium.scale == 0.5


def test_swapping_one_axis_leaves_the_other_structurally_intact(budget, stub_kinds):
    """Independence, asserted on structure rather than values.

    The equilibrium map is built first, so a map with a different parameter count
    shifts the RNG stream the operator draws from -- the operator is structurally
    untouched but *not* numerically identical. Asserting equality here would encode
    a false invariant.
    """
    _, g_kind = stub_kinds
    z = torch.randn(16, 4)
    torch.manual_seed(0)
    default_A = build_latent_ftnode(_cfg(), budget).dynamics.A(z)
    torch.manual_seed(0)
    swapped_A = build_latent_ftnode(_cfg(g_kind=g_kind), budget).dynamics.A(z)

    assert default_A.shape == swapped_A.shape
    sym = swapped_A + swapped_A.transpose(1, 2)
    assert torch.linalg.eigvalsh(sym).max().item() <= -2 * budget.sigma_min + 1e-5


def test_latent_node_is_on_neither_axis(model_cfg):
    """3.5: the unstructured baseline exposes no `A` and no `g`, on purpose.

    `train_one` duck-types on `hasattr(dynamics, 'g')` to know the residual
    regularizer is inert, and the kappa diagnostics duck-type on `.A`.
    """
    torch.manual_seed(0)
    dyn = build_latent_node(model_cfg).dynamics
    for attr in ("A", "g", "operator", "equilibrium"):
        assert not hasattr(dyn, attr), f"LatentNODE grew a {attr}"


# ------------------------------------------------------------ config round-trips


def test_both_kinds_round_trip_through_yaml(tmp_path, stub_kinds):
    """Both axes are strings, not classes, so a run directory can record them."""
    a_kind, g_kind = stub_kinds
    cfg = LatentModelConfig(
        operator=OperatorConfig(kind=a_kind, kwargs={"floor": 0.25}),
        equilibrium=EquilibriumConfig(kind=g_kind, kwargs={"scale": 0.5}),
    )
    p = tmp_path / "model.yaml"
    save_config(cfg, p)
    assert load_config(LatentModelConfig, p) == cfg


def test_default_config_round_trips_through_yaml(tmp_path):
    cfg = LatentModelConfig()
    p = tmp_path / "model.yaml"
    save_config(cfg, p)
    loaded = load_config(LatentModelConfig, p)
    assert loaded == cfg
    assert loaded.operator.kind == "svd_clamp" and loaded.equilibrium.kind == "tanh_mlp"


@pytest.mark.parametrize(
    "section,message",
    [
        (OperatorConfig(kind="no_such_operator"), "unknown operator kind"),
        (EquilibriumConfig(kind="no_such_map"), "unknown g_kind"),
    ],
)
def test_unknown_kind_raises_at_config_construction(section, message):
    """Fail where the typo is, not deep inside a layer build -- as `activation` does."""
    key = "operator" if isinstance(section, OperatorConfig) else "equilibrium"
    with pytest.raises(ValueError, match=message):
        LatentModelConfig(**{key: section})


def test_resolvers_accept_a_class_and_reject_junk():
    assert resolve_g("tanh_mlp") is BoundedTanhG
    assert resolve_operator("youla") is YoulaOperator
    assert resolve_operator("unbounded") is UnboundedOperator
    assert resolve_g(_AffineG) is _AffineG
    assert resolve_operator(_DiagonalOperator) is _DiagonalOperator
    for fn in (resolve_g, resolve_operator):
        with pytest.raises(TypeError):
            fn(42)
