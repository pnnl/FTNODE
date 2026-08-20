"""Each config section sizes its own sub-networks and nothing else.

Before the sections existed, every variant hardcoded `MLP(..., 64, 3)` for its
operator sub-networks -- inherited verbatim from the source notebooks -- so a bare
`LatentModelConfig.hidden` reached the equilibrium map alone. Setting
`hidden=128` widened under a third of the model and silently left the operator
networks at 64.

The nested config removes the trap at the source: there is no longer an
unqualified `hidden` whose scope a reader has to look up. These tests pin that the
three roles stay independent, and -- more importantly -- that the defaults did not
move.
"""
import pytest
import torch

from ftnode.latent import (
    EncoderConfig,
    EquilibriumConfig,
    LatentModelConfig,
    OperatorConfig,
    build_clamp,
    build_latent_node,
    build_unbounded,
    build_youla,
)

# Which operator sub-networks each variant actually builds. Youla has no S_net.
OPERATOR_SUBNETS = {
    "clamp": ("L_net", "S_net"),
    "unbounded": ("L_net", "S_net"),
    "youla": ("L_net", "W_net", "b_net"),
}


def _build(name, cfg, budget):
    return {
        "clamp": lambda: build_clamp(cfg, budget),
        "unbounded": lambda: build_unbounded(cfg, budget),
        "youla": lambda: build_youla(cfg, budget),
    }[name]()


def _widths(mod):
    """(hidden width, depth) actually realized by an MLP."""
    lin = [m for m in mod.net if isinstance(m, torch.nn.Linear)]
    return lin[0].out_features, len(lin) - 1


@pytest.mark.parametrize("variant", sorted(OPERATOR_SUBNETS))
def test_default_op_size_is_still_64_by_3(variant, budget):
    """The literal the notebooks hardcoded. If this moves, checkpoints break."""
    torch.manual_seed(0)
    op = _build(variant, LatentModelConfig(), budget).dynamics.operator
    for name in OPERATOR_SUBNETS[variant]:
        assert _widths(getattr(op, name)) == (64, 3), f"{variant}.{name}"


@pytest.mark.parametrize("variant", sorted(OPERATOR_SUBNETS))
def test_op_size_reaches_every_operator_subnet(variant, budget):
    """Including W_net and b_net -- the two easiest to forget, since only Youla has them."""
    cfg = LatentModelConfig(operator=OperatorConfig(hidden=48, depth=2))
    torch.manual_seed(0)
    op = _build(variant, cfg, budget).dynamics.operator
    for name in OPERATOR_SUBNETS[variant]:
        assert _widths(getattr(op, name)) == (48, 2), f"{variant}.{name} kept the old size"


@pytest.mark.parametrize("variant", sorted(OPERATOR_SUBNETS))
def test_op_size_does_not_touch_the_equilibrium_map(variant, budget):
    """The operator and equilibrium sections must stay independent."""
    cfg = LatentModelConfig(operator=OperatorConfig(hidden=48, depth=2))
    torch.manual_seed(0)
    dyn = _build(variant, cfg, budget).dynamics
    assert _widths(dyn.equilibrium) == (cfg.equilibrium.hidden, cfg.equilibrium.depth)


def test_equilibrium_size_does_not_touch_the_operator_subnets(budget):
    """The converse -- this asymmetry is what used to be silent."""
    cfg = LatentModelConfig(equilibrium=EquilibriumConfig(hidden=128, depth=4))
    torch.manual_seed(0)
    dyn = build_clamp(cfg, budget).dynamics
    assert _widths(dyn.equilibrium) == (128, 4)
    assert _widths(dyn.operator.L_net) == (64, 3)
    assert _widths(dyn.operator.S_net) == (64, 3)


def test_encoder_size_is_independent_of_both(budget):
    """Three sections, three scopes."""
    cfg = LatentModelConfig(encoder=EncoderConfig(hidden=16, depth=1))
    torch.manual_seed(0)
    model = build_clamp(cfg, budget)
    assert _widths(model.encoder.net) == (16, 1)
    assert _widths(model.dynamics.operator.L_net) == (64, 3)
    assert _widths(model.dynamics.equilibrium) == (64, 3)


def test_latent_node_is_unaffected(budget):
    """LatentNODE has no A(z), so no operator sub-networks for the section to size."""
    cfg = LatentModelConfig(operator=OperatorConfig(hidden=8, depth=1))
    torch.manual_seed(0)
    dyn = build_latent_node(cfg).dynamics
    assert not hasattr(dyn, "operator")
    assert _widths(dyn.f_net) == (95, 4)  # build_latent_node's own parameter-matched size


def test_defaults_leave_weights_bitwise_unchanged(budget):
    """Threading the values through must not disturb RNG draw order."""
    explicit_cfg = LatentModelConfig(
        encoder=EncoderConfig(hidden=64, depth=2, z_scale=2.0, tau=8),
        operator=OperatorConfig(kind="svd_clamp", hidden=64, depth=3, sigma_min=0.1),
        equilibrium=EquilibriumConfig(kind="tanh_mlp", hidden=64, depth=3, R_g=2.0),
    )
    torch.manual_seed(0)
    explicit = build_clamp(explicit_cfg, budget).state_dict()
    torch.manual_seed(0)
    default = build_clamp(LatentModelConfig(), budget).state_dict()
    assert sorted(explicit) == sorted(default)
    assert all(torch.equal(explicit[k], default[k]) for k in default)


def test_off_default_changes_shapes_so_checkpoints_fail_loudly(budget):
    """A resized model is a different architecture; strict loading should reject it."""
    torch.manual_seed(0)
    ref = build_clamp(LatentModelConfig(), budget).state_dict()
    torch.manual_seed(0)
    small = build_clamp(LatentModelConfig(operator=OperatorConfig(hidden=32, depth=2)), budget)
    with pytest.raises(RuntimeError):
        small.load_state_dict(ref, strict=True)


def test_kappa_bound_survives_resizing(budget, latent_box):
    """The cap is structural, so it cannot depend on sub-network capacity."""
    from ftnode.diagnostics import A_stats

    cfg = LatentModelConfig(operator=OperatorConfig(hidden=32, depth=2))
    for build in (lambda: build_clamp(cfg, budget), lambda: build_youla(cfg, budget)):
        torch.manual_seed(0)
        maxre, _, kappa = A_stats(build().dynamics, latent_box)
        assert kappa.max() <= budget.kappa_max + 1e-4
        assert maxre.max() <= -budget.sigma_min + 1e-4


def test_nested_config_round_trips_through_yaml(tmp_path):
    """`save_config`/`load_config` must recurse into the sections, not flatten them."""
    from ftnode.utils import load_config, save_config

    cfg = LatentModelConfig(
        encoder=EncoderConfig(hidden=16, depth=1),
        operator=OperatorConfig(kind="youla", hidden=48, depth=2, kwargs={"beta_min_frac": 0.1}),
        equilibrium=EquilibriumConfig(hidden=32, depth=2, R_g=1.5),
    )
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    assert load_config(LatentModelConfig, p) == cfg


def test_unknown_key_in_a_nested_section_raises(tmp_path):
    """Strictness has to survive the recursion, or an old config silently defaults."""
    from ftnode.utils import load_config

    (tmp_path / "cfg.yaml").write_text("m: 4\noperator:\n  kind: svd_clamp\n  op_hidden: 64\n")
    with pytest.raises(ValueError, match="OperatorConfig has no fields"):
        load_config(LatentModelConfig, tmp_path / "cfg.yaml")
