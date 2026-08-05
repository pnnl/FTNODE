"""`op_hidden`/`op_depth` size the operator sub-networks that assemble A(z).

Before these existed, every variant hardcoded `MLP(..., 64, 3)` for its operator
sub-networks -- inherited verbatim from the source notebooks -- so
`LatentModelConfig.hidden` reached `g_net` alone. Setting `hidden=128` widened
under a third of the model and silently left the operator networks at 64. These
tests pin the new plumbing and, more importantly, pin that the defaults did not
move.
"""
import pytest
import torch

from ftnode.latent import (
    LatentModelConfig,
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
        "unbounded": lambda: build_unbounded(cfg),
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
    dyn = _build(variant, LatentModelConfig(), budget).dynamics
    for name in OPERATOR_SUBNETS[variant]:
        assert _widths(getattr(dyn, name)) == (64, 3), f"{variant}.{name}"


@pytest.mark.parametrize("variant", sorted(OPERATOR_SUBNETS))
def test_op_size_reaches_every_operator_subnet(variant, budget):
    """Including W_net and b_net -- the two easiest to forget, since only Youla has them."""
    cfg = LatentModelConfig(op_hidden=48, op_depth=2)
    torch.manual_seed(0)
    dyn = _build(variant, cfg, budget).dynamics
    for name in OPERATOR_SUBNETS[variant]:
        assert _widths(getattr(dyn, name)) == (48, 2), f"{variant}.{name} kept the old size"


@pytest.mark.parametrize("variant", sorted(OPERATOR_SUBNETS))
def test_op_size_does_not_touch_g_net(variant, budget):
    """`hidden`/`depth` and `op_hidden`/`op_depth` must stay independent."""
    cfg = LatentModelConfig(op_hidden=48, op_depth=2)
    torch.manual_seed(0)
    dyn = _build(variant, cfg, budget).dynamics
    assert _widths(dyn.g_net) == (cfg.hidden, cfg.depth)


def test_g_net_size_does_not_touch_the_operator_subnets(budget):
    """The converse -- this asymmetry is what used to be silent."""
    cfg = LatentModelConfig(hidden=128, depth=4)
    torch.manual_seed(0)
    dyn = build_clamp(cfg, budget).dynamics
    assert _widths(dyn.g_net) == (128, 4)
    assert _widths(dyn.L_net) == (64, 3)
    assert _widths(dyn.S_net) == (64, 3)


def test_latent_node_is_unaffected(budget):
    """LatentNODE has no A(z), so no operator sub-networks for op_* to size."""
    cfg = LatentModelConfig(op_hidden=8, op_depth=1)
    torch.manual_seed(0)
    dyn = build_latent_node(cfg).dynamics
    assert not hasattr(dyn, "L_net")
    assert _widths(dyn.f_net) == (95, 4)  # build_latent_node's own parameter-matched size


def test_defaults_leave_weights_bitwise_unchanged(budget):
    """Threading the values through must not disturb RNG draw order."""
    torch.manual_seed(0)
    explicit = build_clamp(LatentModelConfig(op_hidden=64, op_depth=3), budget).state_dict()
    torch.manual_seed(0)
    default = build_clamp(LatentModelConfig(), budget).state_dict()
    assert sorted(explicit) == sorted(default)
    assert all(torch.equal(explicit[k], default[k]) for k in default)


def test_off_default_changes_shapes_so_checkpoints_fail_loudly(budget):
    """A resized model is a different architecture; strict loading should reject it."""
    torch.manual_seed(0)
    ref = build_clamp(LatentModelConfig(), budget).state_dict()
    torch.manual_seed(0)
    small = build_clamp(LatentModelConfig(op_hidden=32, op_depth=2), budget)
    with pytest.raises(RuntimeError):
        small.load_state_dict(ref, strict=True)


def test_kappa_bound_survives_resizing(budget, latent_box):
    """The cap is structural, so it cannot depend on sub-network capacity."""
    from ftnode.diagnostics import A_stats

    cfg = LatentModelConfig(op_hidden=32, op_depth=2)
    for build in (lambda: build_clamp(cfg, budget), lambda: build_youla(cfg, budget)):
        torch.manual_seed(0)
        maxre, _, kappa = A_stats(build().dynamics, latent_box)
        assert kappa.max() <= budget.kappa_max + 1e-4
        assert maxre.max() <= -budget.sigma_min + 1e-4


def test_op_fields_round_trip_through_yaml(tmp_path):
    from ftnode.utils import load_config, save_config

    cfg = LatentModelConfig(op_hidden=48, op_depth=2)
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    assert load_config(LatentModelConfig, p) == cfg
