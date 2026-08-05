"""Activation selection, and the Lipschitz property that motivates it.

Composed per-layer Lipschitz bounds -- what contraction and conditioning
arguments rest on -- are only valid if every hidden nonlinearity has unit gain.
The default `silu` does not, so the point of these tests is twofold: pin the
numbers so the `ACTIVATIONS` table cannot drift away from reality, and prove the
activation is genuinely selectable without disturbing checkpoints or seeds.
"""
import pytest
import torch
import torch.nn as nn

from ftnode.diagnostics import empirical_lipschitz
from ftnode.latent import (
    ACTIVATIONS,
    MLP,
    Encoder,
    LatentModelConfig,
    build_clamp,
    build_latent_node,
    build_unbounded,
    build_youla,
    is_lipschitz_1,
    resolve_activation,
)


# ------------------------------------------------------------------ Lipschitz


def test_silu_is_not_1_lipschitz():
    """The default activation exceeds unit gain -- this is why the flag exists."""
    L = empirical_lipschitz(nn.SiLU())
    assert L > 1.0
    assert L == pytest.approx(1.0998, abs=1e-3)
    assert is_lipschitz_1("silu") is False


def test_gelu_is_not_1_lipschitz():
    L = empirical_lipschitz(nn.GELU())
    assert L > 1.0
    assert L == pytest.approx(1.1289, abs=1e-3)
    assert is_lipschitz_1("gelu") is False


@pytest.mark.parametrize("name", [n for n, (_, ok) in ACTIVATIONS.items() if ok])
def test_flagged_activations_really_are_1_lipschitz(name):
    """Every activation the table flags as 1-Lipschitz must measure at most 1."""
    cls, _ = ACTIVATIONS[name]
    assert empirical_lipschitz(cls()) <= 1.0 + 1e-9, name


@pytest.mark.parametrize("name", [n for n, (_, ok) in ACTIVATIONS.items() if not ok])
def test_unflagged_activations_really_are_not(name):
    """And every activation it does not flag must actually exceed 1, or the flag is noise."""
    cls, _ = ACTIVATIONS[name]
    assert empirical_lipschitz(cls()) > 1.0, name


def test_empirical_lipschitz_on_a_known_function():
    assert empirical_lipschitz(lambda x: 3.0 * x) == pytest.approx(3.0)
    assert empirical_lipschitz(torch.sin) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ resolution


def test_resolve_accepts_name_class_and_factory():
    assert resolve_activation("tanh") is nn.Tanh
    assert resolve_activation(nn.Tanh) is nn.Tanh
    factory = lambda: nn.Tanh()  # noqa: E731
    assert resolve_activation(factory) is factory


def test_unknown_activation_name_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        resolve_activation("swish_but_misspelled")


def test_config_rejects_bad_activation_at_construction():
    """Fail where the mistake was made, not deep inside a layer build."""
    with pytest.raises(ValueError, match="unknown activation"):
        LatentModelConfig(activation="nope")


def test_each_layer_gets_its_own_activation_instance():
    """A shared instance would silently tie any stateful activation across layers."""
    net = MLP(4, 4, hidden=8, depth=3, activation="tanh").net
    acts = [m for m in net if isinstance(m, nn.Tanh)]
    assert len(acts) == 3
    assert len({id(a) for a in acts}) == 3


# ------------------------------------------------------------------ threading


def test_mlp_uses_the_requested_activation():
    net = MLP(4, 4, hidden=8, depth=2, activation="relu").net
    assert any(isinstance(m, nn.ReLU) for m in net)
    assert not any(isinstance(m, nn.SiLU) for m in net)


def test_encoder_threads_activation():
    enc = Encoder(8, 4, activation="elu")
    assert any(isinstance(m, nn.ELU) for m in enc.net.net)


@pytest.mark.parametrize("builder", ["clamp", "youla", "unbounded", "latent_node"])
def test_builders_thread_activation_into_every_subnet(builder, budget):
    """Reaches g_net, L_net, S_net, W_net, b_net and the encoder -- not just one of them."""
    cfg = LatentModelConfig(activation="tanh")
    torch.manual_seed(0)
    model = {
        "clamp": lambda: build_clamp(cfg, budget),
        "youla": lambda: build_youla(cfg, budget),
        "unbounded": lambda: build_unbounded(cfg),
        "latent_node": lambda: build_latent_node(cfg),
    }[builder]()
    mods = list(model.modules())
    assert any(isinstance(m, nn.Tanh) for m in mods)
    assert not any(isinstance(m, nn.SiLU) for m in mods), "a subnet kept the default activation"


def test_split_operator_threads_activation(budget):
    from ftnode.control import SplitOperator

    fpsi = SplitOperator.from_budget(budget, activation="tanh")
    mods = list(fpsi.modules())
    assert any(isinstance(m, nn.Tanh) for m in mods)
    assert not any(isinstance(m, nn.SiLU) for m in mods)


def test_lipschitz_1_property_tracks_the_config():
    assert LatentModelConfig().lipschitz_1 is False  # default silu
    assert LatentModelConfig(activation="tanh").lipschitz_1 is True
    assert LatentModelConfig(activation="relu").lipschitz_1 is True


# --------------------------------------------------- invariants that must hold


def test_activation_does_not_change_state_dict_keys(budget):
    """Checkpoints stay loadable: activations are parameterless and hold their slot."""
    torch.manual_seed(0)
    a = build_clamp(LatentModelConfig(activation="silu"), budget).state_dict()
    torch.manual_seed(0)
    b = build_clamp(LatentModelConfig(activation="tanh"), budget).state_dict()
    assert sorted(a) == sorted(b)
    assert all(a[k].shape == b[k].shape for k in a)


def test_activation_does_not_perturb_rng_draw_order(budget):
    """Weights must be untouched by the activation choice, or seeds stop reproducing."""
    torch.manual_seed(0)
    a = build_clamp(LatentModelConfig(activation="silu"), budget).state_dict()
    torch.manual_seed(0)
    b = build_clamp(LatentModelConfig(activation="tanh"), budget).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_default_is_still_silu(budget):
    """The committed results were produced with SiLU; the default must not drift."""
    assert LatentModelConfig().activation == "silu"
    torch.manual_seed(0)
    model = build_clamp(LatentModelConfig(), budget)
    assert any(isinstance(m, nn.SiLU) for m in model.modules())


def test_config_with_activation_round_trips_through_yaml(tmp_path):
    """A class object would break yaml.safe_dump; the string form must survive."""
    from ftnode.utils import load_config, save_config

    cfg = LatentModelConfig(activation="tanh")
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    assert "tanh" in p.read_text()
    assert load_config(LatentModelConfig, p) == cfg


def test_kappa_bound_holds_under_a_1_lipschitz_activation(budget, latent_box):
    """The structural cap must not depend on which activation was chosen."""
    from ftnode.diagnostics import A_stats

    torch.manual_seed(0)
    model = build_clamp(LatentModelConfig(activation="tanh"), budget)
    maxre, _, kappa = A_stats(model.dynamics, latent_box)
    assert kappa.max() <= budget.kappa_max + 1e-4
    assert maxre.max() <= -budget.sigma_min + 1e-4
