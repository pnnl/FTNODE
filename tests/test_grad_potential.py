"""The symmetric-Jacobian equilibrium map: its structure, its bound, and its plumbing.

Every test here is algebraic and holds at initialization -- no training, matching the rest
of the suite.  The one thing that needs care is the `last_zero` start: `GradPotentialG`
begins at `Phi == 0`, hence `g == 0` and `J_g == 0`, so a symmetry or bound check run
straight after construction passes vacuously.  `live_g` below breaks that degeneracy.
"""
import copy

import numpy as np
import pytest
import torch

from ftnode.latent import (
    ACTIVATION_LIPSCHITZ,
    ACTIVATIONS,
    GradPotentialG,
    KappaBudget,
    lipschitz_bound,
)


def _spec_norms(g):
    return [torch.linalg.matrix_norm(l.weight, ord=2).item() for l in g._linears()]


@pytest.fixture
def live_g():
    """A `GradPotentialG` with a non-degenerate potential.

    The default `init_scale` already gives a non-zero readout, so this is just the default
    map at a smaller width.  It is a named fixture rather than an inline constructor
    because several tests below would be *vacuously true* on a zero potential -- symmetry
    and bound checks both pass trivially when `J_g == 0` and `g == 0`.
    """
    torch.manual_seed(0)
    return GradPotentialG(m=4, q=1, hidden=32, depth=3)


@pytest.fixture
def zu():
    gen = torch.Generator().manual_seed(1)
    return (2 * torch.rand(96, 4, generator=gen) - 1) * 2.0, \
           (2 * torch.rand(96, generator=gen) - 1) * 0.25


def _jacobians(g, Z, U):
    from torch.func import jacrev, vmap

    def single(zi, ui):
        return g(zi.unsqueeze(0), ui.reshape(1, 1)).squeeze(0)

    return vmap(jacrev(single, argnums=0))(Z, U).detach()


# ------------------------------------------------------------------ the structural claim


def test_jacobian_is_symmetric(live_g, zu):
    """`J_g` is a Hessian, so it is symmetric -- the entire point of this map.

    Tolerance is RELATIVE.  Absolute asymmetry is ~1e-9 at init but scales with ||J_g||,
    which reaches O(1) once trained; an absolute bound would pass vacuously here and fail
    on a trained model for no reason.
    """
    J = _jacobians(live_g, *zu)
    scale = J.abs().max().item()
    assert scale > 1e-4, "degenerate potential -- the test would be vacuous"
    assert (J - J.transpose(-1, -2)).abs().max().item() <= 1e-5 * scale


def test_the_incumbent_map_is_not_symmetric(zu):
    """Negative control: `tanh_mlp` FAILS the symmetry check.

    Without this, `test_jacobian_is_symmetric` could pass for a reason unrelated to the
    construction -- a vanishing Jacobian, a broken `jacrev` call -- and nobody would know.
    """
    from ftnode.latent import BoundedTanhG

    torch.manual_seed(0)
    g = BoundedTanhG(m=4, q=1, hidden=32, depth=3)
    with torch.no_grad():
        g.net[-1].weight.normal_(0.0, 0.5)
    J = _jacobians(g, *zu)
    assert (J - J.transpose(-1, -2)).abs().max().item() > 1e-3 * J.abs().max().item()


def test_grad_of_sum_equals_per_sample_gradients(live_g, zu):
    """`forward` differentiates the SUMMED potential; that must equal per-sample grads.

    Valid only because `MLP` has no cross-sample layer.  Dropping a `BatchNorm` into it
    would make the map silently wrong everywhere, and this is the test that would catch it.
    """
    from torch.func import grad, vmap

    Z, U = zu
    reference = vmap(grad(lambda zi, ui: live_g.phi(zi.unsqueeze(0), ui).squeeze(0)))(
        Z, U.unsqueeze(-1)
    )
    assert torch.equal(live_g(Z, U), reference)


# ------------------------------------------------------------------------- the bound


def test_projection_enforces_every_cap(live_g):
    """After `project_`, each weight sits at or under its own spectral cap."""
    with torch.no_grad():
        for layer in live_g._linears():
            layer.weight.mul_(10.0)
    assert max(n - c for n, c in zip(_spec_norms(live_g), live_g._caps)) > 1.0
    live_g.project_()
    for norm, cap in zip(_spec_norms(live_g), live_g._caps):
        assert norm <= cap + 1e-5


def test_the_a_priori_bound_survives_adversarial_training(live_g):
    """The certificate has to hold under an optimizer actively pushing against it.

    A one-shot projection check only proves the projection works.  What the guarantee needs
    is that the invariant survives the training loop -- so this maximizes ||g|| with a
    projection after every step, exactly as `train_one` does, and re-checks.
    """
    gen = torch.Generator().manual_seed(3)
    Z = (2 * torch.rand(256, 4, generator=gen) - 1) * 2.0
    U = torch.zeros(256)
    opt = torch.optim.Adam(live_g.parameters(), lr=5e-2)
    for _ in range(30):
        opt.zero_grad()
        (-live_g(Z, U).norm(dim=-1).max()).backward()
        opt.step()
        live_g.project_()
    assert live_g(Z, U).norm(dim=-1).max().item() <= live_g.g_bound + 1e-5
    for norm, cap in zip(_spec_norms(live_g), live_g._caps):
        assert norm <= cap + 1e-5


def test_projection_is_idempotent(live_g):
    """Projecting twice changes nothing -- to float32 tolerance, not bitwise.

    `eigvalsh` plus a multiply drifts ~1e-7; `test_clamp.py` uses tolerances for the same
    reason.  A drift that grew step over step would silently shrink the weights across a
    200-epoch run.
    """
    live_g.project_()
    first = [l.weight.clone() for l in live_g._linears()]
    live_g.project_()
    for a, layer in zip(first, live_g._linears()):
        assert torch.allclose(a, layer.weight, atol=1e-6)


def test_zero_init_scale_starves_the_hidden_layers_of_gradient():
    """Why `init_scale` defaults non-zero: a zero readout gives EVERY hidden layer zero grad.

    With `Phi == 0` the map sits at a point where the only parameter receiving a gradient
    is the readout -- and here the readout's gradient is attenuated by the product of the
    capped hidden norms, so the map escapes the cold start far more slowly than
    `BoundedTanhG` does.  Measured at 15 epochs: `max||g||` of 0.011 at `init_scale=0`
    against 0.377 at 0.5.  This pins the mechanism, not the training outcome.
    """
    z, u = torch.randn(8, 4), torch.randn(8)

    cold = GradPotentialG(hidden=32, depth=3, init_scale=0.0)
    assert torch.equal(cold(z, u), torch.zeros(8, 4))
    cold(z, u).sum().backward()
    hidden = cold._linears()[:-1]
    assert all(l.weight.grad is None or l.weight.grad.abs().max() == 0 for l in hidden)

    warm = GradPotentialG(hidden=32, depth=3, init_scale=0.5)
    assert warm(z, u).abs().max() > 0
    warm(z, u).sum().backward()
    assert any(l.weight.grad.abs().max() > 0 for l in warm._linears()[:-1])


def test_init_scale_still_respects_the_cap():
    """A large `init_scale` must not start the map outside its own constraint set."""
    g = GradPotentialG(hidden=32, depth=3, init_scale=50.0)
    for norm, cap in zip(_spec_norms(g), g._caps):
        assert norm <= cap + 1e-5


def test_uncapped_declares_an_infinite_bound():
    """`cap=False` is the free potential -- it must not claim a certificate it lacks."""
    g = GradPotentialG(cap=False)
    assert g.g_bound == float("inf")
    before = [l.weight.clone() for l in g._linears()]
    g.project_()
    for a, layer in zip(before, g._linears()):
        assert torch.equal(a, layer.weight)


def test_caps_multiply_out_to_g_cap():
    """`g_bound` is the product of the caps times the activation gain, per layer."""
    for depth in (1, 2, 3):
        g = GradPotentialG(depth=depth, g_cap=3.0, w1_cap=10.0, activation="tanh")
        assert len(g._caps) == depth + 1        # MLP(depth=d) has d+1 weight matrices
        assert g._caps[0] == pytest.approx(10.0)
        assert float(np.prod(g._caps)) == pytest.approx(3.0, rel=1e-6)
        assert g.g_bound == pytest.approx(3.0, rel=1e-6)


def test_silu_carries_its_activation_gain_into_the_bound():
    """A non-1-Lipschitz activation must inflate the certificate, once per activation."""
    tanh_g = GradPotentialG(depth=3, g_cap=3.0, activation="tanh")
    silu_g = GradPotentialG(depth=3, g_cap=3.0, activation="silu")
    assert tanh_g.l_sigma == 1.0
    assert silu_g.l_sigma == pytest.approx(1.10)
    # Both certify g_cap; silu gets there with proportionally smaller weights.
    assert silu_g.g_bound == pytest.approx(tanh_g.g_bound, rel=1e-6)
    assert float(np.prod(silu_g._caps)) < float(np.prod(tanh_g._caps))


# ------------------------------------------------------------------- activation table


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_lipschitz_table_upper_bounds_the_measurement(name):
    """The certificate constant must be an UPPER bound on the measured gain.

    `empirical_lipschitz` is a grid maximum, hence a lower bound on the true supremum, so
    the table has to sit above it -- `silu` at 1.10 against a measured 1.0998.  A table
    entry below the measurement would produce certificates that are simply wrong.
    """
    from ftnode.diagnostics import empirical_lipschitz

    cls = ACTIVATIONS[name][0]
    assert ACTIVATION_LIPSCHITZ[name] >= empirical_lipschitz(cls()) - 1e-9


def test_unknown_activation_has_no_silent_default():
    """An unrecorded activation must raise, not quietly certify at 1.0."""
    with pytest.raises(ValueError, match="no Lipschitz bound recorded"):
        lipschitz_bound("mish")


@pytest.mark.parametrize("act", ["relu", "leaky_relu"])
def test_piecewise_linear_activations_are_rejected(act):
    """A piecewise-linear potential has `grad^2 Phi == 0`, so it cannot be multistable.

    Both are flagged `lipschitz_1=True`, which is exactly what "pick a 1-Lipschitz
    activation for a tight certificate" steers toward -- and the failure is silent: the
    model trains, every bound holds, and the pitchfork simply never forms.
    """
    with pytest.raises(ValueError, match="piecewise-linear"):
        GradPotentialG(activation=act)


# ------------------------------------------------------------------------- plumbing


@pytest.mark.parametrize(
    "z_shape,u_shape",
    [((8, 4), (8,)), ((8, 4), (8, 1)), ((5, 7, 4), (5, 7)), ((5, 7, 4), (5, 7, 1))],
)
def test_rank_handling(live_g, z_shape, u_shape):
    """Rank-2 from the field, rank-3 from the residual penalty, both `u` conventions.

    `ftnode.train.train_one` calls `g(zs, u_steps)` with `zs` of `(b, L+1, m)`, which is
    the shape a `vmap`-based implementation would have to reshape around.
    """
    out = live_g(torch.randn(*z_shape), torch.randn(*u_shape))
    assert out.shape == z_shape
    assert torch.isfinite(out).all()


def test_no_grad_path_matches_the_grad_path(live_g, zu):
    """Every eval rollout and no-grad diagnostic runs `g` under `torch.no_grad()`.

    `torch.func.grad` inside a no-grad region is exactly where a differentiation-based `g`
    could silently return zeros or stale values.  It does not -- and this is the guard.
    """
    Z, U = zu
    with_grad = live_g(Z, U).detach()
    with torch.no_grad():
        without = live_g(Z, U)
    assert torch.equal(with_grad, without)
    assert not without.requires_grad


def test_composes_with_torch_func_jvp(live_g):
    """The control stage jvps straight through `g` (`FrozenLatentPlant.DuF_closed`).

    An implementation built on `torch.autograd.grad` raises here -- functorch refuses the
    nested `requires_grad_` -- which is why `forward` uses `torch.func.grad`.  Without this
    test the faster-looking `autograd.grad` version reads as a harmless simplification.
    """
    z, u = torch.randn(8, 4), torch.randn(8, 1)
    _, dg = torch.func.jvp(lambda uu: live_g(z, uu), (u,), (torch.ones_like(u),))
    assert dg.shape == (8, 4)
    assert torch.isfinite(dg).all()
    assert dg.abs().max() > 0


def test_readout_bias_never_receives_a_gradient(live_g, zu):
    """`grad_z Phi` does not depend on the readout bias, so it is permanently dead.

    Documented rather than fixed: dropping the bias would fork `MLP`, and `last_zero`
    leaves it at exactly 0, which is the `Phi(0,u) = 0` normalization the trapping-radius
    argument assumes.  Pinned so the next person to notice has an answer.
    """
    live_g(*zu).sum().backward()
    assert live_g._linears()[-1].bias.grad is None
    assert all(l.weight.grad is not None for l in live_g._linears())
    assert torch.isfinite(
        torch.nn.utils.clip_grad_norm_(live_g.parameters(), 1.0)
    )


def test_curvature_headroom_clears_the_saddle_threshold_and_leaves_weights_alone(live_g):
    """A saddle needs `lambda_max(J_g) > 1`; the default caps must admit that.

    The probe runs on a copy -- it optimizes a potential, and doing that to the caller's
    weights would be a diagnostic with a side effect.

    Uses the default step budget deliberately.  The result is an optimizer lower bound and
    climbs with it (measured at `hidden=32`: 0.82 at 120 steps, 2.01 at 300, 4.02 at 600),
    so a probe that is cheap enough to be worth calling has to be budgeted to clear the
    threshold -- which is a property of the default, not of the test.
    """
    before = [l.weight.clone() for l in live_g._linears()]
    assert live_g.curvature_headroom() > 1.0
    for a, layer in zip(before, live_g._linears()):
        assert torch.equal(a, layer.weight)


# --------------------------------------------------------------- the training-loop hook


def test_train_one_projects_after_every_optimizer_step():
    """The certificate is an invariant only if EVERY step is followed by a projection."""
    from ftnode.systems import DuffingDataConfig, make_dataset
    from ftnode.train import TrainConfig, train_one

    calls = {"n": 0}

    class Counting(GradPotentialG):
        def project_(self):
            calls["n"] += 1
            super().project_()

    from dataclasses import replace

    from ftnode.latent import G_KINDS, LatentModelConfig, build_latent_ftnode

    G_KINDS["_counting_test"] = Counting
    try:
        cfg = LatentModelConfig()
        cfg = replace(cfg, equilibrium=replace(cfg.equilibrium, kind="_counting_test"))
        torch.manual_seed(0)
        model = build_latent_ftnode(cfg, KappaBudget())
        data = make_dataset(DuffingDataConfig(n_traj=8, L=5, seed=0))
        val = make_dataset(DuffingDataConfig(n_traj=4, L=5, seed=1))
        calls["n"] = 0  # discard the construction-time projection
        n_epochs, batch = 2, 4
        train_one(
            model, data, val,
            TrainConfig(n_epochs=n_epochs, batch=batch, L=5, L_eval=5, lam_res=1e-2),
            ckpt_path="/dev/null", verbose=False,
        )
        expected = n_epochs * -(-len(data) // batch)
        assert calls["n"] == expected, f"{calls['n']} projections, expected {expected}"
    finally:
        del G_KINDS["_counting_test"]


def test_the_hook_is_inert_for_models_that_predate_it(model_cfg, budget):
    """No existing model has `project_`, so the loop must collect an EMPTY list.

    This is what keeps the bitwise notebook-equivalence suite unaffected: an empty list
    means the hook draws no RNG, allocates nothing, and runs no tensor op.
    """
    from ftnode.latent import build_clamp, build_latent_node

    for build in (lambda: build_clamp(model_cfg, budget), lambda: build_latent_node(model_cfg)):
        model = build()
        assert [m for m in model.modules() if callable(getattr(m, "project_", None))] == []


def test_checkpoint_round_trips_without_cap_metadata():
    """Caps are plain floats, not buffers -- they must NOT appear in the state dict.

    A buffer would freeze the derivation into the checkpoint format, exactly the trap the
    `KappaBudget` floats avoid on the operator side.  The architecture is rebuilt from the
    config; only weights are serialized.
    """
    torch.manual_seed(0)
    g = GradPotentialG(hidden=16, depth=2)
    keys = set(g.state_dict())
    assert not any("cap" in k or "bound" in k for k in keys)
    clone = GradPotentialG(hidden=16, depth=2)
    clone.load_state_dict(g.state_dict())
    z, u = torch.randn(4, 4), torch.randn(4)
    assert torch.equal(g(z, u), clone(z, u))
    assert clone.g_bound == g.g_bound
