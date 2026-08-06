"""Shape and wiring checks for the identification stack."""
import numpy as np
import pytest
import torch

from ftnode.systems import (
    DuffingDataConfig,
    DuffingParams,
    duffing_field,
    duffing_field_torch,
    equilibria,
    make_dataset,
    sinks,
)
from ftnode.train import TrainConfig, rk4_step, restore_best, rollout_y, train_one


def test_duffing_has_three_equilibria_inside_the_pitchfork():
    eqs = equilibria(0.0)
    assert len(eqs) == 3
    assert [round(float(e[0]), 6) for e in eqs] == [-1.0, 0.0, 1.0]
    s = sinks(0.0)
    assert set(s) == {"left", "right"}


def test_duffing_has_one_equilibrium_outside_the_pitchfork():
    assert len(equilibria(2.0)) == 1
    assert set(sinks(2.0)) == {"only"}


def test_numpy_and_torch_fields_agree():
    params = DuffingParams()
    x = np.random.default_rng(0).normal(size=(16, 2))
    u = np.random.default_rng(1).normal(size=(16,))
    a = duffing_field(x, u, params)
    b = duffing_field_torch(torch.tensor(x), torch.tensor(u), params).numpy()
    assert np.allclose(a, b)


def test_make_dataset_shapes():
    cfg = DuffingDataConfig(n_traj=8, L=20, tau=4)
    d = make_dataset(cfg)
    assert d.W.shape == (8, 4)
    assert d.U.shape == (8,)
    assert d.Y.shape == (8, 21)
    assert d.Xfull.shape == (8, 21, 2)
    assert len(d) == 8
    assert torch.isfinite(d.Y).all()


def test_make_dataset_is_seed_reproducible():
    cfg = DuffingDataConfig(n_traj=4, L=10, tau=4, seed=7)
    assert torch.equal(make_dataset(cfg).Y, make_dataset(cfg).Y)
    other = DuffingDataConfig(n_traj=4, L=10, tau=4, seed=8)
    assert not torch.equal(make_dataset(cfg).Y, make_dataset(other).Y)


def test_measured_output_is_only_q():
    """The identification problem is latent because qdot is never observed."""
    d = make_dataset(DuffingDataConfig(n_traj=4, L=10, tau=4))
    assert torch.allclose(d.Y, d.Xfull[..., 0])


def test_rk4_step_matches_a_linear_solution():
    """RK4 is exact to O(h^5); on z' = -z it should track exp(-t) closely."""
    z = torch.ones(1, 1)
    h = 0.05
    for _ in range(20):
        z = rk4_step(lambda zz, uu: -zz, z, torch.zeros(1, 1), h)
    assert abs(z.item() - float(np.exp(-1.0))) < 1e-6


def test_rollout_shapes(model_cfg, budget):
    from ftnode.latent import build_clamp

    torch.manual_seed(0)
    model = build_clamp(model_cfg, budget)
    w = torch.randn(5, model_cfg.encoder.tau)
    u = torch.zeros(5)
    ys, zs = rollout_y(model, w, u, L=7, h=0.05)
    assert ys.shape == (5, 8)
    assert zs.shape == (5, 8, model_cfg.m)


def test_builders_construct_the_encoder_first(model_cfg, budget):
    """Module construction order is part of the reproducibility contract.

    Each submodule draws from the global torch RNG as it initializes, so building
    them in a different sequence yields different weights for the same seed. The
    notebooks evaluate `LatentSysID(Encoder(...), Dynamics(...), Decoder(...))`
    left to right; the builders must match, or a seed no longer reproduces a
    frozen notebook run.
    """
    from ftnode.latent import Encoder, build_clamp, build_latent_node, build_unbounded, build_youla

    torch.manual_seed(0)
    reference = Encoder(model_cfg.encoder.tau, model_cfg.m, model_cfg.encoder.hidden, model_cfg.encoder.depth)

    for build in (
        lambda: build_clamp(model_cfg, budget),
        lambda: build_youla(model_cfg, budget),
        lambda: build_unbounded(model_cfg),
        lambda: build_latent_node(model_cfg),
    ):
        torch.manual_seed(0)
        model = build()
        assert torch.equal(model.encoder.net.net[0].weight, reference.net.net[0].weight)


def test_builders_construct_the_equilibrium_map_before_the_operator(model_cfg, budget):
    """The same RNG-order trap as above, on the axis the funnels introduced.

    Both halves draw from the global torch RNG, so the order fixes what
    `torch.manual_seed(s)` produces. It matches the frozen notebooks, where
    `g_net` was built in the base `__init__` ahead of the operator sub-networks.
    Building the operator first raises no error and gives correct kappa values --
    it silently stops reproducing every committed result.

    Checked by building the equilibrium map alone against the same seed: it must
    see the *first* draws, which is only true if the builder constructs it first.
    """
    from ftnode.latent import BoundedTanhG, build_clamp, build_unbounded, build_youla

    g = model_cfg.equilibrium
    torch.manual_seed(0)
    _ = model_cfg._encoder()  # the builder's first draws (see the test above)
    reference = BoundedTanhG(model_cfg.m, model_cfg.q, g.hidden, g.depth, g.R_g, model_cfg.activation)

    for build in (
        lambda: build_clamp(model_cfg, budget),
        lambda: build_youla(model_cfg, budget),
        lambda: build_unbounded(model_cfg, budget),
    ):
        torch.manual_seed(0)
        equilibrium = build().dynamics.equilibrium
        assert torch.equal(equilibrium.net[0].weight, reference.net[0].weight)


def test_latent_node_exposes_no_A_or_g(model_cfg):
    """The residual regularizer and the kappa diagnostics are duck-typed off these."""
    from ftnode.latent import build_latent_node

    dyn = build_latent_node(model_cfg).dynamics
    assert not hasattr(dyn, "A")
    assert not hasattr(dyn, "g")


def test_train_one_smoke(tmp_path, model_cfg, budget):
    from ftnode.latent import build_clamp

    torch.manual_seed(0)
    model = build_clamp(model_cfg, budget)
    train = make_dataset(DuffingDataConfig(n_traj=16, L=10, tau=model_cfg.encoder.tau, seed=0))
    val = make_dataset(DuffingDataConfig(n_traj=4, L=20, tau=model_cfg.encoder.tau, seed=1))
    cfg = TrainConfig(n_epochs=2, batch=8, lam_res=1e-2, L=10, L_eval=20)
    ckpt = tmp_path / "smoke.pth"
    model, hist = train_one(model, train, val, cfg, ckpt_path=ckpt, verbose=False)

    assert len(hist["train"]) == 2
    assert hist["diverged_at"] is None
    assert np.isfinite(hist["best_val"])
    assert hist["res"][0] > 0.0, "residual penalty should be active for an FT model"
    assert ckpt.exists()
    restore_best(model, hist, verbose=False)
    assert not model.training


def test_train_one_defaults_ckpt_path(tmp_path, monkeypatch, model_cfg, budget):
    """The prototype dropped this fallback while keeping ckpt_path=None, so a
    best-val epoch reached torch.save(state_dict, None)."""
    from ftnode.latent import build_clamp

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = build_clamp(model_cfg, budget)
    train = make_dataset(DuffingDataConfig(n_traj=8, L=6, tau=model_cfg.encoder.tau, seed=0))
    val = make_dataset(DuffingDataConfig(n_traj=4, L=8, tau=model_cfg.encoder.tau, seed=1))
    cfg = TrainConfig(n_epochs=1, batch=8, L=6, L_eval=8)
    _, hist = train_one(model, train, val, cfg, label="fallback", verbose=False)
    assert hist["ckpt_path"] == "best-fallback.pth"
    assert (tmp_path / "best-fallback.pth").exists()


def test_residual_penalty_is_inert_for_latent_node(tmp_path, model_cfg):
    from ftnode.latent import build_latent_node

    torch.manual_seed(0)
    model = build_latent_node(model_cfg)
    train = make_dataset(DuffingDataConfig(n_traj=8, L=6, tau=model_cfg.encoder.tau, seed=0))
    val = make_dataset(DuffingDataConfig(n_traj=4, L=8, tau=model_cfg.encoder.tau, seed=1))
    cfg = TrainConfig(n_epochs=1, batch=8, lam_res=1e-2, L=6, L_eval=8)
    _, hist = train_one(model, train, val, cfg, ckpt_path=tmp_path / "ln.pth", verbose=False)
    assert hist["res"][0] == 0.0


def test_config_roundtrip(tmp_path):
    from ftnode.utils import load_config, save_config

    cfg = DuffingDataConfig(n_traj=32, seed=5)
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    assert load_config(DuffingDataConfig, p) == cfg


def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("n_traj: 4\nnot_a_field: 1\n")
    from ftnode.utils import load_config

    with pytest.raises(ValueError, match="not_a_field"):
        load_config(DuffingDataConfig, p)
