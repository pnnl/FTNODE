"""Tests for the out-of-distribution Duffing splits (``ftnode.ood_splits``).

Fast: a handful of short trajectories per split.  Every check is a property of the
plant or of the sampler, so no model is trained.
"""
import dataclasses

import numpy as np
import pytest
import torch

from ftnode.ood_splits import (
    REGIONS,
    SplitConfig,
    barrier_energy,
    energy,
    is_trapped,
    make_split,
    mirror,
    saddle,
)
from ftnode.physical.config import PhysicalConfig, build_cold_unstructured
from ftnode.physical.metrics import rollout_test_split
from ftnode.systems import DuffingDataConfig, make_dataset


def _cfg(region, **kw):
    return SplitConfig(region=region, **{"n_traj": 16, "L": 60, **kw})


def _full_q(ds):
    """The whole measured ``q`` record: the context window, then the rollout targets."""
    return torch.cat([ds.W, ds.Y], 1)


def test_saddle_and_barrier_at_zero_input():
    assert saddle(0.0) == pytest.approx(0.0, abs=1e-12)
    assert barrier_energy(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("u", [-0.25, -0.1, 0.0, 0.1, 0.25])
def test_saddle_is_the_middle_root_and_a_maximum_of_the_potential(u):
    qs = saddle(u)
    assert qs**3 - qs - u == pytest.approx(0.0, abs=1e-10)
    assert abs(qs) < 0.28
    # The potential has a local maximum at the saddle.
    eps = 1e-3
    assert energy(qs, 0.0, u) > energy(qs + eps, 0.0, u)
    assert energy(qs, 0.0, u) > energy(qs - eps, 0.0, u)


def test_saddle_rejects_a_single_well_input():
    with pytest.raises(ValueError):
        saddle(1.0)


def test_is_trapped_examples():
    assert is_trapped(1.0, 0.0, 0.0, side=+1)          # at the right sink
    assert not is_trapped(1.0, 0.0, 0.0, side=-1)      # wrong side
    assert is_trapped(-1.0, 0.0, 0.0, side=-1)
    assert not is_trapped(1.0, 1.5, 0.0, side=+1)      # above the barrier


def test_trapped_right_never_reaches_the_saddle():
    ds = make_split(_cfg("trapped_right"))
    q = _full_q(ds)
    qs = torch.tensor([saddle(float(u)) for u in ds.U]).unsqueeze(1)
    assert (q > qs).all()


def test_trapped_right_draws_satisfy_the_energy_criterion_at_t0():
    # tau = 0 puts the drawn initial state at Xfull[:, 0], before any damping.
    ds = make_split(_cfg("trapped_right", tau=0))
    x0 = ds.Xfull[:, 0].double()
    for (q0, qd0), u in zip(x0.tolist(), ds.U.tolist()):
        assert is_trapped(q0, qd0, u, side=+1)


def test_left_regions_are_exact_mirrors_of_the_right_regions():
    for left, right in (("trapped_left", "trapped_right"), ("half_left", "half_right")):
        a, b = make_split(_cfg(left, seed=3)), make_split(_cfg(right, seed=3))
        assert torch.equal(a.W, -b.W) and torch.equal(a.Y, -b.Y)
        assert torch.equal(a.U, -b.U) and torch.equal(a.Xfull, -b.Xfull)


def test_mirror_is_a_solution_of_the_plant():
    # Odd symmetry: simulating the mirrored initial state gives the mirrored trajectory.
    from ftnode.systems import DuffingParams, simulate

    t = np.arange(40) * 0.05
    x0, u = np.array([0.7, -0.3]), 0.2
    a = simulate(x0, lambda _: u, DuffingParams(), t)
    b = simulate(-x0, lambda _: -u, DuffingParams(), t)
    assert np.allclose(a, -b, atol=1e-12)


def test_trapped_left_stays_left():
    ds = make_split(_cfg("trapped_left"))
    qs = torch.tensor([saddle(float(u)) for u in ds.U]).unsqueeze(1)
    assert (_full_q(ds) < qs).all()


def test_half_right_starts_right_and_some_trajectories_cross():
    ds = make_split(_cfg("half_right", n_traj=64, L=200))
    q = _full_q(ds)
    assert (q[:, 0] > 0).all()
    final = q[:, -40:].mean(1)
    assert (final < 0).any(), "expected some right-half starts to settle in the left well"
    assert (final > 0).any()


def test_far_ring_starts_outside_the_standard_box():
    cfg = _cfg("far_ring", n_traj=32)
    ds = make_split(cfg)
    q0 = ds.W[:, 0].abs()
    assert (q0 >= cfg.far_q[0]).all() and (q0 <= cfg.far_q[1]).all()
    assert (ds.W[:, 0] > 0).any() and (ds.W[:, 0] < 0).any()
    assert torch.isfinite(ds.Y).all()


@pytest.mark.parametrize("region", REGIONS)
def test_layout_matches_make_dataset(region):
    cfg = _cfg(region, n_traj=4, L=12)
    ds = make_split(cfg)
    ref = make_dataset(DuffingDataConfig(n_traj=4, L=12))
    for name in ("W", "U", "Y", "Xfull"):
        assert getattr(ds, name).shape == getattr(ref, name).shape
        assert getattr(ds, name).dtype == getattr(ref, name).dtype
    # Y continues the window: the measured q is Xfull's first component.
    assert torch.equal(ds.Y, ds.Xfull[..., 0])


def test_window_and_targets_are_consecutive_slices(monkeypatch):
    # A fake plant whose trajectory is its own time index pins the slicing exactly.
    import ftnode.ood_splits as mod

    def fake_simulate(x0, u_of_t, params, t_grid):
        k = np.arange(len(t_grid), dtype=np.float64)
        return np.stack([k, -k], axis=1)

    monkeypatch.setattr(mod, "simulate", fake_simulate)
    tau, L = 5, 7
    ds = make_split(_cfg("far_ring", n_traj=2, tau=tau, L=L))
    k = torch.arange(tau + L + 1, dtype=torch.float32)
    assert torch.equal(ds.W, k[:tau].expand(2, -1))
    assert torch.equal(ds.Y, k[tau:].expand(2, -1))
    assert torch.equal(ds.Xfull[..., 1], -k[tau:].expand(2, -1))


def test_same_seed_same_split_and_seeds_differ():
    a, b = make_split(_cfg("trapped_right", seed=5)), make_split(_cfg("trapped_right", seed=5))
    c = make_split(_cfg("trapped_right", seed=6))
    assert torch.equal(a.W, b.W) and not torch.equal(a.W, c.W)


def test_unknown_region_and_draw_cap():
    with pytest.raises(ValueError):
        make_split(SplitConfig(region="box"))
    with pytest.raises(RuntimeError):
        make_split(_cfg("trapped_right", max_draws=3))
    with pytest.raises(ValueError):
        make_split(_cfg("trapped_right", delta=-0.05))
    with pytest.raises(ValueError):
        make_split(_cfg("half_right", h=0.0))


def test_mirror_negates_every_field():
    ds = make_split(_cfg("half_right", n_traj=4, L=12))
    m = mirror(ds)
    assert torch.equal(m.W, -ds.W) and torch.equal(m.U, -ds.U)


def test_rollout_test_split_accepts_a_split_builder():
    torch.manual_seed(0)
    model = build_cold_unstructured(PhysicalConfig())
    cfg = _cfg("trapped_left", n_traj=4, L=12)
    test_cfgs = [(1000, dataclasses.replace(cfg, seed=1000)), (1001, dataclasses.replace(cfg, seed=1001))]
    qhat, y, per = rollout_test_split(model, test_cfgs, 12, cfg.h, "cpu", make=make_split)
    assert qhat.shape == y.shape == (8, 13)
    assert set(per) == {1000, 1001}
    expect = np.concatenate([make_split(c).Y.numpy() for _, c in test_cfgs])
    assert np.array_equal(y, expect)
