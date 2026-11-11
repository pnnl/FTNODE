"""Plant registry, plant datasets, and the equilibrium finder of the pH head test.

Fast and training-free: the fields, their critical points, the split rules, the noise
draw, and the Newton finder are all checked against facts known in closed form or by
symmetry.
"""
import importlib.util
import pathlib

import numpy as np
import pytest
import torch

from ftnode.plant_data import PlantDataConfig, add_noise, dataset_hash, make_clean
from ftnode.plants import PLANTS, critical_points, get_plant, separator_output, sinks
from ftnode.systems import DuffingParams, duffing_field

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _field(name, x, u):
    return get_plant(name).field(torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64),
                                 torch.as_tensor(u, dtype=torch.float64)).numpy()


@pytest.mark.parametrize("name,delta", [("duffing", 0.2), ("duffing_light", 0.05)])
def test_duffing_plants_match_the_notebook_field(name, delta):
    rng = np.random.default_rng(0)
    x, u = rng.normal(size=(64, 2)), rng.normal(size=64)
    np.testing.assert_allclose(_field(name, x, u), duffing_field(x, u, DuffingParams(delta)),
                               rtol=0, atol=1e-12)


# (sinks, index-1 points, index-2 points) and the sink outputs at u = 0.
CRIT_U0 = {
    "duffing": ((2, 1, 0), (-1.0, 1.0)),
    "duffing_light": ((2, 1, 0), (-1.0, 1.0)),
    "asym_well": ((2, 1, 0), (-1.2808, 0.7808)),
    "hardening": ((1, 0, 0), (0.0,)),
    "toggle": ((2, 1, 0), (0.2679, 3.7321)),
    "coupled": ((4, 4, 1), (-1.5, -0.3873, 0.3873, 1.5)),
}


@pytest.mark.parametrize("name", sorted(CRIT_U0))
def test_critical_points_at_zero_input(name):
    counts, sink_y = CRIT_U0[name]
    cps = critical_points(get_plant(name), 0.0)
    assert tuple(sum(p.index == k for p in cps) for k in range(3)) == counts
    assert len(cps) == sum(counts)
    np.testing.assert_allclose([p.y for p in cps if p.index == 0], sink_y, atol=1e-4)


@pytest.mark.parametrize("name,ratio", [("duffing", 14.142), ("duffing_light", 56.57),
                                        ("coupled", 15.49)])
def test_sink_sector_ratios(name, ratio):
    """|lambda| / |Re lambda| at the sinks: the quantity the pH sector cap bounds."""
    worst = max(float(np.max(np.abs(p.eig) / np.abs(p.eig.real)))
                for p in sinks(get_plant(name), 0.0))
    assert worst == pytest.approx(ratio, abs=0.01)


@pytest.mark.parametrize("name", ["duffing", "duffing_light", "hardening", "coupled"])
def test_odd_plants_are_odd(name):
    rng = np.random.default_rng(1)
    n = get_plant(name).n
    x, u = rng.normal(size=(32, n)), rng.normal(size=32)
    np.testing.assert_allclose(_field(name, -x, -u), -_field(name, x, u), atol=1e-12)


def test_toggle_swap_symmetry():
    rng = np.random.default_rng(2)
    x, u = rng.uniform(0, 5, size=(32, 2)), rng.normal(size=32)
    np.testing.assert_allclose(_field("toggle", x[:, ::-1], -u), _field("toggle", x, u)[:, ::-1],
                               atol=1e-12)


def test_coupled_keeps_two_sinks_per_side_over_its_training_range():
    plant = get_plant("coupled")
    for u in np.linspace(*plant.u_train, 9):
        ysep = separator_output(plant, u)
        ys = [p.y for p in sinks(plant, u)]
        assert sum(y > ysep for y in ys) == 2 and sum(y < ysep for y in ys) == 2


@pytest.mark.parametrize("name", ["duffing", "coupled"])
def test_one_side_split_holds_over_the_whole_horizon(name):
    plant = get_plant(name)
    for region, sign in (("upper", 1.0), ("lower", -1.0)):
        ds = make_clean(PlantDataConfig(name, region, "val", 0, n_traj=8, L=50))
        y = torch.cat([ds.W_clean, ds.Y_clean], dim=1).double().numpy()
        sep = np.array([separator_output(plant, float(u)) for u in ds.U])
        assert np.all(sign * (y - sep[:, None]) > 0)
        assert np.all(sign * (ds.rest_y.double().numpy() - sep) > 0)


def test_far_and_input_regions():
    # Duffing's far split moves q0 (the first window sample) to |q0| in [2, 4].
    ds = make_clean(PlantDataConfig("duffing", "far", "val", 0, n_traj=32, L=20))
    q0 = ds.W_clean[:, 0].abs().numpy()
    assert np.all((q0 >= 2.0 - 1e-6) & (q0 <= 4.0 + 1e-6))
    ds = make_clean(PlantDataConfig("toggle", "input_fold", "val", 0, n_traj=32, L=20))
    u = np.abs(ds.U.numpy())
    assert np.all((u >= 0.6 - 1e-6) & (u <= 0.8 + 1e-6))
    assert (ds.U > 0).any() and (ds.U < 0).any()


def test_noise_is_deterministic_and_leaves_the_clean_fields():
    cfg = PlantDataConfig("duffing", "box", "train", 0, n_traj=16, L=30)
    ds = make_clean(cfg)
    assert torch.equal(add_noise(ds, cfg, 0.0, 1.0).Y, ds.Y_clean)
    a, b = add_noise(ds, cfg, 0.03, 1.0), add_noise(ds, cfg, 0.03, 1.0)
    assert torch.equal(a.Y, b.Y) and torch.equal(a.W, b.W)
    assert torch.equal(a.Y_clean, ds.Y_clean) and not torch.equal(a.Y, ds.Y_clean)
    c = add_noise(ds, cfg, 0.01, 1.0)
    # Independent draws per level, not one draw rescaled.
    assert not torch.allclose((c.Y - ds.Y_clean) * 3.0, a.Y - ds.Y_clean)


def test_same_config_gives_the_same_dataset():
    cfg = PlantDataConfig("asym_well", "upper", "train", 1, n_traj=16, L=30)
    assert dataset_hash(make_clean(cfg)) == dataset_hash(make_clean(cfg))
    other = PlantDataConfig("asym_well", "upper", "train", 2, n_traj=16, L=30)
    assert dataset_hash(make_clean(cfg)) != dataset_hash(make_clean(other))


def _script():
    spec = importlib.util.spec_from_file_location("ph_head_experiment", ROOT / "scripts" / "ph_head_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _PlantAsModel(torch.nn.Module):
    """The true plant behind the model interface the finder uses (``F`` and ``decode``)."""

    def __init__(self, name):
        super().__init__()
        self.plant = get_plant(name)

    def F(self, z, u):
        return self.plant.field(z, u)

    def decode(self, z):
        return z @ torch.as_tensor(self.plant.c_vec, dtype=z.dtype)


def test_newton_finder_recovers_the_coupled_critical_points():
    script = _script()
    g = torch.Generator().manual_seed(0)
    starts = (2 * torch.rand(512, 4, generator=g, dtype=torch.float64) - 1) * 1.6
    roots = script.newton(_PlantAsModel("coupled"), 0.0, starts, step_cap=0.5)
    true = critical_points(get_plant("coupled"), 0.0)
    assert len(roots) == len(true) == 9
    got = sorted((round(r["y"], 4), r["index"]) for r in roots)
    assert got == sorted((round(p.y, 4), p.index) for p in true)


def test_classification_of_known_rests():
    script = _script()
    crit_y = np.array([[-1.0, 0.0, 1.0] + [np.nan] * 6] * 4)
    crit_ix = np.array([[0, 1, 0] + [-1] * 6] * 4)
    t = np.arange(301) * 0.05
    y = np.stack([np.full(301, 1.0), np.full(301, 0.02), np.sin(t), np.full(301, 2.0)])
    cls, which = script.classify(y, crit_y, crit_ix, sy=1.0)
    assert [script.CLASSES[c] for c in cls] == ["sink", "nonsink", "oscillating", "elsewhere"]
    assert list(which[:2]) == [2, 1]
