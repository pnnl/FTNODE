"""Tests for the rollout metrics that split the factorial score."""
import numpy as np
import pytest

from ftnode.physical.metrics import rollout_metrics


def _damped(n=8, L=600, seed=0):
    """Damped oscillations that settle at +-1; half of them start in the other well."""
    rng = np.random.default_rng(seed)
    t = np.arange(L + 1) * 0.05
    well = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    start = np.where(np.arange(n) % 4 < 2, well, -well)  # every other pair crosses
    amp = (start - well)[:, None] + 0.5 * rng.standard_normal((n, 1))
    return well[:, None] + amp * np.exp(-0.1 * t) * np.cos(1.4 * t)


def test_perfect_prediction_scores_zero_and_every_well_right():
    y = _damped()
    m = rollout_metrics(y, y)
    assert m["mse"] == 0.0 and m["transient_mse"] == 0.0
    assert m["wrong_wells"] == 0 and m["well_acc"] == 1.0
    assert m["osc_energy_ratio"] == pytest.approx(1.0)
    assert m["dom_freq_match"] == 1.0
    assert m["n_crossers"] > 0


def test_a_flipped_well_is_counted_and_carries_the_error():
    y = _damped()
    p = y.copy()
    p[0] = -p[0]
    m = rollout_metrics(p, y)
    assert m["wrong_wells"] == 1
    assert m["wrong_well_error_share"] == pytest.approx(1.0)
    assert m["late_mse_correct_wells"] == 0.0


def test_a_flat_prediction_has_no_oscillation_energy():
    y = _damped()
    p = np.repeat(np.sign(y[:, -1:]), y.shape[1], axis=1)
    m = rollout_metrics(p, y)
    assert m["wrong_wells"] == 0
    assert m["osc_energy_ratio"] == 0.0
    assert m["mse_0-50"] > m["mse_400-601"]


def test_nonfinite_rollouts_count_as_wrong_wells():
    y = _damped()
    p = y.copy()
    p[1, 300:] = np.nan
    m = rollout_metrics(p, y)
    assert m["nonfinite_traj"] == 1 and m["wrong_wells"] == 1
    assert np.isfinite(m["mse"])


def test_short_horizon_returns_the_aggregate_only():
    y = _damped(L=40)
    m = rollout_metrics(y, y)
    assert m["mse"] == 0.0 and "wrong_wells" not in m


def test_horizon_of_exactly_400_steps_gets_the_windowed_metrics():
    y = _damped(L=400)
    assert "wrong_wells" in rollout_metrics(y, y)


def test_nonfinite_or_flat_predictions_never_match_frequency():
    y = _damped()
    p = np.full_like(y, np.nan)
    assert rollout_metrics(p, y)["dom_freq_match"] == 0.0
    flat = np.zeros_like(y)
    assert rollout_metrics(flat, y)["dom_freq_match"] == 0.0


def test_settle_metrics_see_a_wrong_place_on_the_right_side():
    y = -1.0 + 0.3 * np.exp(-0.1 * np.arange(601) * 0.05)[None, :].repeat(4, 0)
    p = y.copy()
    p[0] = -2.0                      # right side (sign), wrong place: the g-box edge
    p[1, 300:] = 1e3                 # blow-up
    m = rollout_metrics(p, y)
    assert m["wrong_wells"] == 1     # the sign rule misses row 0 (and counts row 1)
    assert m["blowups"] == 1
    assert m["settle_near_true"] == 2
    assert m["settle_abs_err_median"] == pytest.approx(0.0)


def test_a_transient_overshoot_that_returns_is_not_a_blowup():
    y = -1.0 + 0.3 * np.exp(-0.1 * np.arange(601) * 0.05)[None, :].repeat(3, 0)
    p = y.copy()
    p[0, 20:40] = 8.0                # far overshoot early, then back on the truth
    p[1, 450:] = 9.0                 # leaves late: a blow-up
    p[2, 100] = np.nan               # non-finite anywhere: a blow-up
    m = rollout_metrics(p, y)
    assert m["blowups"] == 2
