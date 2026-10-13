"""Tests for the quick FIM-retune helpers (``ftnode.fim.quick``).

Mocked backbone, so fast and weight-free -- the same stand-in pattern as
``tests/test_fim_backbone.py``.  These check the routines run, stay finite, and preserve
the by-construction invariants; they do not train the real 13M model.
"""
import pytest
import torch
import torch.nn as nn

from ftnode.fim import quick
from ftnode.fim.fields import FreeFIMField, StructuredFIMField
from ftnode.systems import DuffingDataConfig, DuffingDataset, make_dataset


class MockBackbone(nn.Module):
    """A stand-in for FIMBackbone: a linear drift and a freezable uncertainty head."""

    def __init__(self, d=2):
        super().__init__()
        self.d = d
        self.fim = nn.Module()
        self.fim.u_model = nn.Linear(d, 1)  # must stay frozen
        self.lin = nn.Linear(d, d)
        self.prepared_shape = None

    def prepare(self, window, h):
        self.prepared_shape = tuple(window.shape)

    def drift(self, x):
        return self.lin(x)


def _ds():
    return make_dataset(DuffingDataConfig(n_traj=4, L=12, tau=8, h=0.05, seed=0))


def _params():
    return DuffingDataConfig().params


def test_drift_mse_vs_true_is_finite_and_per_traj():
    ds = _ds()
    out = quick.drift_mse_vs_true(MockBackbone(), ds, 0.05, _params())
    assert torch.isfinite(torch.tensor(out["mean"])) and out["mean"] >= 0.0
    assert len(out["per_traj"]) == 4


def test_retune_short_rollout_runs_and_stays_finite():
    torch.manual_seed(0)
    field = FreeFIMField(MockBackbone(), d=2)
    cfg = quick.RetuneConfig(epochs=3, L_short=4, batch=2, lr=1e-3, lr_backbone=1e-4)
    field, losses = quick.retune_short_rollout(field, _ds(), 0.05, cfg)
    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)


def test_distillation_trains_head_and_keeps_A_negative_definite():
    torch.manual_seed(0)
    bb = MockBackbone()
    field = StructuredFIMField(bb, d=2, sigma_min=0.1)
    field, losses = quick.distill_structured(field, bb, _ds(), 0.05, quick.DistillConfig(epochs=3, batch=8))
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    # A stays negative definite by construction throughout the fit.
    x = (2 * torch.rand(64, 2) - 1) * 2.0
    A = field.A(x)
    sym = 0.5 * (A + A.transpose(-1, -2))
    assert torch.linalg.eigvalsh(sym).max().item() <= -field.sigma_min + 1e-5
    # Distillation freezes the whole backbone; only the head trained.
    assert all(not p.requires_grad for p in field.backbone.parameters())


def test_grid_eval_splits_in_and_out_region():
    field = FreeFIMField(MockBackbone(), d=2)
    g = quick.drift_mse_on_grid(field, _ds(), 0.05, _params(), n=5, scale=2.0)
    assert g["in_points"] > 0 and g["out_points"] > 0
    assert torch.isfinite(torch.tensor(g["in_region_mse"]))
    assert torch.isfinite(torch.tensor(g["out_region_mse"]))


def test_short_horizon_qmse_is_finite():
    field = FreeFIMField(MockBackbone(), d=2)
    assert torch.isfinite(torch.tensor(quick.short_horizon_qmse(field, _ds(), 0.05, 4)))


def test_set_backbone_trainable_keeps_uncertainty_head_frozen():
    field = StructuredFIMField(MockBackbone(), d=2)
    quick.set_backbone_trainable(field, True)
    assert all(not p.requires_grad for p in field.backbone.fim.u_model.parameters())
    assert all(p.requires_grad for p in field.backbone.lin.parameters())
    quick.set_backbone_trainable(field, False)
    assert all(not p.requires_grad for p in field.backbone.lin.parameters())


def test_context_comes_from_the_window_only():
    """The distillation target is built from the measured q-window, never the true state."""
    ds = _ds()
    bb = MockBackbone()
    quick.distill_structured(StructuredFIMField(bb, d=2), bb, ds, 0.05, quick.DistillConfig(epochs=1, batch=8))
    assert bb.prepared_shape == tuple(ds.W.shape)  # (n_traj, tau) -- the window, not Y/Xfull


def test_reconstructed_states_track_the_true_state():
    ds = _ds()
    X = quick.reconstruct_states(ds, 0.05)
    assert X.shape == ds.Xfull.shape
    assert torch.equal(X[..., 0], ds.Y)                       # q is measured exactly
    assert (X[..., 1] - ds.Xfull[..., 1]).abs().max() < 0.02  # q_dot by central difference


def test_distillation_never_reads_the_true_state():
    """Poisoning Xfull must not change the distillation: training uses measured q only."""
    ds = _ds()
    poisoned = DuffingDataset(W=ds.W, U=ds.U, Y=ds.Y, Xfull=torch.full_like(ds.Xfull, float("nan")))
    torch.manual_seed(0)
    bb = MockBackbone()
    _, losses = quick.distill_structured(StructuredFIMField(bb, d=2), bb, poisoned, 0.05,
                                         quick.DistillConfig(epochs=2, batch=8))
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)


def test_drift_targets_match_the_true_field_on_clean_data():
    from ftnode.systems import duffing_field_torch

    ds = make_dataset(DuffingDataConfig(n_traj=4, L=40, tau=8, h=0.05, seed=0))
    X, Xdot = quick.drift_targets(ds, 0.05)
    assert X.shape == Xdot.shape == (4, 40, 2)
    true = duffing_field_torch(ds.Xfull[:, :-1].reshape(-1, 2), ds.U.repeat_interleave(40), _params())
    assert (Xdot.reshape(-1, 2) - true).abs().max() < 0.05


def test_drift_match_never_reads_the_true_state_and_lowers_the_loss():
    from ftnode.physical import UnstructuredField

    ds = make_dataset(DuffingDataConfig(n_traj=8, L=30, tau=8, h=0.05, seed=0))
    poisoned = DuffingDataset(W=ds.W, U=ds.U, Y=ds.Y, Xfull=torch.full_like(ds.Xfull, float("nan")))
    torch.manual_seed(0)
    field = UnstructuredField(m=2, q=1)
    _, losses = quick.retune_drift_match(field, poisoned, 0.05,
                                         quick.DriftMatchConfig(epochs=40, batch=8, n_times=8))
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < 0.5 * losses[0]


def test_drift_match_respects_a_frozen_backbone():
    torch.manual_seed(0)
    bb = MockBackbone()
    field = StructuredFIMField(bb, d=2)
    quick.set_backbone_trainable(field, False)
    before = [p.detach().clone() for p in bb.lin.parameters()]
    quick.retune_drift_match(field, _ds(), 0.05, quick.DriftMatchConfig(epochs=2, batch=2, n_times=3))
    assert all(torch.equal(a, b) for a, b in zip(before, bb.lin.parameters()))
    assert bb.prepared_shape == (2, 8)  # context rebound per minibatch, from the window



class LayeredMockBackbone(nn.Module):
    """A stand-in with FIM's block names and a dropout on the drift path."""

    def __init__(self, d=2):
        super().__init__()
        self.d = d
        self.fim = nn.Module()
        self.fim.trajectory_encoder = nn.Linear(d, d)
        self.fim.location_proj = nn.Linear(d, d)
        self.fim.functional_decoder = nn.Linear(d, d)
        self.fim.u_model = nn.Linear(d, 1)
        self.drop = nn.Dropout(0.5)

    def prepare(self, window, h):
        pass

    def drift(self, x):
        f = self.fim
        return f.functional_decoder(self.drop(f.location_proj(x) + f.trajectory_encoder(x)))


def _trainable(module):
    return all(p.requires_grad for p in module.parameters())


def _frozen(module):
    return all(not p.requires_grad for p in module.parameters())


def test_retune_scopes_select_the_backbone_blocks():
    field = StructuredFIMField(LayeredMockBackbone(), d=2)
    fim = field.backbone.fim
    quick.set_retune_scope(field, "R0")
    assert _frozen(field.backbone) and _trainable(field.L_net)
    quick.set_retune_scope(field, "R1")
    assert _trainable(fim.location_proj) and _trainable(fim.functional_decoder)
    assert _frozen(fim.trajectory_encoder) and _frozen(fim.u_model)
    quick.set_retune_scope(field, "R2")
    assert _trainable(fim.trajectory_encoder) and _trainable(fim.functional_decoder)
    assert _frozen(fim.u_model)
    with pytest.raises(ValueError):
        quick.set_retune_scope(field, "R3")


def test_backbone_dropout_is_off_in_retune_and_metrics():
    torch.manual_seed(0)
    bb = LayeredMockBackbone()
    assert bb.training  # the pretrained FIM also loads in training mode
    field = StructuredFIMField(bb, d=2)
    quick.set_retune_scope(field, "R2")
    quick.retune_drift_match(field, _ds(), 0.05, quick.DriftMatchConfig(epochs=1, batch=4, n_times=2))
    assert not bb.training
    bb.train()
    a = quick.drift_mse_vs_true(bb, _ds(), 0.05, _params())["mean"]
    b = quick.drift_mse_vs_true(bb, _ds(), 0.05, _params())["mean"]
    assert a == b  # deterministic: no dropout noise in the metric


def test_start_windows_default_is_the_context_window():
    ds = _ds()
    idx = torch.arange(4)
    w, y = quick._start_windows(ds, idx, 3, random_start=False)
    assert torch.equal(w, ds.W) and torch.equal(y, ds.Y[:, :4])


def test_random_start_windows_are_consecutive_slices_of_the_record():
    ds = _ds()
    q = torch.cat([ds.W, ds.Y], dim=1)
    torch.manual_seed(0)
    w, y = quick._start_windows(ds, torch.arange(4), 3, random_start=True)
    tau = ds.W.shape[1]
    for i in range(4):
        # Locate the window in the record; the targets must follow it directly.
        starts = [k for k in range(tau, q.shape[1] - 3) if torch.equal(q[i, k - tau : k], w[i])]
        assert starts and torch.equal(q[i, starts[0] : starts[0] + 4], y[i])


def test_short_rollout_retune_runs_on_a_cold_field_with_random_starts():
    from ftnode.physical.config import PhysicalConfig, build_cold_structured

    torch.manual_seed(0)
    field = build_cold_structured(PhysicalConfig()).field
    _, losses = quick.retune_short_rollout(field, _ds(), 0.05,
                                           quick.RetuneConfig(epochs=3, batch=4, random_start=True))
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(x)) for x in losses)


def test_on_epoch_hook_runs_once_per_epoch():
    seen = []
    field = FreeFIMField(MockBackbone(), d=2)
    quick.retune_short_rollout(field, _ds(), 0.05, quick.RetuneConfig(epochs=3, batch=4),
                               on_epoch=lambda ep, loss: seen.append(ep))
    quick.retune_drift_match(field, _ds(), 0.05, quick.DriftMatchConfig(epochs=2, batch=4, n_times=2),
                             on_epoch=lambda ep, loss: seen.append(ep))
    assert seen == [0, 1, 2, 0, 1]
