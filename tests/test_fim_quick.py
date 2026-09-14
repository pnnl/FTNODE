"""Tests for the quick FIM-retune helpers (``ftnode.fim.quick``).

Mocked backbone, so fast and weight-free -- the same stand-in pattern as
``tests/test_fim_backbone.py``.  These check the routines run, stay finite, and preserve
the by-construction invariants; they do not train the real 13M model.
"""
import torch
import torch.nn as nn

from ftnode.fim import quick
from ftnode.fim.fields import FreeFIMField, StructuredFIMField
from ftnode.systems import DuffingDataConfig, make_dataset


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
