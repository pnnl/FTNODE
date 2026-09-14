"""Tests for the FIM-backed cells (3 and 4).

The unit tests use a mock backbone, so they are fast and need no weights. One
integration test loads the real ``base_model`` and runs only when the checkpoint is
already cached under ``weights/openfim`` (it never triggers a download).
"""
import pathlib

import pytest
import torch
import torch.nn as nn

from ftnode.fim.fields import FreeFIMField, StructuredFIMField

_WEIGHTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "weights" / "openfim" / "base_model" / "checkpoints" / "best-model" / "model.safetensors"
)


class MockBackbone(nn.Module):
    """A stand-in for FIMBackbone: a linear drift and a freezable uncertainty head."""

    def __init__(self, d=2):
        super().__init__()
        self.d = d
        self.fim = nn.Module()
        self.fim.u_model = nn.Linear(d, 1)  # must end up frozen
        self.lin = nn.Linear(d, d)

    def prepare(self, window, h):
        self._prepared = True

    def drift(self, x):
        return self.lin(x)


def test_free_field_starts_at_the_fim_drift():
    torch.manual_seed(0)
    bb = MockBackbone()
    field = FreeFIMField(bb, d=2)
    x = torch.randn(9, 2)
    out = field.F(x, torch.randn(9))
    assert out.shape == (9, 2) and torch.isfinite(out).all()
    # residual is zero-init, so F == drift at init.
    assert torch.allclose(out, bb.drift(x), atol=1e-6)


def test_uncertainty_head_is_frozen_in_both_fim_fields():
    for field in (FreeFIMField(MockBackbone()), StructuredFIMField(MockBackbone())):
        assert all(not p.requires_grad for p in field.backbone.fim.u_model.parameters())


def test_structured_fim_operator_is_negative_definite():
    torch.manual_seed(0)
    field = StructuredFIMField(MockBackbone(), d=2, sigma_min=0.1)
    x = (2 * torch.rand(200, 2) - 1) * 2.0
    A = field.A(x)
    sym = 0.5 * (A + A.transpose(-1, -2))
    assert torch.linalg.eigvalsh(sym).max().item() <= -field.sigma_min + 1e-5


def test_structured_fim_field_value_is_finite():
    torch.manual_seed(0)
    field = StructuredFIMField(MockBackbone(), d=2)
    out = field.F(torch.randn(7, 2), torch.randn(7))
    assert out.shape == (7, 2) and torch.isfinite(out).all()


def test_param_groups_split_backbone_from_head():
    field = StructuredFIMField(MockBackbone(), d=2)
    groups = field.param_groups(lr=3e-3, lr_backbone=5e-6)
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [5e-6, 3e-3]
    # The frozen uncertainty head is in no group.
    frozen = set(id(p) for p in field.backbone.fim.u_model.parameters())
    grouped = {id(p) for g in groups for p in g["params"]}
    assert grouped.isdisjoint(frozen)


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="FIM base_model weights not cached")
def test_real_backbone_drift_is_finite_and_structured_A_is_negative_definite():
    from ftnode.fim import FIMBackbone

    bb = FIMBackbone.from_pretrained(device="cpu")
    field = StructuredFIMField(bb, d=2)
    window = torch.randn(3, 8)
    field.prepare(window, 0.05)
    x = torch.randn(3, 2)
    out = field.F(x, torch.zeros(3))
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    A = field.A(x)
    sym = 0.5 * (A + A.transpose(-1, -2))
    assert torch.linalg.eigvalsh(sym).max().item() <= -field.sigma_min + 1e-4
