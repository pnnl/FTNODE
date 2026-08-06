"""Latent-geometry and hidden-coordinate-recovery diagnostics."""
import numpy as np
import pytest
import torch

from ftnode.diagnostics import g_image, linear_recovery_r2, pca_2d


@pytest.fixture
def split():
    perm = np.random.default_rng(7).permutation(600)
    return perm[:300], perm[300:]


def test_pca_2d_shapes_and_orthonormal_basis():
    Z = np.random.default_rng(0).normal(size=(500, 4))
    proj, basis = pca_2d(Z)
    assert proj.shape == (500, 2)
    assert basis.shape == (2, 4)
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-8)


def test_pca_2d_recovers_a_planted_plane():
    """Latents confined to a 2-D plane must project back onto it exactly."""
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(400, 2))
    basis = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
    Z = coords @ basis
    proj, _ = pca_2d(Z)
    # distances are preserved up to an orthogonal transform of the plane
    d_in = np.linalg.norm(coords[:50, None] - coords[None, :50], axis=-1)
    d_out = np.linalg.norm(proj[:50, None] - proj[None, :50], axis=-1)
    assert np.allclose(d_in, d_out, atol=1e-8)


def test_linear_recovery_r2_is_one_for_a_linear_target(split):
    fit_i, ev_i = split
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(600, 4))
    target = Z @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.7
    assert linear_recovery_r2(Z, target, fit_i, ev_i) == pytest.approx(1.0, abs=1e-8)


def test_linear_recovery_r2_is_near_zero_for_noise(split):
    fit_i, ev_i = split
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(600, 4))
    target = rng.normal(size=600)
    assert linear_recovery_r2(Z, target, fit_i, ev_i) < 0.15


def test_linear_recovery_r2_is_held_out(split):
    """Scoring on the fit set would hide overfitting; confirm the split is used.

    With as many latent dims as points, an in-sample fit is exact while the
    held-out score is not.
    """
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(40, 39))
    target = rng.normal(size=40)
    fit_i, ev_i = np.arange(20), np.arange(20, 40)
    assert linear_recovery_r2(Z, target, fit_i, ev_i) < 0.99


def test_g_image_respects_the_R_g_box(model_cfg, budget):
    """`g = R_g * tanh(...)` bounds the equilibrium map by construction."""
    from ftnode.latent import build_clamp

    torch.manual_seed(0)
    dyn = build_clamp(model_cfg, budget).dynamics
    Z = (2 * torch.rand(512, budget.m, generator=torch.Generator().manual_seed(0)) - 1) * 2.0
    U = (2 * torch.rand(512, generator=torch.Generator().manual_seed(1)) - 1) * 0.25
    G = g_image(dyn, Z, U)
    assert G.shape == (512, budget.m)
    assert G.abs().max().item() <= model_cfg.equilibrium.R_g + 1e-5
