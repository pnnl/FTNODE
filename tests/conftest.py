"""Shared fixtures.

Everything here runs on CPU at small sizes and needs no training: the invariants
under test are algebraic identities that hold at initialization, which is why the
whole suite is fast.
"""
import pytest
import torch

from ftnode.latent import KappaBudget, LatentModelConfig, build_clamp


@pytest.fixture
def budget():
    """The kappa budget the duffing notebooks train with."""
    return KappaBudget(sigma_min=0.1, kappa_max=25.0, skew_frac=0.6, m=4)


@pytest.fixture
def model_cfg():
    return LatentModelConfig()


@pytest.fixture
def plant(model_cfg, budget):
    """A frozen plant around a randomly-initialized SVD-clamp model.

    The identities under test hold for any theta, trained or not, so no
    checkpoint and no training is involved.
    """
    from ftnode.control import FrozenLatentPlant

    torch.manual_seed(0)
    return FrozenLatentPlant(build_clamp(model_cfg, budget))


@pytest.fixture
def latent_box(budget):
    """4000 points sampled uniformly from the latent box [-z_scale, z_scale]^m."""
    g = torch.Generator().manual_seed(0)
    return (2 * torch.rand(4000, budget.m, generator=g) - 1) * 2.0


@pytest.fixture
def small_batch(budget):
    """A small (z, u) pair for the derivative checks."""
    g = torch.Generator().manual_seed(0)
    z = (2 * torch.rand(7, budget.m, generator=g) - 1) * 2.0
    u = (2 * torch.rand(7, 1, generator=g) - 1) * 0.25
    return z, u
