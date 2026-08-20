"""The eigvalsh clamp must agree with the SVD clamp it replaces.

`spectral_clamp_safe` exists because the SVD in `spectral_clamp` fails to
converge on skew inputs during psi-training. That is only a legitimate swap if
the two compute the same projection everywhere the SVD does work -- which is what
these tests pin down, on the four regimes the control notebook identified.
"""
import pytest
import torch

from ftnode.latent import spectral_clamp, spectral_clamp_safe

CAP = 1.44  # c_K at the notebook budget


def _cases(m=4, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Bs = torch.randn(n, m, m, generator=g)
    v = torch.randn(n, m, 1, generator=g)
    return {
        "generic": torch.randn(n, m, m, generator=g),
        "skew": Bs - Bs.transpose(1, 2),  # paired singular values: the failure mode
        "low-rank": v @ v.transpose(1, 2),  # repeated zero singular values
        "tiny": 1e-6 * torch.randn(n, m, m, generator=g),
    }


@pytest.mark.parametrize("name", ["generic", "skew", "low-rank", "tiny"])
def test_safe_clamp_matches_svd_clamp(name):
    B = _cases()[name]
    assert torch.allclose(spectral_clamp(B, CAP), spectral_clamp_safe(B, CAP), atol=1e-5)


@pytest.mark.parametrize("name", ["generic", "skew", "low-rank", "tiny"])
def test_cap_is_respected(name):
    """The projection is exact, not merely conservative: no output exceeds the cap."""
    out = spectral_clamp_safe(_cases()[name], CAP)
    assert torch.linalg.matrix_norm(out, ord=2).max().item() <= CAP + 1e-5


def test_cap_is_tight_for_a_scaled_input():
    """A matrix well above the cap comes back sitting exactly on it."""
    g = torch.Generator().manual_seed(1)
    B = 50.0 * torch.randn(32, 4, 4, generator=g)
    smax = torch.linalg.matrix_norm(spectral_clamp_safe(B, CAP), ord=2)
    assert torch.allclose(smax, torch.full_like(smax, CAP), atol=1e-4)


def test_small_caps_stay_exact():
    """Caps below 1e-4 are handled exactly.

    The prototype computed `sqrt(lam + 1e-8)`, flooring the norm estimate at 1e-4
    and silently over-shrinking below that cap. `clamp_min` on the eigenvalue
    keeps the projection exact.
    """
    g = torch.Generator().manual_seed(2)
    B = torch.randn(16, 4, 4, generator=g)
    cap = 1e-6
    smax = torch.linalg.matrix_norm(spectral_clamp_safe(B, cap), ord=2)
    assert torch.allclose(smax, torch.full_like(smax, cap), rtol=1e-3)


def test_rank_agnostic_batch_shape():
    """Works on batch shapes other than 3-D; the prototype hardcoded `.view(-1, 1, 1)`."""
    g = torch.Generator().manual_seed(3)
    B = torch.randn(3, 5, 4, 4, generator=g)
    out = spectral_clamp_safe(B, CAP)
    assert out.shape == B.shape
    assert torch.linalg.matrix_norm(out, ord=2).max().item() <= CAP + 1e-5


def test_gradient_is_finite_on_degenerate_input():
    """A vanishing matrix must not produce a non-finite gradient."""
    B = torch.zeros(4, 4, 4, requires_grad=True)
    spectral_clamp_safe(B, CAP).sum().backward()
    assert torch.isfinite(B.grad).all()
