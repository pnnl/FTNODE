"""The linear-skip encoder: it wraps an Encoder without moving the global RNG, reproduces
it when the linear term is zero, and lets latents grow with the window far out."""
import torch

from ftnode.latent import Encoder, LinearSkipEncoder


def _encoder():
    torch.manual_seed(0)
    return Encoder(8, 4)


def test_wrapping_draws_no_global_random_numbers():
    enc = _encoder()
    state = torch.get_rng_state()
    LinearSkipEncoder.from_encoder(enc, z_scale=0.5, seed=3)
    assert torch.equal(state, torch.get_rng_state())


def test_zero_linear_term_reproduces_the_encoder():
    enc = _encoder()
    skip = LinearSkipEncoder.from_encoder(enc)
    with torch.no_grad():
        skip.skip.weight.zero_()
    w = torch.randn(32, 8)
    torch.testing.assert_close(skip(w), enc(w), rtol=0, atol=0)


def test_latent_grows_linearly_with_the_window_far_out():
    enc = _encoder()
    skip = LinearSkipEncoder.from_encoder(enc, z_scale=0.5, seed=1)
    w = torch.randn(16, 8)
    with torch.no_grad():
        near, far = skip(10 * w).norm(dim=1), skip(100 * w).norm(dim=1)
        bounded = enc(100 * w).norm(dim=1)
    # The bounded term contributes at most z_scale * sqrt(m) = 1 to the norm.
    assert torch.all((far / near - 10).abs() < 0.2 * 10)
    assert torch.all(bounded <= 2.0 * 4 ** 0.5 + 1e-6)


def test_quartic_confinement_is_the_gradient_of_its_energy():
    """g = z - grad H with H = |z|^2/2 + (eps/4)|z|^4 - Phi, so F = A grad H still holds."""
    from ftnode.latent import GradPotentialG, QuarticConfinement

    torch.manual_seed(0)
    inner = GradPotentialG(m=4, q=1, cap=False)
    state = torch.get_rng_state()
    g = QuarticConfinement(inner, eps=0.1, learn=True)
    assert torch.equal(state, torch.get_rng_state())
    z, u = torch.randn(16, 4, requires_grad=True), torch.randn(16)
    H = 0.5 * (z * z).sum(-1) - g.phi(z, u)
    gradH = torch.autograd.grad(H.sum(), z)[0]
    torch.testing.assert_close(z - g(z, u), gradH, rtol=1e-5, atol=1e-5)
    assert abs(float(g.eps) - 0.1) < 1e-6
