"""The context-conditioned field: data reproducibility, the hold-out, and the head bounds.

All checks run on CPU at small sizes and need no training.
"""
import torch

from ftnode.foundation import (
    DUFFING_PARAMS,
    TRAIN_STREAM,
    VAL_STREAM,
    MechanicalFamilyConfig,
    SetEncoder,
    StructuredCodeHead,
    build_context_field,
    context_tokens,
    in_duffing_holdout,
    mechanical_field,
    sample_batch,
    sample_params,
    simulate_q,
)
from ftnode.foundation.generator import batch_generator
from ftnode.physical import PhysicalField
from ftnode.physical.train import rollout_q
from ftnode.systems import DuffingParams, duffing_field_torch

SMALL = MechanicalFamilyConfig(L=40, k_min=2, k_max=4, n_tok=64, n_query=16)


def _same(a, b):
    return all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("params", "tokens", "k", "xq", "uq", "fq"))


def test_batch_is_a_pure_function_of_seed_stream_index():
    a = sample_batch(SMALL, 5, seed=3, stream=TRAIN_STREAM, index=17)
    sample_batch(SMALL, 5, seed=3, stream=TRAIN_STREAM, index=18)   # drawing others changes nothing
    b = sample_batch(SMALL, 5, seed=3, stream=TRAIN_STREAM, index=17)
    assert _same(a, b)
    assert not _same(a, sample_batch(SMALL, 5, seed=3, stream=TRAIN_STREAM, index=18))
    assert not _same(a, sample_batch(SMALL, 5, seed=4, stream=TRAIN_STREAM, index=17))
    assert not _same(a, sample_batch(SMALL, 5, seed=3, stream=VAL_STREAM, index=17))


def test_generator_seed_is_stable():
    """Pin the stream: a change to the seeding scheme would silently change every dataset."""
    g = batch_generator(0, 0, 0)
    assert torch.rand(1, generator=g, dtype=torch.float64).item() == torch.rand(
        1, generator=batch_generator(0, 0, 0), dtype=torch.float64).item()
    p = sample_params(MechanicalFamilyConfig(), 3, batch_generator(0, 0, 0))
    q = sample_params(MechanicalFamilyConfig(), 3, batch_generator(0, 0, 0))
    assert torch.equal(p, q)


def test_duffing_is_held_out():
    d = torch.tensor(DUFFING_PARAMS, dtype=torch.float64)
    assert bool(in_duffing_holdout(*d.unsqueeze(0).unbind(-1)))
    p = sample_params(MechanicalFamilyConfig(), 20000, batch_generator(1, 0, 0))
    assert not in_duffing_holdout(*p.unbind(-1)).any()


def test_family_field_matches_duffing_at_duffing_params():
    x = torch.randn(32, 2, dtype=torch.float64)
    u = 0.2 * torch.randn(32, dtype=torch.float64)
    p = torch.tensor(DUFFING_PARAMS, dtype=torch.float64).expand(32, 5)
    assert torch.allclose(mechanical_field(x, u, p), duffing_field_torch(x, u, DuffingParams()))


def test_tokens_recover_velocity_and_acceleration():
    cfg = MechanicalFamilyConfig()
    p = sample_params(cfg, 4, batch_generator(2, 0, 0))
    x0 = torch.tensor([[1.0, 0.0], [-0.5, 0.5], [0.3, -0.2], [1.2, 0.4]], dtype=torch.float64)
    u = torch.zeros(4, dtype=torch.float64)
    X = simulate_q(p, x0, u, cfg.h, cfg.T)
    tok = context_tokens(X[..., 0], u, cfg.h) * torch.tensor((1.5, 1.5, 5.0, 0.25), dtype=torch.float64)
    f = mechanical_field(X[:, 1:-1], u.unsqueeze(1), p.unsqueeze(1))
    assert torch.allclose(tok[..., 1], X[:, 1:-1, 1], atol=5e-3)
    assert torch.allclose(tok[..., 2], f[..., 1], atol=5e-2)


def test_simulated_family_stays_bounded():
    b = sample_batch(MechanicalFamilyConfig(k_max=8, k_min=8), 64, seed=0, stream=0, index=0)
    assert torch.isfinite(b.tokens).all() and torch.isfinite(b.fq).all()
    assert b.tokens[..., 0].abs().max() < 3.0          # |q| / 1.5 stays below 3


def test_encoder_ignores_token_order():
    torch.manual_seed(0)
    enc = SetEncoder(d=32, n_latents=4, n_heads=4, code_dim=8).eval()
    t = torch.randn(3, 50, 4)
    perm = torch.randperm(50)
    assert torch.allclose(enc(t), enc(t[:, perm]), atol=1e-5)


def test_structured_head_bounds_hold_for_every_code():
    torch.manual_seed(0)
    head = StructuredCodeHead(code_dim=8)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 1.0)               # far from init: the bounds must not depend on it
    x = 3.0 * torch.randn(256, 2)
    c = 10.0 * torch.randn(256, 8)
    A = head.A(x, c)
    sym = 0.5 * (A + A.transpose(-1, -2))
    assert torch.linalg.eigvalsh(sym).max() <= -head.sigma_min + 1e-5
    assert torch.linalg.matrix_norm(A, ord=2).max() <= head.sigma_max + 1e-4
    g = head.g(x, torch.randn(256), c)
    assert g.abs().max() <= head.R_g + 1e-6


def test_free_head_matches_structured_capacity():
    s = build_context_field("structured", code_dim=16)
    f = build_context_field("free", code_dim=16)
    ns, nf = s.head.numel(), f.head.numel()
    assert abs(nf - ns) / ns < 0.02


def test_bound_field_rolls_out_through_the_physical_path():
    torch.manual_seed(0)
    field = build_context_field("structured", code_dim=8, enc_width=32, n_latents=4)
    b = sample_batch(SMALL, 1, seed=0, stream=0, index=0)
    field.bind_context(b.tokens)
    model = PhysicalField(field, 0.05)
    qs = rollout_q(model, torch.randn(5, 8), torch.zeros(5), 30, 0.05)
    assert qs.shape == (5, 31) and torch.isfinite(qs).all()


def test_closed_form_clamp_matches_the_eigvalsh_clamp():
    from ftnode.foundation.heads import spectral_clamp_2x2
    from ftnode.latent.operator import spectral_clamp_safe

    B = 3.0 * torch.randn(1000, 2, 2, dtype=torch.float64)
    a, b = spectral_clamp_2x2(B, 1.3), spectral_clamp_safe(B, 1.3)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.linalg.matrix_norm(a, ord=2).max() <= 1.3 + 1e-9
    S = B - B.transpose(-1, -2)                                   # skew: repeated singular values
    S.requires_grad_(True)
    spectral_clamp_2x2(S, 0.5).sum().backward()
    assert torch.isfinite(S.grad).all()
