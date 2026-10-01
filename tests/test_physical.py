"""Algebraic-at-init checks for the physical-state cells (the FIM factorial's cold row).

Fast and checkpoint-free, like the rest of the suite: every property here holds for
any parameters, trained or not.  The FIM cells (3 and 4) are covered separately once
the OpenFIM backbone is vendored.
"""
import torch

from ftnode.physical import PhysicalField, StructuredField, UnstructuredField, estimate_x0
from ftnode.physical.config import (
    PhysicalConfig,
    build_cold_structured,
    build_cold_unstructured,
)
from ftnode.physical.train import rollout_q


def test_estimate_x0_recovers_a_linear_ramp():
    # q(t) = a + b*t on the window grid t = 0..(tau-1)*h. The first rollout time is
    # tau*h, so q0 must be the extrapolation a + b*tau*h, and q_dot0 must be b.
    h, tau, b = 0.05, 8, 1.3
    a = -0.4
    t = torch.arange(tau) * h
    window = (a + b * t).unsqueeze(0)  # (1, tau)
    x0 = estimate_x0(window, h)
    assert x0.shape == (1, 2)
    assert torch.allclose(x0[0, 1], torch.tensor(b), atol=1e-5)          # q_dot0 = slope
    assert torch.allclose(x0[0, 0], torch.tensor(a + b * tau * h), atol=1e-5)  # target-time q0


def test_estimate_x0_is_batched_and_two_dimensional():
    w = torch.randn(16, 8)
    x0 = estimate_x0(w, 0.05)
    assert x0.shape == (16, 2)
    assert torch.isfinite(x0).all()


def test_unstructured_field_shape_and_finiteness():
    f = UnstructuredField(m=2, q=1)
    x = torch.randn(10, 2)
    u = torch.randn(10)  # (b,) input, as the dataset stores it
    out = f.F(x, u)
    assert out.shape == (10, 2)
    assert torch.isfinite(out).all()


def test_structured_operator_is_negative_definite_by_construction():
    # sym(A) <= -sigma_min I at every x, for any parameters. Check the eigenvalues of
    # the symmetric part are all at most -sigma_min.
    cfg = PhysicalConfig()
    torch.manual_seed(0)
    field = build_cold_structured(cfg).field
    assert isinstance(field, StructuredField)
    x = (2 * torch.rand(200, 2) - 1) * 2.0
    A = field.A(x)
    sym = 0.5 * (A + A.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(sym)
    assert eig.max().item() <= -cfg.sigma_min + 1e-5


def test_structured_field_value_is_finite():
    cfg = PhysicalConfig()
    torch.manual_seed(0)
    field = build_cold_structured(cfg).field
    x = torch.randn(10, 2)
    u = torch.randn(10)
    out = field.F(x, u)
    assert out.shape == (10, 2)
    assert torch.isfinite(out).all()


def test_readout_is_the_first_state():
    x = torch.randn(5, 2)
    assert torch.equal(PhysicalField.readout(x), x[..., 0])


def test_rollout_shape_for_both_cold_cells():
    cfg = PhysicalConfig()
    L = 12
    w = torch.randn(6, cfg.tau)
    u = torch.randn(6)
    for build in (build_cold_unstructured, build_cold_structured):
        torch.manual_seed(0)
        model = build(cfg)
        qs = rollout_q(model, w, u, L, cfg.h)
        assert qs.shape == (6, L + 1)
        assert torch.isfinite(qs).all()


def test_structured_budget_caps_condition_number():
    # kappa(A) <= kappa_max by construction.
    cfg = PhysicalConfig()
    torch.manual_seed(0)
    field = build_cold_structured(cfg).field
    x = (2 * torch.rand(200, 2) - 1) * 2.0
    A = field.A(x)
    sv = torch.linalg.svdvals(A)
    sym = 0.5 * (A + A.transpose(-1, -2))
    lam_min = (-torch.linalg.eigvalsh(sym)).min(dim=-1).values  # >= sigma_min
    kappa = sv.max(dim=-1).values / lam_min
    assert kappa.max().item() <= cfg.kappa_max + 1e-3


def test_fim_head_budgets_are_matched():
    # The cell-3 free head is sized to the cell-4 structured head's parameter count,
    # so the warm-row interaction isolates structure, not capacity.
    from ftnode.physical.config import (
        _matched_free_hidden,
        _mlp_numel,
        _structured_head_numel,
    )

    cfg = PhysicalConfig()
    struct = _structured_head_numel(cfg)
    free = _mlp_numel(2 * cfg.m, cfg.m, _matched_free_hidden(cfg), cfg.depth)
    assert abs(free - struct) / struct < 0.02


def test_fim_mlp_head_budget_is_matched():
    # The cell-5 head reads [drift, x, u] and is sized to the cell-4 head's parameter count.
    from ftnode.physical.config import (
        _matched_free_hidden,
        _mlp_numel,
        _structured_head_numel,
    )

    cfg = PhysicalConfig()
    in_dim = 2 * cfg.m + cfg.q
    struct = _structured_head_numel(cfg)
    head = _mlp_numel(in_dim, cfg.m, _matched_free_hidden(cfg, in_dim=in_dim), cfg.depth)
    assert abs(head - struct) / struct < 0.02


def test_cell3_head_width_is_unchanged():
    # The cell-3 width feeds the committed checkpoints; the in_dim default must keep it.
    from ftnode.physical.config import _matched_free_hidden

    cfg = PhysicalConfig()
    assert _matched_free_hidden(cfg) == _matched_free_hidden(cfg, in_dim=2 * cfg.m)


def test_same_seed_reproduces_the_structured_build():
    cfg = PhysicalConfig()
    torch.manual_seed(0)
    a = build_cold_structured(cfg)
    torch.manual_seed(0)
    b = build_cold_structured(cfg)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
