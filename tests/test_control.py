"""The four numerical invariants the control stage rests on.

These were the `if __name__ == '__main__'` checks at the bottom of
`examples/duffing/_proto_ctrl.py`. They are promoted here because they are the
actual acceptance criteria for the control design, not incidental debug output.

All four are algebraic identities in theta and psi, so they hold at random
initialization -- no checkpoint, no training.
"""
import numpy as np
import pytest
import torch

from ftnode.control import (
    ControlConfig,
    SplitOperator,
    g_psi,
    grad_u_J,
    model_closed_loop,
    sat_u,
    udot_dir,
)
from ftnode.diagnostics import op_stats


@pytest.fixture
def fpsi(budget):
    torch.manual_seed(1)
    return SplitOperator.from_budget(budget)


def test_DuF_jvp_matches_the_split_form(plant, small_batch):
    """Check 1/4: the two routes to D_u F_theta must agree.

    `A_theta` does not depend on u, so the forward-mode jvp and the closed form
    `-A(z) dg/du` are the same quantity computed two ways. Disagreement would mean
    the split structure is not what the model actually implements.
    """
    z, u = small_batch
    assert torch.allclose(plant.DuF(z, u), plant.DuF_closed(z, u), atol=1e-5)


def test_gauge_reproduction_is_exact(plant, fpsi, small_batch):
    """Check 2/4: `f_psi (z - g_psi) == F_theta` for a RANDOM psi.

    This is the load-bearing identity of the whole approach: the splitting is a
    gauge choice, so training psi cannot move the represented plant. If this ever
    fails, control training is silently altering the identified dynamics.
    """
    z, u = small_batch
    gp = g_psi(fpsi, plant, z, u)
    lhs = torch.einsum("bij,bj->bi", fpsi(z), z - gp)
    rhs = plant.F(z, u)
    assert torch.allclose(lhs, rhs, atol=1e-5)
    assert ((lhs - rhs).norm() / rhs.norm()).item() < 1e-5


def test_kappa_of_fpsi_is_capped(fpsi, budget, latent_box):
    """Check 3/4: kappa(f_psi) <= kappa_max over the latent box, by construction."""
    maxre, _, kappa = op_stats(fpsi, latent_box)
    assert kappa.max() <= fpsi.kappa_bound + 1e-4
    assert fpsi.kappa_bound == pytest.approx(budget.kappa_max)
    assert maxre.max() <= -budget.sigma_min + 1e-4


def test_grad_u_J_matches_autograd(plant, fpsi, small_batch):
    """Check 4/4: the analytic grad_u J against autograd on J directly.

    `udot_dir` carries a sign flip that is easy to get wrong (the minus of
    `-eta grad_u J` is folded into the expression); this pins it.
    """
    z, u = small_batch
    z_target = torch.zeros(plant.m)

    def J_of_u(uu):
        gp = g_psi(fpsi, plant, z, uu)
        return 0.5 * ((gp - z_target) ** 2).sum(-1).sum()

    u_ = u.clone().requires_grad_(True)
    auto = torch.autograd.grad(J_of_u(u_), u_)[0]
    analytic = grad_u_J(fpsi, plant, z, u, z_target)
    assert torch.allclose(auto, analytic, atol=1e-5)
    assert torch.allclose(analytic, -udot_dir(fpsi, plant, z, u, z_target))


def test_sat_u_stays_in_the_admissible_set():
    """Admissibility holds by parameterization, for every control state."""
    w = torch.linspace(-50, 50, 401).unsqueeze(-1)
    u = sat_u(w, -0.5, 0.5)
    assert u.min() >= -0.5 and u.max() <= 0.5
    assert torch.isfinite(u).all()


def test_split_operator_loads_from_budget(budget):
    fpsi = SplitOperator.from_budget(budget)
    assert fpsi.c_P == pytest.approx(budget.c_P)
    assert fpsi.c_K == pytest.approx(budget.c_K)
    assert fpsi.sigma_max == pytest.approx(budget.sigma_max)


def test_truncated_bptt_keeps_gradients_finite(plant, fpsi):
    """The k_trunc window is what keeps design-time gradients from exploding.

    Full backprop through a long loop drives gradient norms to ~1e17-1e19; this
    asserts the truncated path stays finite and bounded over the same horizon.
    """
    cfg = ControlConfig()
    z0 = 0.35 * torch.randn(4, plant.m, generator=torch.Generator().manual_seed(0))
    z_star = torch.zeros(plant.m)
    zs, us = model_closed_loop(
        fpsi, plant, z0, z_star, 60, cfg.h, cfg.eta, cfg.u_lo, cfg.u_hi, k_trunc=cfg.k_trunc
    )
    (zs - z_star).pow(2).sum().backward()
    gn = torch.nn.utils.clip_grad_norm_(fpsi.parameters(), float("inf"))
    assert torch.isfinite(gn), "design-time gradient went non-finite"
    assert us.abs().max().item() <= cfg.u_bound + 1e-6


def test_control_config_authority_exceeds_identification_range():
    """u_bound must not silently inherit the identification excitation range.

    The reachability study stabilizes 19/39 initial conditions at |u| <= 0.25 but
    39/39 at |u| <= 0.5, so defaulting to the ID range ships a controller that
    cannot do the task.
    """
    cfg = ControlConfig()
    assert cfg.u_bound > 0.25
    assert (cfg.u_lo, cfg.u_hi) == (-cfg.u_bound, cfg.u_bound)


def test_importing_control_has_no_side_effects():
    """The prototypes loaded a checkpoint and built a dataset at import time."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import ftnode.control; print('ok')"],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok", f"import printed extra output: {out.stdout!r}"
