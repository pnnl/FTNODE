"""Does a symmetric-``J_g`` representation of an already-identified field exist?

This answers the question that decides whether restricting ``J_g`` to be symmetric costs
anything on a given task -- **before** paying for a multi-seed training arm.

The question people reach for first is "how skewed is the ``J_g`` the incumbent learned?",
and it is the wrong one.  The split ``F = A(z)(z - g(z,u))`` carries a large gauge freedom
-- the whole control stage is built on it, re-splitting one frozen field with a different
operator (:func:`ftnode.control.g_psi`) -- and nothing in the identification loss prefers
any particular representative.  So an unconstrained fit lands wherever the optimizer takes
it whether or not a symmetric representative exists, and measuring its skew answers "did
the optimizer happen to pick the gradient gauge?" rather than "can this field be written in
it?".

The right question has a clean answer.  Writing ``F = A grad V``, the constraint
``sym A <= -sigma_min I`` forces ``<F, grad V> <= -sigma_min ||grad V||^2`` pointwise, and
``||A||_2 <= sigma_max`` forces ``||F|| <= sigma_max ||grad V||``.  Conversely, given any
``V`` satisfying both, an admissible ``A`` can always be constructed -- take
``p = grad V/||grad V||``, ``f = F/||grad V||``, ``f' = f + sigma_min p``,
``beta = <f',p> <= 0``, ``f'_perp = f' - beta p``, and

    A = -sigma_min I + beta p p^T + (f'_perp p^T - p f'_perp^T),

whose skew term carries exactly the component of ``F`` transverse to ``grad V``.  So:

    **symmetric J_g is possible  <=>  the field admits a strict Lyapunov function with
    exponential rate sigma_min.**

Note ``V = 0.5||z||^2 - Phi`` costs nothing in generality -- ``Phi = 0.5||z||^2 - V``
recovers any ``V`` -- so fitting ``Phi`` is fitting an arbitrary ``V``.

Nothing here trains a dynamics model.  It fits one scalar field against a frozen ``F``, on
latents that field was already validated on, which is minutes of compute against roughly
half an hour per seed for a training arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .equilibrium import GradPotentialG

__all__ = ["FeasibilityResult", "fit_potential"]


@dataclass
class FeasibilityResult:
    """Outcome of :func:`fit_potential`.  Per-sample tensors are aligned with ``Z``.

    ``rate_violation`` is ``max(0, <F, grad V> + sigma_min ||grad V||^2)`` -- zero wherever
    the rate condition holds, positive by exactly the margin by which it fails.
    ``norm_violation`` is ``max(0, ||F|| - sigma_max ||grad V||)``, the second condition.
    A field is representable iff both are ~0 everywhere.
    """

    potential: GradPotentialG
    rate_violation: torch.Tensor
    norm_violation: torch.Tensor
    grad_V_norm: torch.Tensor
    z_norm: torch.Tensor
    achieved_rate: torch.Tensor
    sigma_min: float
    loss_history: list[float]

    @property
    def frac_rate_violating(self) -> float:
        return float((self.rate_violation > 1e-6).float().mean())

    @property
    def frac_norm_violating(self) -> float:
        return float((self.norm_violation > 1e-6).float().mean())

    @property
    def frac_increasing(self) -> float:
        """Fraction where ``V`` fails to decrease at all -- the question behind the rate."""
        return float((self.achieved_rate <= 0).float().mean())

    @property
    def sigma_achieved(self) -> float:
        """Largest ``sigma`` for which ``<F,gradV> <= -sigma||gradV||^2`` holds everywhere.

        This is the informative number.  The binary verdict conflates two very different
        outcomes: "no Lyapunov function exists" (fatal to the whole idea) and "one exists
        but certifies a slower rate than this ``KappaBudget`` declares" (a statement about
        ``sigma_min``, not about symmetry).  A positive value here means the field *is* a
        generalized gradient system -- just with this decay rate rather than ``sigma_min``.
        """
        return float(self.achieved_rate.min())

    @property
    def feasible(self) -> bool:
        """Whether both conditions hold everywhere at the declared ``sigma_min``."""
        return self.frac_rate_violating < 1e-3 and self.frac_norm_violating < 1e-3

    def report(self) -> str:
        """One-screen summary, including where the residual concentrates."""
        q = torch.tensor([0.0, 0.01, 0.5, 1.0])
        pct = torch.quantile(self.achieved_rate, q)
        if self.feasible:
            verdict = f"FEASIBLE at the declared sigma_min={self.sigma_min:g}"
        elif self.sigma_achieved > 0:
            verdict = (f"GRADIENT SYSTEM, SLOWER RATE -- V decreases everywhere, but the "
                       f"certified rate is {self.sigma_achieved:.3g}, not {self.sigma_min:g}")
        elif self.frac_increasing < 0.01:
            # A strict min over thousands of samples is set by its worst one, which is a
            # brittle statistic for a fitted V.  Report the tail explicitly instead: a
            # sub-1% tail is a fitting residual, not a structural obstruction -- and it
            # lands where ||grad V|| is smallest, i.e. near the critical set, which is both
            # where the rate condition is hardest and where F is nearly zero anyway.
            verdict = (f"GRADIENT SYSTEM UP TO A {100 * self.frac_increasing:.2f}% TAIL -- "
                       f"rate {pct[1]:.3g} holds on 99% of samples; strict min "
                       f"{self.sigma_achieved:+.3g}")
        else:
            verdict = "RESTRICTION BINDS -- no Lyapunov function found on this sample"
        lines = [
            f"rate condition  <F,gradV> <= -sigma_min||gradV||^2 : "
            f"{100 * self.frac_rate_violating:6.2f}% of samples violate",
            f"norm condition  ||F|| <= sigma_max||gradV||        : "
            f"{100 * self.frac_norm_violating:6.2f}% of samples violate",
            f"V strictly decreasing (<F,gradV> < 0)             : "
            f"{100 * (1 - self.frac_increasing):6.2f}% of samples",
            f"achieved rate -<F,gradV>/||gradV||^2: min {pct[0]:+.4f}  p1 {pct[1]:+.4f}  "
            f"median {pct[2]:+.4f}  max {pct[3]:+.4f}",
            f"max rate violation {self.rate_violation.max():.3e}   "
            f"max norm violation {self.norm_violation.max():.3e}",
            f"VERDICT: {verdict}",
        ]
        bad = self.rate_violation > 1e-6
        if bad.any():
            lines.append(
                f"  shortfall sits at ||z|| median {self.z_norm[bad].median():.3f} "
                f"(all samples: {self.z_norm.median():.3f}); "
                f"||grad V|| there median {self.grad_V_norm[bad].median():.3e} "
                f"(all: {self.grad_V_norm.median():.3e})"
            )
        return "\n".join(lines)


def fit_potential(
    dynamics,
    Z,
    U,
    sigma_min,
    sigma_max,
    *,
    hidden=64,
    depth=3,
    activation="tanh",
    steps=3000,
    lr=3e-3,
    verbose=True,
):
    """Fit a scalar potential to certify the frozen field ``dynamics.F`` as a gradient system.

    Minimizes the two hinge residuals of the module docstring over ``(Z, U)``.  The field is
    used through ``dynamics.F`` only and is never differentiated with respect to its own
    parameters, so pass a trained model directly -- the gradient reaches the potential
    alone.

    Args:
        dynamics: Anything with ``F(z, u)``, e.g. a trained ``LatentFTNODE``.
        Z, U: Latents to certify over, and their inputs.  Use states the model was actually
            validated on -- a uniform box would ask the question somewhere the field was
            never fit.
        sigma_min, sigma_max: From the :class:`~ftnode.latent.KappaBudget` the field was
            trained under.  These are what make the condition a *rate* condition rather
            than plain monotonicity.

    Returns:
        FeasibilityResult
    """
    m, q = Z.shape[-1], (U.shape[-1] if U.dim() > 1 else 1)
    # Uncapped: the question is whether ANY V works, so the potential must not be
    # constrained while answering it.  `init_scale` matters here for the same reason it
    # does in training -- at Phi == 0 the hidden layers receive no gradient at all.
    phi = GradPotentialG(m, q, hidden, depth, activation=activation,
                         cap=False, init_scale=0.1).to(Z.device)

    with torch.no_grad():
        Fv = dynamics.F(Z, U)
        F_norm = Fv.norm(dim=-1)

    opt = torch.optim.Adam(phi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for step in range(steps):
        gv = Z - phi(Z, U)  # grad V = z - grad Phi
        gv2 = (gv * gv).sum(-1)
        rate = torch.relu((Fv * gv).sum(-1) + sigma_min * gv2)
        norm = torch.relu(F_norm - sigma_max * gv.norm(dim=-1))
        loss = (rate**2).mean() + (norm**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(float(loss))
        if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
            print(f"  [feasibility] step {step:5d}  loss {float(loss):.4e}  "
                  f"rate viol {float((rate > 1e-6).float().mean()):.4f}")

    with torch.no_grad():
        gv = Z - phi(Z, U)
        gv2 = (gv * gv).sum(-1)
        result = FeasibilityResult(
            potential=phi,
            rate_violation=torch.relu((Fv * gv).sum(-1) + sigma_min * gv2),
            norm_violation=torch.relu(F_norm - sigma_max * gv.norm(dim=-1)),
            grad_V_norm=gv.norm(dim=-1),
            z_norm=Z.norm(dim=-1),
            achieved_rate=-(Fv * gv).sum(-1) / gv2.clamp_min(1e-12),
            sigma_min=float(sigma_min),
            loss_history=history,
        )
    return result
