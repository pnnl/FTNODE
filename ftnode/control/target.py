"""Finding the latent target ``z*``, and the latent-LQR baseline.

The running cost penalizes ``z - z*``, so ``z*`` had better be an actual steady
state of the identified field -- otherwise the cost can never reach zero and the
design problem is inconsistent.  That is the entire content of
:func:`latent_target`, and :func:`latent_target_encode` is kept as the documented
negative control showing what goes wrong without it.
"""
from __future__ import annotations

import numpy as np
import torch

from ..systems import DuffingParams, duffing_field_torch

__all__ = ["latent_target_encode", "latent_target", "latent_lqr", "lqr_closed_loop_true"]


def latent_target_encode(plant, u_star, q_star, tau):
    """NEGATIVE CONTROL: ``z* = encoder(constant-q window)``.

    A constant window is not a consistent information state -- the encoder never
    saw one during identification -- so this ``z*`` is NOT an equilibrium of
    ``F_theta``: ``||F_theta(z*, u*)||`` comes out O(0.1), violating the
    equilibrium-pair condition the running cost assumes.  Regulating to it is
    ill-posed.  Use :func:`latent_target` instead; this exists to make the
    failure measurable, and as the warm start for the real solve.
    """
    w = torch.full((1, tau), float(q_star), device=plant.device)
    with torch.no_grad():
        return plant.encode(w).squeeze(0)


def latent_target(
    plant,
    u_star,
    q_star,
    tau,
    iters=4000,
    lr=1e-2,
    lam_dec=10.0,
    z_init=None,
    verbose=True,
):
    """Solve the equilibrium pair FOR THE FROZEN MODEL::

        find z*  s.t.  F_theta(z*, u*) = 0   and   decode(z*) = q*.

    Solved as a penalized least-squares problem, warm-started from the encoder
    image.  Returns ``(z_star, residual, decoded_q)``.
    """
    if z_init is None:
        z_init = latent_target_encode(plant, u_star, q_star, tau)
    z = z_init.clone().unsqueeze(0).requires_grad_(True)
    us = torch.full((1, 1), float(u_star), device=plant.device)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        r_dyn = (plant.F(z, us) ** 2).sum()
        r_dec = (plant.decode(z) - q_star) ** 2
        loss = r_dyn + lam_dec * r_dec.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zf = z.detach()
        res = plant.F(zf, us).norm().item()
        qd = plant.decode(zf).item()
    if verbose:
        print(
            f"  z* solve: ||F_theta(z*,u*)|| = {res:.3e}   decode(z*) = {qd:+.5f} "
            f"(target {q_star:+.3f})"
        )
    return zf.squeeze(0), res, qd


def latent_lqr(plant, z_star, u_star, Qz, R_u):
    """LQR on the linearization of the FROZEN MODEL at ``(z*, u*)``.

    ``A_lin = dF_theta/dz`` and ``B_lin = dF_theta/du`` at the target; returns
    ``(A_lin, B_lin, P, K)`` with the law ``u = u* - K (z - z*)``.  This is the
    baseline the learned splitting is measured against, and it anchors the
    LQR-correspondence claim numerically.

    Falls back to Kleinman iteration if scipy's CARE solver fails.
    """
    from scipy.linalg import solve_continuous_are

    zs = z_star.unsqueeze(0).clone().requires_grad_(True)
    us = torch.full((1, 1), float(u_star), device=plant.device, requires_grad=True)
    A_lin = (
        torch.autograd.functional.jacobian(lambda zz: plant.F(zz, us.detach()).squeeze(0), zs)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    B_lin = (
        torch.func.jvp(lambda uu: plant.F(zs.detach(), uu), (us.detach(),), (torch.ones_like(us),))[
            1
        ]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    B_lin = B_lin.reshape(-1, 1)
    Q = Qz.cpu().numpy()
    R = np.array([[float(R_u)]])
    try:
        P = solve_continuous_are(A_lin, B_lin, Q, R)
    except Exception as ex:  # pragma: no cover
        print("  (scipy CARE failed, falling back to Kleinman):", ex)
        from scipy.linalg import solve_sylvester

        P = np.eye(A_lin.shape[0])
        for _ in range(500):
            K = np.linalg.solve(R, B_lin.T @ P)
            Ac = A_lin - B_lin @ K
            P = solve_sylvester(Ac.T, Ac, -(Q + K.T @ R @ K))
    K = np.linalg.solve(R, B_lin.T @ P)
    return A_lin, B_lin, P, K


def lqr_closed_loop_true(
    plant,
    K,
    x0,
    w_hist0,
    z_star,
    u_star,
    T_steps,
    h,
    u_lo,
    u_hi,
    params: DuffingParams | None = None,
):
    """True-plant closed loop under the static latent LQR law ``u = clip(u* - K(z - z*))``.

    Same observation structure as :func:`~ftnode.control.policy.closed_loop` --
    the gain sees only the encoded window -- so the comparison isolates the
    control law rather than the information available to it.
    """
    params = params or DuffingParams()
    device = x0.device
    x = x0.clone()
    wq = w_hist0.clone()
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    xs, us, zs = [x], [], []
    for _ in range(T_steps):
        with torch.no_grad():
            z = plant.encode(wq)
            u = u_star - torch.einsum("ij,bj->bi", Kt, z - z_star)
            u = u.clamp(u_lo, u_hi)
            k1 = duffing_field_torch(x, u, params)
            k2 = duffing_field_torch(x + 0.5 * h * k1, u, params)
            k3 = duffing_field_torch(x + 0.5 * h * k2, u, params)
            k4 = duffing_field_torch(x + h * k3, u, params)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            wq = torch.cat([wq[:, 1:], x[:, :1]], dim=1)
        xs.append(x)
        us.append(u)
        zs.append(z)
    return {
        "x": torch.stack(xs, 1),
        "u": torch.stack(us, 1).squeeze(-1),
        "z": torch.stack(zs, 1),
    }
