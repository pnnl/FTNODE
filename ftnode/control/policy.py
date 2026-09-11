"""The gradient control law, its closed loops, and the training cost.

The control law is gradient flow on the instantaneous cost
``J(u) = 1/2 ||g_psi(z, u) - z*||^2_W``: steer the input so that the *equilibrium
the plant is currently heading toward* moves onto the target.  Because ``g_psi``
is a gauge choice rather than a property of the plant, redesigning ``psi``
reshapes this cost landscape without touching the dynamics.

Every function here takes the frozen plant explicitly.  The earlier prototype
instead closed over module-global ``idm``, ``F_theta``, ``m``, ``Z_SCALE``,
``h_dt``, ``device`` and ``u_range``.
"""
from __future__ import annotations

import torch

from ..systems import DuffingParams, duffing_field_torch
from ..train import rk4_step

__all__ = [
    "g_psi",
    "sat_u",
    "udot_dir",
    "grad_u_J",
    "closed_loop",
    "model_closed_loop",
    "cost_T",
    "g_range_penalty",
]


def g_psi(fpsi, plant, z, u):
    """``g_psi = z - f_psi(z)^{-1} F_theta(z, u)``.

    Exact reproduction of ``F_theta`` for *any* ``psi`` -- this identity is what
    makes the control design safe, and it is asserted in the tests.
    """
    A = fpsi(z)
    Fv = plant.F(z, u)
    sol = torch.linalg.solve(A, Fv.unsqueeze(-1)).squeeze(-1)
    return z - sol


def sat_u(w, u_lo, u_hi):
    """Smooth bounded input ``u = u_c + du * tanh(w)`` in ``[u_lo, u_hi]``.

    The plant ALWAYS receives an admissible input, for every value of the control
    state ``w`` -- the admissibility hypothesis of the invariance theorem holds
    by parameterization rather than by projection.
    """
    u_c = 0.5 * (u_hi + u_lo)
    du = 0.5 * (u_hi - u_lo)
    return u_c + du * torch.tanh(w)


def udot_dir(fpsi, plant, z, u, z_star, W=None):
    """The control velocity ``u_dot / eta = -grad_u J = +D_uF^T f_psi^{-T} W (g_psi - z*)``.

    The leading ``+`` is correct precisely because the minus of ``-eta grad_u J``
    is already folded in::

        D_u g_psi = -f_psi^{-1} D_u F_theta
        grad_u J  = D_u g_psi^T W (g_psi - z*) = -D_uF^T f_psi^{-T} W (g_psi - z*)

    so ``-grad_u J = +D_uF^T f_psi^{-T} W (g_psi - z*)``.  Returns ``(b, q)``.
    """
    A = fpsi(z)
    gp = g_psi(fpsi, plant, z, u)
    e = gp - z_star
    if W is not None:
        e = torch.einsum("ij,bj->bi", W, e)
    v = torch.linalg.solve(A.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)  # f_psi^{-T} e
    DuF = plant.DuF(z, u)  # (b, m)
    return (DuF * v).sum(-1, keepdim=True)  # (b, 1) == D_uF^T v


def grad_u_J(fpsi, plant, z, u, z_star, W=None):
    """``grad_u J`` itself (``= -udot_dir``), for gradient-checking against autograd on ``J``."""
    return -udot_dir(fpsi, plant, z, u, z_star, W)


def closed_loop(
    fpsi,
    plant,
    x0,
    w_hist0,
    z_star,
    T_steps,
    h,
    eta,
    u_lo,
    u_hi,
    W=None,
    w0=None,
    params: DuffingParams | None = None,
    detach_plant=False,
):
    """DEPLOYMENT: the complete output-feedback system on the TRUE plant.

    ::

        x_dot = F_true(x, u)          <- TRUE Duffing, integrated in x-space
        y     = q = x_1               <- partial observation
        z     = encoder(window of y)  <- causal information state
        w_dot = -eta grad_w J,  u = s(w) in U

    The controller never sees ``x_2 = qdot``.  Returns a dict with keys
    ``x``, ``u``, ``z``, ``J``, ``w_final``.

    ``u_lo``/``u_hi`` are required, not defaulted.  The prototype defaulted them
    to the identification excitation range, which the reachability study shows is
    too little authority -- see :class:`~ftnode.control.ControlConfig`.
    """
    params = params or DuffingParams()
    device = x0.device
    b = x0.shape[0]
    x = x0.clone()
    wq = w_hist0.clone()  # (b, tau) rolling window of q
    w = torch.zeros(b, 1, device=device) if w0 is None else w0.clone()

    xs, us, zs, Js = [x], [], [], []
    for _ in range(T_steps):
        z = plant.encode(wq)  # information state from history
        u = sat_u(w, u_lo, u_hi)

        # control update: w_dot = -eta grad_w J = -eta (du/dw) grad_u J.
        # The chain rule through the saturation keeps u in U for all time.
        duw = 0.5 * (u_hi - u_lo) * (1 - torch.tanh(w) ** 2)  # ds/dw
        wdot = eta * duw * udot_dir(fpsi, plant, z, u, z_star, W)
        w = w + h * wdot

        # plant step: rk4 on the TRUE Duffing with u held over the step
        xin = x.detach() if detach_plant else x
        k1 = duffing_field_torch(xin, u, params)
        k2 = duffing_field_torch(xin + 0.5 * h * k1, u, params)
        k3 = duffing_field_torch(xin + 0.5 * h * k2, u, params)
        k4 = duffing_field_torch(xin + h * k3, u, params)
        x = xin + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # slide the observation window with the new measurement y = q
        wq = torch.cat([wq[:, 1:], x[:, :1]], dim=1)

        with torch.no_grad():
            gp = g_psi(fpsi, plant, z, u)
            Js.append(0.5 * ((gp - z_star) ** 2).sum(-1))
        xs.append(x)
        us.append(u)
        zs.append(z)

    return {
        "x": torch.stack(xs, 1),
        "u": torch.stack(us, 1).squeeze(-1),
        "z": torch.stack(zs, 1),
        "J": torch.stack(Js, 1),
        "w_final": w,
    }


def model_closed_loop(
    fpsi, plant, z0, z_star, T_steps, h, eta, u_lo, u_hi, W=None, w0=None, k_trunc=None
):
    """DESIGN TIME: closed loop on the FROZEN MODEL, ``z_dot = F_theta(z, u)``.

    This is what ``psi`` is trained against -- THE TRUE PLANT IS NOT AVAILABLE
    HERE.  Gradients reach ``psi`` only; ``F_theta`` is frozen.

    ``k_trunc`` is the truncated-BPTT window: the augmented ``(z, w)`` state is
    detached every ``k_trunc`` steps, so the gradient sees a ``k_trunc``-step
    window while the FORWARD trajectory runs the full horizon.  This is essential,
    not cosmetic -- without it gradient norms reach 1e17-1e19 and corrupt Adam's
    moments permanently.
    """
    device = z0.device
    b = z0.shape[0]
    z = z0
    w = torch.zeros(b, 1, device=device) if w0 is None else w0
    zs, us = [z], []
    for t in range(T_steps):
        if k_trunc and t % k_trunc == 0:
            z = z.detach()
            w = w.detach()
        u = sat_u(w, u_lo, u_hi)
        duw = 0.5 * (u_hi - u_lo) * (1 - torch.tanh(w) ** 2)
        wdot = eta * duw * udot_dir(fpsi, plant, z, u, z_star, W)
        w = w + h * wdot
        z = rk4_step(plant.F, z, u, h)
        zs.append(z)
        us.append(u)
    return torch.stack(zs, 1), torch.stack(us, 1).squeeze(-1)


def cost_T(
    fpsi,
    plant,
    z0,
    z_star,
    u_star,
    T_steps,
    h,
    eta,
    Qz,
    R_u,
    u_lo,
    u_hi,
    W=None,
    k_trunc=None,
):
    """``C_T``, the finite-horizon training cost in latent coordinates.

    ``C_T = 1/2 sum_t [ (z-z*)^T Qz (z-z*) + R (u-u*)^2 ] h``, left-endpoint
    quadrature: the state is penalized on nodes ``1..T`` and the input over the
    ``T`` held steps.  Returns ``(mean cost, zs, us)``.
    """
    zs, us = model_closed_loop(
        fpsi, plant, z0, z_star, T_steps, h, eta, u_lo, u_hi, W, k_trunc=k_trunc
    )
    e = zs - z_star
    run_z = torch.einsum("btj,jk,btk->bt", e, Qz, e)
    v = us - u_star
    run_u = R_u * v**2
    C = 0.5 * (run_z[:, 1:].sum(1) + run_u.sum(1)) * h
    return C.mean(), zs, us


def g_range_penalty(fpsi, plant, Z, U, c_z, r_in_target):
    """``L_g = E[ max(0, ||g_psi - c|| - r_in)^2 ]``.

    A SAMPLED surrogate for the uniform range condition -- it keeps the image of
    the redesigned ``g_psi`` inside a ball of radius ``r_in_target`` around
    ``c_z``.  A real certificate needs verification over the whole box, not a
    sample; see the certificate discussion in the control notebook.
    Returns ``(penalty, distances)``.
    """
    gp = g_psi(fpsi, plant, Z, U)
    d = (gp - c_z).norm(dim=-1)
    return torch.clamp(d - r_in_target, min=0.0).pow(2).mean(), d
