"""Spectral diagnostics that verify the kappa bounds actually hold on a trained model.

Promoted from the two kappa-bounded duffing notebooks and the learned-splitting
control notebook.  These are the measurements that turn "bounded by construction"
into a checked claim: every bound in :mod:`ftnode.latent` is a design-time
guarantee, and these functions confirm it empirically over the latent box.

The notebooks' ``M_LAT`` global becomes an explicit ``m`` argument.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "A_stats",
    "skew_stats",
    "op_stats",
    "eigvals_A",
    "eig_field_jac",
    "pca_2d",
    "linear_recovery_r2",
    "g_image",
    "jg_stats",
    "empirical_lipschitz",
]


@torch.no_grad()
def A_stats(dyn, Z):
    """Per-sample ``(max Re eig(A), sigma_max(A), kappa(A))`` for a structured model.

    ``kappa = sigma_max(A) / lambda_min(-sym A)``.  The first return should be at
    most ``-sigma_min`` (asymptotic stability) and the third at most
    ``KappaBudget.kappa_max`` for the bounded variants.
    """
    A = dyn.A(Z).cpu().numpy()
    sv = np.linalg.svd(A, compute_uv=False)
    lmin = -np.linalg.eigvalsh((A + A.transpose(0, 2, 1)) / 2)[:, -1]
    maxre = np.linalg.eigvals(A).real.max(1)
    return maxre, sv[:, 0], sv[:, 0] / np.maximum(lmin, 1e-12)


@torch.no_grad()
def skew_stats(dyn, Z):
    """Per-sample ``(||K||_2, kappa(K))`` of the skew part ``K = (A - A^T)/2``.

    ``||K||_2`` is what the ``c_K`` budget caps.  For the Youla variant the
    singular values come in equal pairs by construction, so ``kappa(K)`` reports
    the spread of the block magnitudes ``beta_j``.
    """
    A = dyn.A(Z)
    Ksk = 0.5 * (A - A.transpose(-1, -2))
    sv = torch.linalg.svdvals(Ksk).cpu().numpy()
    return sv[:, 0], sv[:, 0] / np.maximum(sv[:, -1], 1e-12)


@torch.no_grad()
def op_stats(op, Z, chunk=2048):
    """:func:`A_stats` for a bare batched operator ``op(z) -> (b, m, m)``.

    Used on :class:`ftnode.control.SplitOperator`, which is the operator itself
    rather than a dynamics module with an ``.A`` method.  Chunked because the
    control stage evaluates it over tens of thousands of samples.
    """
    mr, sm, kp = [], [], []
    for i in range(0, Z.shape[0], chunk):
        A = op(Z[i : i + chunk])
        sv = torch.linalg.svdvals(A)
        lmin = -torch.linalg.eigvalsh(0.5 * (A + A.transpose(1, 2)))[:, -1]
        mr.append(torch.linalg.eigvals(A).real.max(1).values.cpu().numpy())
        sm.append(sv[:, 0].cpu().numpy())
        kp.append((sv[:, 0] / lmin.clamp_min(1e-12)).cpu().numpy())
    return np.concatenate(mr), np.concatenate(sm), np.concatenate(kp)


@torch.no_grad()
def eigvals_A(dyn, Z, m, chunk=2048):
    """Complex eigenvalues of ``A(z)`` over ``Z``, shape ``(N, m)``."""
    out = []
    for i in range(0, Z.shape[0], chunk):
        A = dyn.A(Z[i : i + chunk])
        out.append(torch.linalg.eigvals(A).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, m), dtype=complex)


def eig_field_jac(dyn, Z, U, m, chunk=1024):
    """Eigenvalues of the field Jacobian ``dF/dz``, for models with no ``A(z)``.

    The structured variants expose ``A(z)`` directly, so :func:`eigvals_A` reads
    the spectrum off for free.  The unstructured
    :class:`~ftnode.latent.LatentNODE` has no such object, and its linearization
    has to be differentiated out -- this is what makes the two comparable in the
    spectrum figure.
    """
    from torch.func import jacrev, vmap

    def f_single(zi, ui):
        return dyn.F(zi.unsqueeze(0), ui.unsqueeze(0)).squeeze(0)

    out = []
    for i in range(0, Z.shape[0], chunk):
        J = vmap(jacrev(f_single, argnums=0))(Z[i : i + chunk], U[i : i + chunk])
        out.append(torch.linalg.eigvals(J.detach()).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, m), dtype=complex)


# ------------------------------------------------------- latent geometry / recovery


def pca_2d(Z):
    """Project latent states onto their first two principal components.

    Returns ``(proj, basis)`` of shapes ``(N, 2)`` and ``(2, m)``.  Used to look at
    the geometry the encoder actually learned -- whether the latent trajectories
    reproduce the plant's phase-portrait structure or fold into something else.
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T, Vt[:2]


def linear_recovery_r2(Z, target, fit_idx, eval_idx):
    """Held-out ``R^2`` of a linear readout from the latent state to ``target``.

    This is the test that matters for a *partially observed* system: ``q`` is
    measured, so recovering it is unsurprising, but ``q_dot`` is never shown to
    the model at any point.  A high ``R^2`` to ``q_dot`` means the identified
    latent state implicitly reconstructed the hidden coordinate -- the encoder
    inferred velocity from the measurement window on its own.

    Fitted on ``fit_idx`` and scored on ``eval_idx`` so the number reports
    generalization rather than the capacity of a least-squares fit.
    """
    Zf = np.concatenate([Z[fit_idx], np.ones((len(fit_idx), 1))], 1)
    w, *_ = np.linalg.lstsq(Zf, target[fit_idx], rcond=None)
    pred = np.concatenate([Z[eval_idx], np.ones((len(eval_idx), 1))], 1) @ w
    resid = ((target[eval_idx] - pred) ** 2).sum()
    total = ((target[eval_idx] - target[eval_idx].mean()) ** 2).sum()
    return 1 - resid / (total + 1e-12)


def empirical_lipschitz(fn, lo=-12.0, hi=12.0, n=240_001, device=None):
    """Numerically estimate the Lipschitz constant of a scalar elementwise function.

    Evaluates ``|f'(x)|`` on a dense grid via autograd and returns the maximum.
    Exact for elementwise activations, where the derivative at each grid point is
    independent of its neighbours.

    Use this to check an activation rather than trusting the ``lipschitz_1`` flag
    in :data:`ftnode.latent.ACTIVATIONS`: a composed Lipschitz bound on a network
    is only as good as the per-layer constant it assumes, and the default
    ``nn.SiLU`` returns roughly 1.0998 here -- above 1, which silently invalidates
    any bound that assumed unit gain.

    Being a grid maximum this is a *lower* bound on the true supremum, so it is
    trustworthy for showing an activation exceeds 1 and only suggestive for
    showing it does not.
    """
    x = torch.linspace(lo, hi, n, device=device, dtype=torch.float64, requires_grad=True)
    y = fn(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    return grad.abs().max().item()


@torch.no_grad()
def g_image(dyn, Z, U):
    """The image of the equilibrium map, ``g(z, u)``.

    Thin wrapper, but it names the thing being measured: decoding this and
    comparing against the plant's true equilibrium branches is what shows whether
    the learned ``g`` found the actual pitchfork.

    **The bound to check against is the map's, not a universal one.**
    :class:`~ftnode.latent.BoundedTanhG` gives ``|g|_inf <= R_g`` -- a box, imposed by its
    ``tanh``.  :class:`~ftnode.latent.GradPotentialG` gives ``||g||_2 <= g_bound`` -- a
    ball, imposed by spectral caps on its potential's weights.  Read the bound off the
    module rather than assuming either.
    """
    return dyn.g(Z, U)


def jg_stats(dyn, Z, U, chunk=4096):
    """Per-sample ``(skew fraction, lambda_max(sym J_g))`` of the equilibrium map's Jacobian.

    Two numbers that decide whether a symmetric-``J_g`` map is doing what it claims:

    * ``||skew J_g||_F / ||J_g||_F`` -- zero by construction for a gradient map, and the
      check that it really is a gradient.  For reference, an *unstructured* ``m x m``
      Jacobian sits at ``sqrt((m-1)/2m)`` = 0.612 at ``m=4``, which is where the
      ``tanh_mlp`` maps measure; the statistic is only meaningful against that null.
    * ``lambda_max(sym J_g)`` -- exceeds 1 exactly at saddles of the potential, so it is
      how you confirm multistability survived a bound on ``g``.  Never constrain it below
      1: that would make ``V`` strictly convex and delete the pitchfork.

    Not ``@torch.no_grad()``: the Jacobian is taken with ``torch.func``, which needs to
    differentiate.  Results are detached.
    """
    from torch.func import jacrev, vmap

    def g_single(zi, ui):
        return dyn.g(zi.unsqueeze(0), ui.reshape(1, -1)).squeeze(0)

    fracs, lams = [], []
    U2 = U.unsqueeze(-1) if U.dim() == Z.dim() - 1 else U
    for i in range(0, Z.shape[0], chunk):
        J = vmap(jacrev(g_single, argnums=0))(Z[i : i + chunk], U2[i : i + chunk]).detach()
        skew = 0.5 * (J - J.transpose(-1, -2))
        denom = J.flatten(1).norm(dim=1).clamp_min(1e-12)
        fracs.append((skew.flatten(1).norm(dim=1) / denom).cpu().numpy())
        sym = 0.5 * (J + J.transpose(-1, -2))
        lams.append(torch.linalg.eigvalsh(sym)[:, -1].cpu().numpy())
    return np.concatenate(fracs), np.concatenate(lams)
