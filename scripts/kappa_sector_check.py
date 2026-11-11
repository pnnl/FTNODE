#!/usr/bin/env python3
"""Smallest operator conditioning that a strict gradient-class field needs at a damped sink.

The strict symmetric-``J_g`` class writes ``F = A grad V`` with ``sym A <= -sigma I`` and
``||A||_2 <= beta``.  At a nondegenerate minimum of ``V`` the Jacobian is ``A M`` with
``M = hess V`` positive definite, and every eigenvalue obeys the sector condition
``|Im lambda| <= kappa |Re lambda|`` with ``kappa = beta / sigma``.  So a damped mode with
damping ratio ``zeta`` needs ``kappa >= sqrt(1 - zeta^2) / zeta``.

This script measures how tight that bound is.  For the linear oscillator
``q'' + delta q' + w^2 q = 0`` it minimizes

    kappa(A) = ||A||_2 / lambda_min(-sym A),   A = J_lin M^{-1},

over every SPD ``M`` (parameterized by the Cholesky factor of ``M^{-1}``, three numbers), with
Nelder-Mead from many starts.  Two coordinate systems:

- raw ``(q, q_dot)``:            ``J_lin = [[0, 1], [-w^2, -delta]]``
- velocity-normalized:           ``J_lin = [[0, w], [-w, -delta]]`` (velocity divided by ``w``,
  the form a per-context normalization gives an oscillator)

Results (``delta = 0.2``): the normalized minimum is ``~1/zeta`` (the bound is attained); the
raw minimum fits ``(1 + w^2) / delta``.  The Duffing sink (``w^2 = 2``) needs 15.03 raw and
14.18 normalized, inside the shipped cap of 25.

Usage::

    python scripts/kappa_sector_check.py            # the default w grid, delta = 0.2
    python scripts/kappa_sector_check.py --delta 0.1 --w 1 2 4

Pure numpy/scipy, single-threaded, a few seconds.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize


def _spd_from(theta):
    """SPD 2x2 matrix ``L L^T`` from a Cholesky factor with a log-parameterized diagonal."""
    L = np.array([[np.exp(theta[0]), 0.0], [theta[1], np.exp(theta[2])]])
    return L @ L.T


def kappa_of(theta, J):
    """``||A||_2 / lambda_min(-sym A)`` for ``A = J P``, ``P = M^{-1}``; a large value if not strict."""
    A = J @ _spd_from(theta)
    lmin = -np.linalg.eigvalsh(0.5 * (A + A.T)).max()
    return np.linalg.norm(A, 2) / lmin if lmin > 0 else 1e6


def _feasible_raw_start(w, delta, frac):
    """A strict starting point for the raw coordinates, which random starts rarely hit.

    With ``p22 = w^2 p11 + delta p12`` the off-diagonal of ``sym(J P)`` vanishes, and
    ``-delta w^2 p11 / (w^2 + delta^2) < p12 < 0`` makes both diagonal entries negative.
    """
    p11 = 1.0
    p12 = -frac * delta * w * w / (w * w + delta * delta)
    p22 = w * w * p11 + delta * p12
    L = np.linalg.cholesky(np.array([[p11, p12], [p12, p22]]))
    return np.array([np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])])


def kappa_min(J, starts, rng):
    """Smallest ``kappa`` found over ``starts`` (an iterable of start vectors)."""
    best = np.inf
    for t0 in starts:
        if kappa_of(t0, J) >= 1e6:
            continue
        res = minimize(kappa_of, t0, args=(J,), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-12, maxiter=20000))
        best = min(best, res.fun)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--w", type=float, nargs="+", default=[0.5, 1.0, np.sqrt(2.0), 2.0, 3.4, 5.0, 8.0])
    ap.add_argument("--raw-starts", type=int, default=300)
    ap.add_argument("--norm-starts", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = args.delta

    print(f"delta = {d}")
    print(f"{'w':>7} {'zeta':>8} {'raw k_min':>10} {'k*zeta':>7} {'(1+w^2)/d':>10} "
          f"{'norm k_min':>11} {'k*zeta':>7}")
    for w in args.w:
        zeta = d / (2.0 * w)
        J_raw = np.array([[0.0, 1.0], [-w * w, -d]])
        raw_starts = [
            _feasible_raw_start(w, d, rng.uniform(0.05, 0.95))
            + (rng.normal(scale=0.02, size=3) if i < args.raw_starts // 2 else 0.0)
            for i in range(args.raw_starts)
        ]
        k_raw = kappa_min(J_raw, raw_starts, rng)
        J_norm = np.array([[0.0, w], [-w, -d]])
        k_norm = kappa_min(J_norm, (rng.normal(size=3) for _ in range(args.norm_starts)), rng)
        print(f"{w:7.3f} {zeta:8.4f} {k_raw:10.2f} {k_raw * zeta:7.3f} {(1 + w * w) / d:10.2f} "
              f"{k_norm:11.2f} {k_norm * zeta:7.3f}")


if __name__ == "__main__":
    main()
