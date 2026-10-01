"""Rollout metrics for the measured-``q`` Duffing test split.

The aggregate ``L = 600`` test MSE weights every step equally.  Duffing damps, so most
steps sit near a well centre ``q = +-1`` and the aggregate is dominated by the few
trajectories that settle in the wrong well.  These metrics split the error so a score
separates transient tracking from well selection:

- ``mse`` and ``mse_<a>-<b>`` -- the aggregate and per-window ``(q_hat - q)^2``.
- ``wrong_wells`` / ``well_acc`` -- final-well selection; the final well is the sign of
  the mean of ``q`` over the late window.
- ``well_acc_crossers`` -- well accuracy on trajectories whose early sign differs from
  their final well (the ones where picking the well needs the dynamics).
- ``wrong_well_error_share`` -- the fraction of squared error from wrong-well trajectories.
- ``late_mse_correct_wells`` -- late-window error on trajectories that pick the right well.
- ``transient_mse`` -- early error after removing each trajectory's own late offset.
- ``osc_energy_ratio`` -- predicted over true per-trajectory variance in the transient
  window (1 = right amplitude, 0 = flat).
- ``dom_freq_match`` -- fraction of trajectories whose dominant transient frequency matches.
"""
from __future__ import annotations

import numpy as np
import torch

from .train import rollout_q

__all__ = ["WINDOWS", "rollout_metrics", "rollout_test_split"]

WINDOWS = ((0, 50), (50, 200), (200, 400), (400, None))
_EARLY = 20          # steps that define a trajectory's early sign
_LATE = 400          # first step of the late (settled) window
_TRANSIENT = 200     # end of the transient window
_OSC = (50, 200)     # window for the oscillation-energy ratio


def _dom_freq(x):
    """Dominant nonzero frequency bin of the de-meaned transient; -1 when it is flat."""
    x = x[:, :_TRANSIENT] - x[:, :_TRANSIENT].mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1))[:, 1:]
    return np.where(spec.max(1) > 0, spec.argmax(1) + 1, -1)


def rollout_metrics(qhat, y) -> dict:
    """Metrics of predicted ``qhat`` against measured ``y``, both ``(n_traj, L+1)``.

    Non-finite predictions are counted in ``nonfinite_traj`` and excluded from the means.
    The windowed metrics need the rollout to reach the late window (``L >= 400``); on a
    shorter horizon (a smoke run) only the aggregate is returned.
    """
    P = np.asarray(qhat, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if P.shape != Y.shape or P.ndim != 2:
        raise ValueError(f"qhat and y must share shape (n_traj, L+1); got {P.shape}, {Y.shape}")
    finite = np.isfinite(P).all(1)
    P = np.where(np.isfinite(P), P, np.nan)
    e = (P - Y) ** 2
    out = {"n_traj": int(len(Y)), "nonfinite_traj": int((~finite).sum()), "mse": float(np.nanmean(e))}
    if Y.shape[1] <= _LATE:
        return out
    for a, b in WINDOWS:
        out[f"mse_{a}-{b if b is not None else Y.shape[1]}"] = float(np.nanmean(e[:, a:b]))

    well = np.sign(Y[:, _LATE:].mean(1))
    crossers = np.sign(Y[:, :_EARLY].mean(1)) != well
    ok = np.sign(P[:, _LATE:].mean(1)) == well            # NaN rows compare False: wrong
    out["wrong_wells"] = int((~ok).sum())
    out["well_acc"] = float(ok.mean())
    out["n_crossers"] = int(crossers.sum())
    out["well_acc_crossers"] = float(ok[crossers].mean()) if crossers.any() else float("nan")
    total = np.nansum(e)
    out["wrong_well_error_share"] = float(np.nansum(e[~ok]) / total) if total > 0 else 0.0
    out["late_mse_correct_wells"] = float(np.nanmean(e[ok, _LATE:])) if ok.any() else float("nan")

    Pd = P[:, :_TRANSIENT] - P[:, _LATE:].mean(1, keepdims=True)
    Yd = Y[:, :_TRANSIENT] - Y[:, _LATE:].mean(1, keepdims=True)
    out["transient_mse"] = float(np.nanmean((Pd - Yd) ** 2))
    a, b = _OSC
    out["osc_energy_ratio"] = float(np.nanmean(P[:, a:b].var(1)) / np.mean(Y[:, a:b].var(1)))
    # A non-finite or flat prediction never matches: it has no dominant frequency.
    match = np.zeros(len(Y), dtype=bool)
    match[finite] = _dom_freq(P[finite]) == _dom_freq(Y[finite])
    match &= _dom_freq(np.nan_to_num(P)) > 0
    out["dom_freq_match"] = float(match.mean())
    return out


def rollout_test_split(model, test_cfgs, L, h, device):
    """Roll ``model`` out on each test config; return concatenated ``(qhat, y)`` arrays.

    ``test_cfgs`` is a list of ``(seed, DuffingDataConfig)``.  Also returns the per-seed
    aggregate MSE, which is what the factorial reported before these metrics existed.
    """
    from ..systems import make_dataset

    model.eval()
    qs, ys, per_seed = [], [], {}
    with torch.no_grad():
        for ts, cfg in test_cfgs:
            ds = make_dataset(cfg).to(device)
            if hasattr(model, "prepare"):
                model.prepare(ds.W)
            qhat = rollout_q(model, ds.W, ds.U, L, h)
            mse = ((qhat - ds.Y) ** 2).mean().item()
            per_seed[ts] = mse if np.isfinite(mse) else float("nan")
            qs.append(qhat.cpu().numpy())
            ys.append(ds.Y.cpu().numpy())
    return np.concatenate(qs), np.concatenate(ys), per_seed
