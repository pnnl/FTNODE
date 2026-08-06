"""Small plotting helpers shared by the duffing notebooks.

Separate from :mod:`ftnode.diagnostics`, which is numerical and deliberately
imports no matplotlib -- importing it should not pull in a plotting backend.  These
were defined inline in ``examples/duffing/pkg_kappa_variants.ipynb`` and needed
again by the analysis notebook; one definition beats two copies drifting apart.
"""
from __future__ import annotations

import numpy as np

__all__ = ["seed_grid", "band", "stack_histories"]


def seed_grid(n, ncols=5, panel=(2.7, 2.4), **kw):
    """A grid of ``n`` panels, one per seed, with the unused cells switched off.

    Args:
        n (int): Number of panels needed.
        ncols (int): Columns; clipped to ``n`` so a 2-seed run does not get five.
        panel (tuple): ``(width, height)`` in inches per panel.
        **kw: Forwarded to ``plt.subplots`` (``sharex``, ``sharey``, ...).

    Returns:
        tuple: ``(fig, axes)`` with ``axes`` flattened to 1-D, so callers index by
        seed regardless of the grid shape.
    """
    import matplotlib.pyplot as plt

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False, **kw
    )
    axes = axes.ravel()
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return fig, axes


def band(ax, arr, color, label, logy=False):
    """Mean +/- standard deviation across seeds, as a line with a shaded band.

    ``arr`` is ``(n_seeds, n_epochs)``.  NaN-aware, so a diverged or missing seed
    thins the band rather than erasing the curve.
    """
    mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
    x = np.arange(arr.shape[1])
    ax.plot(x, mu, color=color, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
    if logy:
        ax.set_yscale("log")
    return ax


def stack_histories(histories, key, n_epochs):
    """Stack one per-epoch series across seeds into a NaN-padded ``(n_seeds, n_epochs)``.

    Padding matters: a run that diverged stops early and a job that was never
    trained is absent entirely, so the rows are ragged.  NaN keeps both out of the
    mean rather than dragging it toward zero.

    Args:
        histories (list): One history dict per seed, as
            :attr:`ftnode.experiments.Run.histories` returns; ``None`` entries
            (untrained jobs) become all-NaN rows.
        key (str): Series to extract -- ``'train'``, ``'val_extrap'``, ``'zmax'``
            or ``'res'``.
        n_epochs (int): Width to pad to, normally ``TrainConfig.n_epochs``.
    """
    out = np.full((len(histories), n_epochs), np.nan)
    for i, hist in enumerate(histories):
        if hist is None:
            continue
        row = hist[key][:n_epochs]
        out[i, : len(row)] = row
    return out
