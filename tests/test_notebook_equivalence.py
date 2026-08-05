"""Bitwise equivalence between the package and the frozen duffing notebooks.

This is the check the rest of the suite cannot make. Every other test asks
"is the package internally consistent?" -- these ask "does it still produce the
numbers the committed results were produced with?"

The distinction is not academic. An early version of `_assemble` built the
dynamics before the encoder. Models constructed fine, kappa bounds held, and the
entire suite passed -- but because each sub-module draws from the global torch
RNG as it initializes, `torch.manual_seed(s)` silently stopped reproducing any
frozen notebook run. Only a bitwise comparison against the notebooks caught it.

Method: execute the frozen notebook's own definition cells into a throwaway
module, build the package's equivalent alongside, seed both identically, and
diff. Anything that changes RNG consumption, layer construction order, arithmetic
or defaults shows up here as a non-zero difference.

Marked `notebook` and run by default -- they are cheap (~4s total) because
execution stops before the notebooks' module-level dataset construction, which we
do not need. Skip with `pytest -m 'not notebook'` if you are deliberately editing
a frozen notebook.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

from ftnode.diagnostics import A_stats, skew_stats
from ftnode.latent import (
    KappaBudget,
    LatentModelConfig,
    build_clamp,
    build_unbounded,
    build_youla,
    spectral_clamp,
)
from ftnode.systems import DuffingDataConfig, make_dataset
from ftnode.train import TrainConfig, train_one

pytestmark = pytest.mark.notebook

DUFFING = pathlib.Path(__file__).resolve().parents[1] / "examples" / "duffing"
NB_SVD = DUFFING / "duffing_kappa_svdclamp_vs_ln_2variant_10seed.ipynb"
NB_YOULA = DUFFING / "duffing_kappa_bounded_youla_skew_3variant_10seed.ipynb"

#: The notebooks build their 512+64 trajectory datasets at module scope, in pure
#: Python RK4. We never use those tensors -- the comparisons inject their own --
#: so execution stops at this line, turning a ~40s exec into well under a second.
DATASET_MARKER = "Wtr, Utr, Ytr, Xtr = make_dataset("

#: Definition cells: config, plant, data, nets, models, clamp, budget/builders,
#: rk4+trainer. Cell 8 onward is the multi-seed driver, which would train.
N_DEF_CELLS = 8


def _exec_notebook_defs(path: pathlib.Path, modname: str) -> dict:
    """Execute a frozen notebook's definition cells and return its namespace."""
    cells = [
        "".join(c["source"])
        for c in json.loads(path.read_text())["cells"]
        if c["cell_type"] == "code"
    ][:N_DEF_CELLS]

    # Trim *within* the one cell that builds the datasets, keeping every later
    # cell -- the nets, models, clamp, budget, builders and trainer all live
    # downstream of it.
    trimmed = 0
    for i, cell in enumerate(cells):
        if DATASET_MARKER in cell:
            cells[i] = cell[: cell.index(DATASET_MARKER)]
            trimmed += 1
    assert trimmed == 1, f"{path.name}: dataset marker matched {trimmed} cells; re-check the trim"
    src = "\n".join(cells)

    # @dataclass resolves sys.modules[cls.__module__], so a bare dict namespace
    # raises AttributeError. Register a real module object instead.
    mod = types.ModuleType(modname)
    sys.modules[modname] = mod
    ns = mod.__dict__

    # The notebooks pick their device with `torch.cuda.is_available()`. Force it
    # false for the duration so these comparisons run on CPU everywhere -- the
    # package side builds on CPU, and a cross-device diff is not bitwise.
    real_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        exec(compile(src, f"<{modname}>", "exec"), ns)
    finally:
        torch.cuda.is_available = real_is_available
    assert ns["device"].type == "cpu", "notebook did not fall back to CPU"
    return ns


@pytest.fixture(scope="module")
def nb_svd():
    return _exec_notebook_defs(NB_SVD, "_nb_svdclamp")


@pytest.fixture(scope="module")
def nb_youla():
    return _exec_notebook_defs(NB_YOULA, "_nb_youla")


@pytest.fixture(scope="module")
def budget4():
    return KappaBudget(sigma_min=0.1, kappa_max=25.0, skew_frac=0.6, m=4)


@pytest.fixture(scope="module")
def cfg4():
    return LatentModelConfig()


@pytest.fixture(scope="module")
def Z():
    return (2 * torch.rand(256, 4, generator=torch.Generator().manual_seed(5)) - 1) * 2.0


# ------------------------------------------------------------------ config math


def test_kappa_budget_matches_notebook(nb_svd, budget4):
    assert budget4.budget == pytest.approx(nb_svd["budget"], abs=1e-12)
    assert budget4.c_P == pytest.approx(nb_svd["c_P"], abs=1e-12)
    assert budget4.c_K == pytest.approx(nb_svd["c_K"], abs=1e-12)


def test_youla_derived_constants_match_notebook(nb_youla, budget4):
    assert budget4.l_bound == pytest.approx(nb_youla["l_bound"], abs=1e-12)
    assert budget4.beta_min(nb_youla["BETA_MIN_FRAC"]) == pytest.approx(
        nb_youla["beta_min"], abs=1e-12
    )


# ----------------------------------------------------------------------- data


def test_make_dataset_is_bitwise_identical(nb_svd):
    pkg = make_dataset(DuffingDataConfig(n_traj=24, L=30, tau=8, seed=0))
    W, U, Y, X = nb_svd["make_dataset"](n_traj=24, L=30, tau=8, h=0.05, seed=0)
    assert torch.equal(pkg.W, W)
    assert torch.equal(pkg.U, U)
    assert torch.equal(pkg.Y, Y)
    assert torch.equal(pkg.Xfull, X)


# ----------------------------------------------------------------------- clamp


def test_spectral_clamp_is_bitwise_identical(nb_svd, budget4):
    B = torch.randn(32, 4, 4, generator=torch.Generator().manual_seed(3))
    assert torch.equal(spectral_clamp(B, budget4.c_K), nb_svd["spectral_clamp"](B, budget4.c_K))


# ---------------------------------------------------------------------- models

VARIANTS = [
    ("clamp", NB_SVD, "build_clamp"),
    ("youla", NB_YOULA, "build_youla"),
    ("unbounded", NB_YOULA, "build_unbounded"),
]


def _pkg_build(name, cfg, budget, nb):
    if name == "clamp":
        return build_clamp(cfg, budget)
    if name == "youla":
        return build_youla(cfg, budget, beta_min_frac=nb["BETA_MIN_FRAC"])
    return build_unbounded(cfg)


@pytest.mark.parametrize("name,_nb_path,nb_builder", VARIANTS)
def test_init_weights_are_bitwise_identical(
    name, _nb_path, nb_builder, cfg4, budget4, nb_svd, nb_youla
):
    """Catches any change to layer construction order or RNG consumption."""
    nb = nb_svd if _nb_path is NB_SVD else nb_youla
    torch.manual_seed(0)
    pkg = _pkg_build(name, cfg4, budget4, nb).state_dict()
    torch.manual_seed(0)
    ref = nb[nb_builder]().state_dict()

    assert sorted(pkg) == sorted(ref), f"{name}: state-dict keys diverged"
    for k in ref:
        assert torch.equal(pkg[k], ref[k]), f"{name}: {k} differs"


@pytest.mark.parametrize("name,_nb_path,nb_builder", VARIANTS)
def test_A_and_F_are_bitwise_identical(
    name, _nb_path, nb_builder, cfg4, budget4, Z, nb_svd, nb_youla
):
    nb = nb_svd if _nb_path is NB_SVD else nb_youla
    U = torch.zeros(Z.shape[0])
    torch.manual_seed(0)
    pkg = _pkg_build(name, cfg4, budget4, nb).dynamics
    torch.manual_seed(0)
    ref = nb[nb_builder]().dynamics

    assert (pkg.A(Z) - ref.A(Z)).abs().max().item() == 0.0, f"{name}: A(z)"
    assert (pkg.g(Z, U) - ref.g(Z, U)).abs().max().item() == 0.0, f"{name}: g(z,u)"
    assert (pkg.F(Z, U) - ref.F(Z, U)).abs().max().item() == 0.0, f"{name}: F(z,u)"


# ----------------------------------------------------------------- diagnostics


def test_diagnostics_match_notebook_helpers(nb_youla, cfg4, budget4, Z):
    torch.manual_seed(0)
    dyn = build_youla(cfg4, budget4, beta_min_frac=nb_youla["BETA_MIN_FRAC"]).dynamics
    for pkg_fn, nb_name in ((A_stats, "A_stats"), (skew_stats, "skew_stats")):
        for a, b in zip(pkg_fn(dyn, Z), nb_youla[nb_name](dyn, Z)):
            assert np.array_equal(a, b), nb_name


# -------------------------------------------------------------------- training


def test_train_one_history_matches_notebook(tmp_path, nb_svd, cfg4, budget4):
    """The end-to-end check: same data, same seed, same optimizer trajectory.

    Catches anything the static comparisons miss -- loss assembly, the residual
    penalty, gradient clipping, the LR schedule, batch ordering.
    """
    tr = make_dataset(DuffingDataConfig(n_traj=32, L=20, tau=8, seed=0))
    va = make_dataset(DuffingDataConfig(n_traj=8, L=40, tau=8, seed=1))

    # The notebook's train_one closes over module globals; point them at our data.
    nb_svd.update(
        Wtr_d=tr.W, Utr_d=tr.U, Ytr_d=tr.Y,
        Wva_d=va.W, Uva_d=va.U, Yva_d=va.Y,
        L=20, L_eval=40, h_dt=0.05, device=torch.device("cpu"),
    )

    torch.manual_seed(0)
    pkg_model = build_clamp(cfg4, budget4)
    torch.manual_seed(42)
    _, pkg_hist = train_one(
        pkg_model, tr, va,
        TrainConfig(n_epochs=3, lr=3e-3, batch=16, clip=1.0, lam_res=1e-2, L=20, L_eval=40),
        ckpt_path=tmp_path / "pkg.pth", verbose=False,
    )

    torch.manual_seed(0)
    ref_model = nb_svd["build_clamp"]()
    torch.manual_seed(42)
    _, ref_hist = nb_svd["train_one"](
        ref_model, n_epochs=3, lr=3e-3, batch=16, clip=1.0,
        ckpt_path=str(tmp_path / "ref.pth"), lam_res=1e-2, verbose=False,
    )

    for key in ("train", "val_extrap", "res", "zmax"):
        assert len(pkg_hist[key]) == len(ref_hist[key]) == 3
        for i, (a, b) in enumerate(zip(pkg_hist[key], ref_hist[key])):
            assert a == pytest.approx(b, abs=1e-12), f"hist[{key!r}][{i}]"
    assert pkg_hist["diverged_at"] == ref_hist["diverged_at"]
