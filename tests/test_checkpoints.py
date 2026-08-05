"""The committed checkpoints must load into the packaged classes.

The `.pth` files under `examples/duffing/` are bare state dicts keyed by module
attribute name -- they carry no architecture metadata at all. That makes every
attribute name in `ftnode.latent` and `ftnode.control.operator` part of the
serialization format. This module is the guard: rename `g_net`, `L_net`, `S_net`,
`encoder`, `dynamics`, `decoder`, `net`, `c` or the `_eye` buffer and these fail.
"""
import pathlib

import pytest
import torch

from ftnode.control import FrozenLatentPlant, SplitOperator
from ftnode.latent import build_clamp

DUFFING_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "duffing"

ID_CKPT = DUFFING_DIR / "best-ctrl-id-svdclamp-seed0.pth"
PSI_CKPT = DUFFING_DIR / "best-ctrl-psi-seed0.pth"

pytestmark = pytest.mark.skipif(
    not ID_CKPT.exists() or not PSI_CKPT.exists(),
    reason="committed duffing checkpoints not present",
)


def test_id_checkpoint_loads_strictly(model_cfg, budget):
    model = build_clamp(model_cfg, budget)
    sd = torch.load(ID_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_psi_checkpoint_loads_strictly(budget):
    fpsi = SplitOperator.from_budget(budget)
    sd = torch.load(PSI_CKPT, map_location="cpu")
    missing, unexpected = fpsi.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_frozen_plant_from_checkpoint(model_cfg, budget):
    plant = FrozenLatentPlant.from_checkpoint(ID_CKPT, model_cfg, budget)
    assert plant.m == budget.m
    assert all(not p.requires_grad for p in plant.model.parameters())
    assert not plant.model.training


def test_from_checkpoint_reports_a_missing_file(model_cfg, budget):
    """A wrong working directory should say so, not fail deep inside torch.load."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        FrozenLatentPlant.from_checkpoint(DUFFING_DIR / "does-not-exist.pth", model_cfg, budget)


def test_loaded_plant_produces_finite_dynamics(model_cfg, budget):
    plant = FrozenLatentPlant.from_checkpoint(ID_CKPT, model_cfg, budget)
    z = (2 * torch.rand(64, budget.m, generator=torch.Generator().manual_seed(0)) - 1) * 2.0
    u = torch.zeros(64, 1)
    assert torch.isfinite(plant.F(z, u)).all()
    assert torch.isfinite(plant.A(z)).all()
    assert torch.isfinite(plant.DuF(z, u)).all()


def test_trained_model_still_satisfies_the_kappa_bound(model_cfg, budget, latent_box):
    """The cap is structural, so training cannot have broken it -- confirm on real weights."""
    from ftnode.diagnostics import A_stats

    plant = FrozenLatentPlant.from_checkpoint(ID_CKPT, model_cfg, budget)
    maxre, smax, kappa = A_stats(plant.model.dynamics, latent_box)
    assert kappa.max() <= budget.kappa_max + 1e-4
    assert smax.max() <= budget.sigma_max + 1e-4
    assert maxre.max() <= -budget.sigma_min + 1e-4
