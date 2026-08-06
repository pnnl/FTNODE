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
from ftnode.latent import build_clamp, migrate_flat_state_dict

DUFFING_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "duffing"

ID_CKPT = DUFFING_DIR / "best-ctrl-id-svdclamp-seed0.pth"
PSI_CKPT = DUFFING_DIR / "best-ctrl-psi-seed0.pth"

pytestmark = pytest.mark.skipif(
    not ID_CKPT.exists() or not PSI_CKPT.exists(),
    reason="committed duffing checkpoints not present",
)


def test_id_checkpoint_loads_strictly(model_cfg, budget):
    """The committed checkpoint predates the operator/equilibrium split.

    It is deliberately not rewritten -- `duffing_learned_splitting_control.ipynb`
    is frozen and loads this same file with its own inline *flat* class
    definitions, so rewriting the binary would silently break it. The shim
    re-keys it on the way in instead.
    """
    model = build_clamp(model_cfg, budget)
    sd = torch.load(ID_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(migrate_flat_state_dict(sd, model), strict=True)
    assert not missing and not unexpected


def test_the_committed_checkpoint_really_is_in_the_legacy_layout(model_cfg, budget):
    """Guards the shim against becoming dead code that silently stops being tested.

    If a future change rewrites the `.pth` files, this fails and forces the
    decision to be explicit rather than discovered later by the frozen notebook.
    """
    model = build_clamp(model_cfg, budget)
    sd = torch.load(ID_CKPT, map_location="cpu")
    assert "dynamics.L_net.net.0.weight" in sd, "checkpoint is no longer flat"
    with pytest.raises(RuntimeError):
        model.load_state_dict(sd, strict=True)


def test_migration_is_idempotent(model_cfg, budget):
    """Applying it to an already-nested dict must be a no-op, so callers need no branch."""
    model = build_clamp(model_cfg, budget)
    once = migrate_flat_state_dict(torch.load(ID_CKPT, map_location="cpu"), model)
    twice = migrate_flat_state_dict(once, model)
    assert sorted(once) == sorted(twice)
    assert all(torch.equal(once[k], twice[k]) for k in once)


def test_migration_rejects_a_state_dict_it_cannot_place(model_cfg, budget):
    """A mis-routed name must fail loudly here, not as a partial load far away."""
    model = build_clamp(model_cfg, budget)
    sd = torch.load(ID_CKPT, map_location="cpu")
    sd["dynamics.mystery_net.weight"] = torch.zeros(1)
    with pytest.raises(KeyError, match="did not land on the target layout"):
        migrate_flat_state_dict(sd, model)


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
