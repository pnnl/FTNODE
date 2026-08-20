"""The experiment runner: does it produce what the notebook loop produced?

`tests/test_notebook_equivalence.py` asks whether the package still reproduces the
frozen notebooks. This file asks the same question one level up: does driving
training from a YAML file and a CLI give bitwise what the notebooks' own multi-seed
cell gives? A runner is exactly the shape of change that reintroduces a §3.2-class
silent break -- reseed in the wrong place and every number shifts, with no error
and no failing shape assertion.

Everything here runs on tiny data for 2 epochs. Nothing trains to convergence;
these are identity checks, not experiments.
"""
import dataclasses
import json
import pathlib

import numpy as np
import pytest
import torch

from ftnode.experiments import (
    REGISTRY,
    ExperimentSpec,
    Variant,
    build_variant,
    load_run,
    resolve_variants,
    run_experiment,
    spec_from_dict,
    spec_from_yaml,
    variant_from_spec,
)
from ftnode.latent import (
    EncoderConfig,
    KappaBudget,
    LatentModelConfig,
    OperatorConfig,
    build_clamp,
    build_latent_node,
    build_unbounded,
    build_youla,
)
from ftnode.systems import make_dataset
from ftnode.train import TrainConfig, restore_best, train_one

CPU = torch.device("cpu")

BASE = dict(
    name="t",
    out="",
    budget=dict(sigma_min=0.1, kappa_max=25.0, skew_frac=0.6, m=4),
    model=dict(m=4, encoder=dict(tau=8), operator=dict(sigma_min=0.1)),
    train=dict(n_epochs=2, lr=3e-3, batch=16, lam_res=1e-2, L=10, L_eval=20),
    data_train=dict(n_traj=16, L=10, tau=8, seed=0),
    data_val=dict(n_traj=8, L=20, tau=8, seed=1),
    variants=["l-n", "l-ft-k-svd-clamp"],
    seeds=[0, 1],
)


def _spec(tmp_path, **over) -> ExperimentSpec:
    data = {**BASE, "out": str(tmp_path / "run"), **over}
    return spec_from_dict(data)


# ------------------------------------------------------------------- registry


def test_registry_slugs_and_axes_are_stable():
    """Slugs name run directories; changing one orphans existing checkpoints."""
    assert sorted(REGISTRY) == [
        "l-ft-k-svd-clamp", "l-ft-k-youla", "l-ft-unbounded", "l-n",
    ]
    assert REGISTRY["l-ft-k-svd-clamp"].operator == "svd_clamp"
    assert REGISTRY["l-ft-k-youla"].operator == "youla"
    assert REGISTRY["l-n"].is_baseline and REGISTRY["l-n"].kind == "ln"
    assert all(v.kind == "ft" for k, v in REGISTRY.items() if k != "l-n")


def test_a_variant_needs_both_axes_or_neither():
    """3.5: LatentNODE is on neither axis. Naming one is a spec error, not a variant."""
    with pytest.raises(ValueError, match="must be given together"):
        Variant("x", "X", "svd_clamp", None)
    with pytest.raises(ValueError, match="must be given together"):
        variant_from_spec({"operator": "youla"})


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"operator": "nope", "equilibrium": "tanh_mlp"}, "unknown operator"),
        ({"operator": "youla", "equilibrium": "nope"}, "unknown equilibrium"),
        ("no-such-slug", "unknown variant"),
        ({"operator": "youla", "equilibrium": "tanh_mlp", "bogus": 1}, "no fields"),
    ],
)
def test_bad_variant_specs_raise(spec, message):
    with pytest.raises(ValueError, match=message):
        variant_from_spec(spec)


def test_inline_pairs_need_no_registry_entry():
    """The point of the two-axis form: a new g is sweepable the day it is written.

    Adding an equilibrium map means a module plus one `G_KINDS` line in
    `ftnode.latent.equilibrium` -- and nothing at all in this package. `REGISTRY`
    is a set of named shortcuts, not a gate.
    """
    v = variant_from_spec({"operator": "youla", "equilibrium": "tanh_mlp"})
    assert v.operator == "youla" and v.equilibrium == "tanh_mlp"
    assert v.slug == "youla+tanh_mlp"
    assert v.slug not in REGISTRY


def test_variants_carry_no_styling(tmp_path):
    """Styling belongs where the figure is drawn, not in the experiment manager.

    A color on `Variant` would mean editing the package to restyle a plot, and
    would still not cover line style, marker or alpha. `Run` exposes labels only.
    """
    assert not hasattr(REGISTRY["l-n"], "color")
    assert not any(f.name == "color" for f in dataclasses.fields(Variant))

    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    run = load_run(spec.out)
    assert not hasattr(run, "colors")
    assert set(run.names) == set(run.slugs)


def test_a_color_in_a_variant_spec_is_rejected_and_says_why():
    """An experiment file carrying styling should fail with a pointer, not silently."""
    with pytest.raises(ValueError, match="styling belongs where the figure is drawn"):
        variant_from_spec({"operator": "youla", "equilibrium": "tanh_mlp", "color": "C7"})


def test_duplicate_slugs_are_rejected():
    """Two variants sharing a slug would train into the same directory."""
    with pytest.raises(ValueError, match="duplicate variant slug"):
        resolve_variants(["l-n", "l-n"])
    with pytest.raises(ValueError, match="at least one variant"):
        resolve_variants([])


# ----------------------------------------------------------------------- spec


@pytest.mark.parametrize(
    "over,message",
    [
        ({"data_train": {**BASE["data_train"], "tau": 4}}, "tau"),
        ({"budget": {**BASE["budget"], "m": 6}}, "model.m"),
        ({"model": {**BASE["model"], "operator": {"sigma_min": 0.5}}}, "sigma_min"),
        ({"train": {**BASE["train"], "L": 99}}, "data_train.L"),
        ({"train": {**BASE["train"], "L_eval": 99}}, "data_val.L"),
        ({"data_val": {**BASE["data_val"], "seed": 0}}, "seed"),
        ({"seeds": [0, 0]}, "duplicate seeds"),
        ({"seeds": []}, "no seeds"),
    ],
)
def test_spec_rejects_inconsistent_sections(tmp_path, over, message):
    """Each of these is silent otherwise -- wrong-width windows, or a stated kappa
    bound the operator was not actually built with."""
    with pytest.raises(ValueError, match=message):
        _spec(tmp_path, **over)


def test_spec_rejects_unknown_keys_at_every_level(tmp_path):
    with pytest.raises(ValueError, match="no fields"):
        _spec(tmp_path, bogus=1)
    with pytest.raises(ValueError, match="LatentModelConfig has no fields"):
        _spec(tmp_path, model={**BASE["model"], "bogus": 1})
    with pytest.raises(ValueError, match="EncoderConfig has no fields"):
        _spec(tmp_path, model={**BASE["model"], "encoder": {"tau": 8, "bogus": 1}})


def test_model_for_specializes_only_the_two_kinds(tmp_path):
    spec = _spec(tmp_path)
    cfg = spec.model_for(REGISTRY["l-ft-k-youla"])
    assert cfg.operator.kind == "youla" and cfg.equilibrium.kind == "tanh_mlp"
    # everything else is shared, so the comparison isolates structure
    assert cfg.operator.hidden == spec.model.operator.hidden
    assert cfg.encoder == spec.model.encoder
    # the baseline has no axes to specialize
    assert spec.model_for(REGISTRY["l-n"]) == spec.model


def test_jobs_order_matches_the_notebook(tmp_path):
    """variant outer, seed inner -- the order cell 11 uses."""
    spec = _spec(tmp_path)
    assert [(v.slug, s) for v, s in spec.jobs()] == [
        ("l-n", 0), ("l-n", 1), ("l-ft-k-svd-clamp", 0), ("l-ft-k-svd-clamp", 1),
    ]


def test_select_rejects_names_not_in_the_spec(tmp_path):
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="--only"):
        spec.select(only=["nope"])
    with pytest.raises(ValueError, match="--seeds"):
        spec.select(seeds=[9])


def test_yaml_round_trip(tmp_path):
    import yaml

    spec = _spec(tmp_path)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec.to_dict()))
    assert spec_from_yaml(path) == spec


def test_the_shipped_experiment_files_load():
    """A broken example costs someone a debugging session before their first run."""
    root = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "duffing"
    for path in sorted(root.glob("*.yaml")):
        spec = spec_from_yaml(path)
        assert spec.variants and spec.seeds, path.name


# ------------------------------------------------------- the equivalence gate


def _notebook_loop(spec, tmp_path):
    """The duffing notebooks' multi-seed cell, inline, for one spec.

    Deliberately written out rather than reusing the runner: this is the reference
    the runner has to match, so sharing code with it would defeat the test.
    """
    cfg, budget = spec.model, spec.budget
    build = {
        "l-n": lambda: build_latent_node(spec.model_for(REGISTRY["l-n"])),
        "l-ft-unbounded": lambda: build_unbounded(spec.model_for(REGISTRY["l-ft-unbounded"])),
        "l-ft-k-svd-clamp": lambda: build_clamp(spec.model_for(REGISTRY["l-ft-k-svd-clamp"]), budget),
        "l-ft-k-youla": lambda: build_youla(spec.model_for(REGISTRY["l-ft-k-youla"]), budget),
    }
    train = make_dataset(spec.data_train)
    val = make_dataset(spec.data_val)
    out = {}
    for variant in spec.variants:
        for seed in spec.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build[variant.slug]().to(CPU)
            model, hist = train_one(
                model, train, val, spec.train,
                ckpt_path=tmp_path / f"nb-{variant.slug}-{seed}.pth",
                label=f"{variant.slug}-s{seed}", device=CPU, verbose=False,
            )
            restore_best(model, hist, CPU, verbose=False)
            out[(variant.slug, seed)] = (model.state_dict(), hist)
    return out


HIST_KEYS = ("train", "val_extrap", "zmax", "res", "best_val", "best_epoch", "diverged_at")


def test_runner_is_bitwise_identical_to_the_notebook_loop(tmp_path):
    """THE gate. Everything else in this file is secondary to it.

    Covers all four variants so the baseline (no g, no A) and both kappa-bounded
    operators are all exercised.
    """
    spec = _spec(tmp_path, variants=list(REGISTRY), seeds=[0, 1])
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    reference = _notebook_loop(spec, tmp_path)

    run = load_run(spec.out)
    models, histories = run.models(), run.histories
    for variant in spec.variants:
        for si, seed in enumerate(spec.seeds):
            ref_sd, ref_hist = reference[(variant.slug, seed)]
            got_sd = models[variant.slug][si].state_dict()
            got_hist = histories[variant.slug][si]
            assert sorted(got_sd) == sorted(ref_sd), variant.slug
            for k in ref_sd:
                assert torch.equal(got_sd[k], ref_sd[k]), f"{variant.slug} s{seed}: {k}"
            for k in HIST_KEYS:
                assert got_hist[k] == ref_hist[k], f"{variant.slug} s{seed}: hist[{k!r}]"


def test_reseeding_makes_every_job_independent(tmp_path):
    """Why sharding is safe: a job's result cannot depend on what ran before it."""
    full = _spec(tmp_path / "a")
    run_experiment(full, device=CPU, rollouts=False, verbose=False)

    shard = _spec(tmp_path / "b")
    run_experiment(shard, only=["l-ft-k-svd-clamp"], seeds=[1],
                   device=CPU, rollouts=False, verbose=False)

    a = torch.load(full.paths.ckpt("l-ft-k-svd-clamp", 1), map_location="cpu")
    b = torch.load(shard.paths.ckpt("l-ft-k-svd-clamp", 1), map_location="cpu")
    assert sorted(a) == sorted(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_the_runner_does_not_flip_global_torch_settings(tmp_path):
    """A runner should seed and nothing else.

    This is the real reason `ftnode.utils.set_global_seed` is not used here.
    Substituting it does *not* change any number -- it calls the same two seeding
    functions, so on CPU the stream is identical and every other test in this file
    still passes. What it does do is enable `use_deterministic_algorithms` and
    disable `cudnn.benchmark` process-wide, which changes GPU kernel selection and
    throughput for everything downstream in the same process. That side effect is
    what this pins.
    """
    def snapshot():
        return (
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )

    original = snapshot()
    try:
        # Establish a known state first. Reading whatever the process happens to be
        # in makes this test order-dependent: an earlier test that flipped the flags
        # would leave `before` already equal to `after`, and the check would pass
        # against exactly the mutation it exists to catch.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        before = snapshot()

        run_experiment(_spec(tmp_path), device=CPU, rollouts=False, verbose=False)

        assert snapshot() == before
    finally:
        torch.use_deterministic_algorithms(original[0])
        torch.backends.cudnn.benchmark = original[1]
        torch.backends.cudnn.deterministic = original[2]


def test_lam_res_is_inert_for_the_baseline(tmp_path):
    """Why `lam_res` can live on TrainConfig alone rather than per variant.

    `train_one` gates the residual penalty on `hasattr(dynamics, 'g')`, and
    LatentNODE has none -- so the notebooks' per-variant 0.0 override changed
    nothing. Verified rather than assumed.
    """
    cfg = LatentModelConfig(m=4, encoder=EncoderConfig(tau=8))
    train = make_dataset(spec_from_dict(BASE | {"out": str(tmp_path)}).data_train)
    val = make_dataset(spec_from_dict(BASE | {"out": str(tmp_path)}).data_val)

    def go(lam):
        torch.manual_seed(0)
        np.random.seed(0)
        model = build_latent_node(cfg)
        model, hist = train_one(
            model, train, val,
            TrainConfig(n_epochs=2, batch=16, lam_res=lam, L=10, L_eval=20),
            ckpt_path=tmp_path / f"ln-{lam}.pth", verbose=False,
        )
        return model.state_dict(), hist

    sd0, h0 = go(0.0)
    sd1, h1 = go(1e-2)
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    assert all(h0[k] == h1[k] for k in HIST_KEYS)
    assert h0["res"] == [0.0] * 2


# ------------------------------------------------------------- run directory


def test_run_directory_layout(tmp_path):
    spec = _spec(tmp_path)
    paths = run_experiment(spec, device=CPU, rollouts=True, verbose=False)
    assert paths.run_yaml.exists() and paths.rollouts.exists()
    for variant, seed in spec.jobs():
        assert paths.ckpt(variant.slug, seed).exists()
        assert paths.hist(variant.slug, seed).exists()


def test_hist_json_is_portable(tmp_path):
    """`ckpt_path` relative, and `inf` written as null -- see `_hist_to_json`."""
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    raw = json.loads(spec.paths.hist("l-n", 0).read_text())
    assert raw["ckpt_path"] == "variants/l-n/seed0.pth"
    assert not pathlib.Path(raw["ckpt_path"]).is_absolute()

    hist = load_run(spec.out).histories["l-n"][0]
    assert pathlib.Path(hist["ckpt_path"]).is_absolute()
    assert pathlib.Path(hist["ckpt_path"]).exists()


def test_a_run_directory_survives_being_moved(tmp_path):
    """Checkpoint paths are resolved against wherever the run actually is."""
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    moved = tmp_path / "relocated"
    pathlib.Path(spec.out).rename(moved)

    run = load_run(moved)
    assert run.spec.paths.root == moved
    hist = run.histories["l-n"][0]
    assert pathlib.Path(hist["ckpt_path"]).exists()
    assert run.models()["l-n"][0] is not None


def test_skip_existing_retrains_nothing(tmp_path):
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    before = {p: p.stat().st_mtime_ns for p in spec.paths.root.rglob("*.pth")}
    assert before

    run_experiment(spec, device=CPU, skip_existing=True, rollouts=False, verbose=False)
    after = {p: p.stat().st_mtime_ns for p in spec.paths.root.rglob("*.pth")}
    assert before == after


def test_the_snapshot_records_the_whole_experiment_even_from_a_shard(tmp_path):
    """A narrowed snapshot would leave `load_run` blind to the other shards' work.

    Two processes writing into one run directory is the intended way to use a
    multi-core box, so the last writer must not shrink the record.
    """
    spec = _spec(tmp_path, variants=["l-n", "l-ft-k-svd-clamp"], seeds=[0, 1])
    run_experiment(spec, only=["l-n"], seeds=[0], device=CPU, rollouts=False, verbose=False)
    run_experiment(spec, only=["l-ft-k-svd-clamp"], seeds=[1],
                   device=CPU, rollouts=False, verbose=False)

    run = load_run(spec.out)
    assert run.slugs == ["l-n", "l-ft-k-svd-clamp"]
    assert list(run.seeds) == [0, 1]
    assert len(run.meta["invocations"]) == 2
    # untrained jobs read back as None rather than exploding
    histories = run.histories
    assert histories["l-n"][0] is not None and histories["l-n"][1] is None
    assert histories["l-ft-k-svd-clamp"][0] is None
    assert histories["l-ft-k-svd-clamp"][1] is not None


def test_load_run_reads_only_the_snapshot(tmp_path):
    """The source experiment file may be edited or deleted after a run."""
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    run = load_run(spec.out)
    assert run.spec.budget == spec.budget
    assert run.spec.model == spec.model
    assert run.spec.train == spec.train


def test_load_run_on_a_non_run_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run.yaml"):
        load_run(tmp_path)


def test_rollout_cache_shapes(tmp_path):
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=True, verbose=False)
    rolls = load_run(spec.out).rollouts()
    n_seed, n_val = len(spec.seeds), spec.data_val.n_traj
    for slug, arrays in rolls.items():
        assert arrays["yhat"].shape == (n_seed, n_val, spec.train.L_eval + 1), slug
        assert arrays["z"].shape == (n_seed, n_val, spec.train.L_eval + 1, spec.model.m)
        assert np.isfinite(arrays["yhat"]).any(), slug


def test_no_rollouts_leaves_no_cache(tmp_path):
    spec = _spec(tmp_path)
    run_experiment(spec, device=CPU, rollouts=False, verbose=False)
    assert not spec.paths.rollouts.exists()
    assert load_run(spec.out).rollouts() is None


def test_build_variant_matches_the_named_builders(tmp_path):
    """`build_variant` must be the same construction the notebooks' builders do."""
    spec = _spec(tmp_path, variants=list(REGISTRY))
    pairs = [
        ("l-n", lambda: build_latent_node(spec.model_for(REGISTRY["l-n"]))),
        ("l-ft-unbounded", lambda: build_unbounded(spec.model_for(REGISTRY["l-ft-unbounded"]))),
        ("l-ft-k-svd-clamp",
         lambda: build_clamp(spec.model_for(REGISTRY["l-ft-k-svd-clamp"]), spec.budget)),
        ("l-ft-k-youla",
         lambda: build_youla(spec.model_for(REGISTRY["l-ft-k-youla"]), spec.budget)),
    ]
    for slug, reference in pairs:
        torch.manual_seed(0)
        got = build_variant(spec, REGISTRY[slug]).state_dict()
        torch.manual_seed(0)
        ref = reference().state_dict()
        assert sorted(got) == sorted(ref), slug
        for k in ref:
            assert torch.equal(got[k], ref[k]), f"{slug}: {k}"


# ------------------------------------------------------------------------ CLI


def test_parse_seeds_handles_lists_and_ranges():
    from ftnode.experiments.cli import parse_seeds

    assert parse_seeds(None) is None
    assert parse_seeds(["0", "1", "2"]) == [0, 1, 2]
    assert parse_seeds(["0-9"]) == list(range(10))
    assert parse_seeds(["0-3", "7"]) == [0, 1, 2, 3, 7]
    assert parse_seeds(["2", "2"]) == [2]


def test_cli_dry_run_trains_nothing(tmp_path, capsys):
    import yaml

    from ftnode.experiments.cli import main

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(tmp_path).to_dict()))
    assert main([str(spec_path), "--dry-run"]) == 0
    assert "l-ft-k-svd-clamp" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_cli_defaults_to_the_train_subcommand(tmp_path):
    """`ftnode-train <yaml>` must work without typing `train`."""
    import yaml

    from ftnode.experiments.cli import main

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(tmp_path).to_dict()))
    assert main([str(spec_path), "--dry-run"]) == 0
    assert main(["train", str(spec_path), "--dry-run"]) == 0
