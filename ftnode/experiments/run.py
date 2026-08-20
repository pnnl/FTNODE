"""Running an experiment, and loading a finished one back.

The unit of work is one ``(variant, seed)`` pair, and it is **hermetic**: the RNG
is reseeded immediately before the model is built, and ``make_dataset`` draws from
its own generator rather than global state, so a job's result does not depend on
what ran before it.  That is what makes ``--only``/``--seeds`` sharding across
processes produce bitwise the same checkpoints as one long run.

Everything a finished run needs for analysis lands in the run directory:
checkpoints, per-epoch histories, the resolved config snapshot, and optionally the
validation rollouts.  :func:`load_run` reads only those -- never the source
experiment file, which is free to drift.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import torch
import yaml

from ..latent import build_latent_ftnode, build_latent_node
from ..systems import make_dataset
from ..train import restore_best, rollout_y, train_one
from .registry import Variant
from .spec import ExperimentSpec, RunPaths, spec_from_dict

__all__ = [
    "build_variant",
    "train_job",
    "run_experiment",
    "compute_rollouts",
    "load_run",
    "Run",
    "run_metadata",
]


# ------------------------------------------------------------------- building


def build_variant(spec: ExperimentSpec, variant: Variant):
    """Build the model for one variant, on the CPU.

    Mirrors the duffing notebooks' ``build`` closures exactly, including that the
    model is constructed before being moved to a device -- moving it is not an RNG
    operation, but building it is, and the order the notebooks use is the order
    that reproduces them.
    """
    cfg = spec.model_for(variant)
    if variant.is_baseline:
        return build_latent_node(cfg)
    return build_latent_ftnode(cfg, spec.budget)


def train_job(
    spec: ExperimentSpec,
    variant: Variant,
    seed: int,
    train,
    val,
    *,
    ckpt_path,
    device=None,
    verbose: bool = True,
):
    """Train one ``(variant, seed)`` and restore its best weights.

    .. warning::
       Seeding **immediately before the build** is the reproducibility contract:
       the encoder, equilibrium map and operator all draw from the global torch RNG
       as they initialize, so moving these two lines after ``build_variant`` gives
       different weights for the same seed, with no error and correct-looking kappa
       values.  Pinned by ``tests/test_experiments.py``.

    .. note::
       :func:`ftnode.utils.set_global_seed` is deliberately **not** used here, but
       not because it would change the numbers -- it calls these same two functions,
       so on CPU the stream is identical and substituting it passes every test.  It
       is avoided because of its *side effects*: it flips
       ``torch.use_deterministic_algorithms`` and ``cudnn.benchmark`` process-wide,
       which alters GPU kernel selection and throughput, and it prints on every
       call -- forty times over a full run.  A runner should seed and nothing else.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_variant(spec, variant).to(device)
    model, hist = train_one(
        model, train, val, spec.train,
        ckpt_path=ckpt_path,
        label=f"{variant.slug}-s{seed}",
        device=device,
        verbose=verbose,
    )
    restore_best(model, hist, device, verbose=verbose)
    return model, hist


# ------------------------------------------------------------ history as JSON


def _hist_to_json(hist: dict, root: pathlib.Path) -> dict:
    """Serializable history.

    Two adjustments.  ``best_val`` is ``inf`` when no epoch ever improved; ``inf``
    is not valid JSON, so it is written as ``null`` and restored on read.  And
    ``ckpt_path`` is stored **relative to the run directory** -- ``train_one``
    records whatever path it was handed, and an absolute one would make the run
    directory unmovable and break ``restore_best`` from a notebook with a
    different working directory.
    """
    out = dict(hist)
    if not math.isfinite(out.get("best_val", math.inf)):
        out["best_val"] = None
    ckpt = out.get("ckpt_path")
    if ckpt:
        out["ckpt_path"] = str(pathlib.Path(ckpt).resolve().relative_to(root.resolve()))
    return out


def _hist_from_json(data: dict, root: pathlib.Path) -> dict:
    hist = dict(data)
    if hist.get("best_val") is None:
        hist["best_val"] = float("inf")
    if hist.get("ckpt_path"):
        hist["ckpt_path"] = str(root / hist["ckpt_path"])
    return hist


# ------------------------------------------------------------------- metadata


def run_metadata(device, argv=None) -> dict:
    """Provenance for one invocation: what ran, when, from which tree.

    ``git_dirty`` matters more than the SHA: a run produced from a modified working
    tree is not reproducible from that commit, and this is the only place that gets
    recorded.
    """
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
                cwd=pathlib.Path(__file__).resolve().parents[2],
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "device": str(device),
        "argv": [str(a) for a in (argv if argv is not None else sys.argv)],
        # str(): torch.__version__ is a TorchVersion, a str subclass that
        # yaml.safe_dump refuses to represent.
        "torch": str(torch.__version__),
    }


def write_snapshot(spec: ExperimentSpec, invocation: dict) -> None:
    """Write the resolved spec plus provenance to ``run.yaml``.

    This is the file :func:`load_run` reads.  The experiment file under
    ``experiments/`` is an input and may change after the run; the checkpoints are
    bare state dicts that cannot be rebuilt without the exact config, so the
    snapshot is what makes a run directory self-contained.

    ``spec`` must be the **whole** experiment, never a shard: a narrowed spec here
    would record only that shard's variants and seeds, and whichever process
    finished last would win, leaving ``load_run`` blind to the rest of the run.

    Invocations accumulate rather than overwrite, so a run assembled from several
    sharded processes -- or resumed after a kill -- keeps the full record of what
    produced it.
    """
    paths = spec.paths
    paths.root.mkdir(parents=True, exist_ok=True)

    created, invocations = invocation["at"], []
    if paths.run_yaml.exists():
        with open(paths.run_yaml) as fh:
            previous = (yaml.safe_load(fh) or {}).get("meta", {})
        created = previous.get("created", created)
        invocations = list(previous.get("invocations", []))
    invocations.append(invocation)

    meta = {"created": created, "invocations": invocations}
    with open(paths.run_yaml, "w") as fh:
        yaml.safe_dump({"meta": meta, **spec.to_dict()}, fh, sort_keys=False)


# -------------------------------------------------------------------- running


def run_experiment(
    spec: ExperimentSpec,
    *,
    only=None,
    seeds=None,
    device=None,
    skip_existing: bool = False,
    rollouts: bool = True,
    verbose: bool = True,
    argv=None,
) -> RunPaths:
    """Train ``(variant, seed)`` jobs from ``spec`` into its run directory.

    ``spec`` is always the **whole** experiment; ``only``/``seeds`` narrow which
    jobs *this process* executes.  Keeping those separate is what lets several
    shards write into one run directory without the snapshot losing track of the
    variants the other shards own.

    Args:
        spec: The complete experiment.
        only: Variant slugs to train; ``None`` for all of them.
        seeds: Seeds to train; ``None`` for all of them.
        device: Torch device; defaults to CUDA when available.
        skip_existing: Skip jobs whose checkpoint *and* history already exist, so
            re-running the identical command after a killed process resumes at
            ``(variant, seed)`` granularity.
        rollouts: Cache validation rollouts into ``rollouts.npz`` afterwards, so
            the analysis notebook does not recompute them.  Covers every variant in
            ``spec``, not just this shard's; jobs with no checkpoint stay ``NaN``.
        verbose: Per-epoch progress from ``train_one``.

    Returns:
        RunPaths: the run directory layout.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = spec.paths
    write_snapshot(spec, run_metadata(device, argv))
    shard = spec.select(only=only, seeds=seeds)

    train = make_dataset(spec.data_train).to(device)
    val = make_dataset(spec.data_val).to(device)
    if verbose:
        n = len(shard.variants) * len(shard.seeds)
        total = len(spec.variants) * len(spec.seeds)
        scope = f"{n} jobs" if n == total else f"{n} of {total} jobs (shard)"
        print(
            f"[{spec.name}] {scope} on {device}\n"
            f"[{spec.name}] train {tuple(train.W.shape)}  val {tuple(val.W.shape)}  "
            f"-> {paths.root}"
        )

    for variant, seed in shard.jobs():
        ckpt = paths.ckpt(variant.slug, seed)
        hist_path = paths.hist(variant.slug, seed)
        if skip_existing and ckpt.exists() and hist_path.exists():
            if verbose:
                print(f"[{variant.slug}-s{seed}] exists, skipping")
            continue
        ckpt.parent.mkdir(parents=True, exist_ok=True)

        _, hist = train_job(
            spec, variant, seed, train, val,
            ckpt_path=ckpt, device=device, verbose=verbose,
        )
        with open(hist_path, "w") as fh:
            json.dump(_hist_to_json(hist, paths.root), fh, indent=1)
        if verbose:
            print(
                f"[{variant.slug}-s{seed}] best_val {hist['best_val']:.3e} "
                f"@ ep {hist['best_epoch']}  diverged={hist['diverged_at']}"
            )

    if rollouts:
        compute_rollouts(spec, device=device, verbose=verbose)
    return paths


def compute_rollouts(spec: ExperimentSpec, *, device=None, verbose: bool = True) -> pathlib.Path:
    """Roll every trained model out over the validation set and cache the result.

    This is the one analysis step expensive enough to be worth precomputing: at the
    frozen settings it is 4 variants x 10 seeds x 600 RK4 steps over 64
    trajectories.  Stored as float32; ``z`` dominates the file size at
    ``n_seeds x n_val x (L_eval+1) x m``.

    Missing jobs are left as ``NaN`` rather than failing, so a partially trained
    run still produces a usable cache.
    """
    device = device or torch.device("cpu")
    paths = spec.paths
    val = make_dataset(spec.data_val).to(device)
    L, h, m = spec.train.L_eval, spec.train.h, spec.model.m
    n_seed, n_val = len(spec.seeds), val.W.shape[0]

    arrays = {}
    for variant in spec.variants:
        yh = np.full((n_seed, n_val, L + 1), np.nan, np.float32)
        zz = np.full((n_seed, n_val, L + 1, m), np.nan, np.float32)
        for si, seed in enumerate(spec.seeds):
            ckpt = paths.ckpt(variant.slug, seed)
            if not ckpt.exists():
                continue
            model = build_variant(spec, variant).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            with torch.no_grad():
                yhat, zs = rollout_y(model, val.W, val.U, L, h)
            y = yhat.cpu().numpy()
            yh[si] = np.where(np.isfinite(y), y, np.nan)
            zz[si] = zs.cpu().numpy()
        arrays[f"{variant.slug}/yhat"] = yh
        arrays[f"{variant.slug}/z"] = zz

    np.savez_compressed(paths.rollouts, **arrays)
    if verbose:
        mb = paths.rollouts.stat().st_size / 1e6
        print(f"[{spec.name}] cached validation rollouts -> {paths.rollouts} ({mb:.1f} MB)")
    return paths.rollouts


# -------------------------------------------------------------------- loading


class Run:
    """A finished run, loaded from its directory.

    Everything is keyed by variant **slug**, which is unique by construction;
    display labels come off :attr:`names`.  Styling is not a run's business.
    """

    def __init__(self, root, spec: ExperimentSpec, meta: dict):
        self.root = pathlib.Path(root)
        self.spec = spec
        self.meta = meta
        self._models: dict[tuple[str, str], list] = {}

    def __repr__(self):
        return (
            f"<Run {self.spec.name!r} at {self.root}: "
            f"{len(self.variants)} variants x {len(self.seeds)} seeds>"
        )

    # ------------------------------------------------------------- accessors

    @property
    def variants(self) -> tuple[Variant, ...]:
        return self.spec.variants

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.spec.seeds

    @property
    def slugs(self) -> list[str]:
        return [v.slug for v in self.variants]

    @property
    def names(self) -> dict[str, str]:
        """``slug -> display label``, for plot legends and axis titles.

        Labels only.  A run carries no styling: colors, line styles and markers
        belong wherever the figure is drawn, so that restyling a plot is an edit to
        the notebook rather than to the package.
        """
        return {v.slug: v.name for v in self.variants}

    @property
    def histories(self) -> dict[str, list[dict]]:
        """``slug -> [hist per seed]``, in the spec's seed order.

        Missing jobs come back as ``None`` so a partially finished run still loads.
        """
        out = {}
        for v in self.variants:
            per_seed = []
            for seed in self.seeds:
                path = self.spec.paths.hist(v.slug, seed)
                if not path.exists():
                    per_seed.append(None)
                    continue
                with open(path) as fh:
                    per_seed.append(_hist_from_json(json.load(fh), self.root))
            out[v.slug] = per_seed
        return out

    def models(self, device=None) -> dict[str, list]:
        """``slug -> [model per seed]``, rebuilt from the snapshot and loaded.

        The architecture is rebuilt from the stored config before loading, because
        the checkpoints are bare state dicts carrying no architecture metadata.
        Cached per device.
        """
        device = device or torch.device("cpu")
        key = str(device)
        out = {}
        for v in self.variants:
            cache_key = (v.slug, key)
            if cache_key not in self._models:
                per_seed = []
                for seed in self.seeds:
                    ckpt = self.spec.paths.ckpt(v.slug, seed)
                    if not ckpt.exists():
                        per_seed.append(None)
                        continue
                    model = build_variant(self.spec, v).to(device)
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                    per_seed.append(model.eval())
                self._models[cache_key] = per_seed
            out[v.slug] = self._models[cache_key]
        return out

    def datasets(self, device=None):
        """Regenerate ``(train, val)`` from the stored configs.

        Deterministic and cheap relative to training -- the datasets are not
        stored, only the settings that produce them.
        """
        device = device or torch.device("cpu")
        return (
            make_dataset(self.spec.data_train).to(device),
            make_dataset(self.spec.data_val).to(device),
        )

    def rollouts(self) -> dict[str, dict[str, np.ndarray]] | None:
        """Cached validation rollouts as ``slug -> {'yhat': ..., 'z': ...}``.

        ``None`` when the run was made with ``--no-rollouts``; recompute with
        :func:`compute_rollouts`.
        """
        path = self.spec.paths.rollouts
        if not path.exists():
            return None
        with np.load(path) as data:
            return {
                v.slug: {"yhat": data[f"{v.slug}/yhat"], "z": data[f"{v.slug}/z"]}
                for v in self.variants
                if f"{v.slug}/yhat" in data
            }


def load_run(root) -> Run:
    """Load a run directory written by :func:`run_experiment`.

    Reads the ``run.yaml`` snapshot only -- never the experiment file it came from,
    which may have been edited since.
    """
    root = pathlib.Path(root)
    snapshot = RunPaths(root).run_yaml
    if not snapshot.exists():
        raise FileNotFoundError(
            f"no run.yaml in {root} (resolved to {root.resolve()}, "
            f"cwd {pathlib.Path.cwd()})\n"
            "Point at a run directory produced by `ftnode-train`, not at an "
            "experiment file.  Note that a notebook's working directory is its own "
            "folder, so a run under the repo root needs a relative path such as "
            "'../../runs/<name>'."
        )
    with open(snapshot) as fh:
        data = yaml.safe_load(fh) or {}
    meta = data.pop("meta", {})
    spec = spec_from_dict(data)
    # The snapshot records the `out` it was written with; honour where it actually
    # is, so a run directory survives being moved or copied off a cluster.
    spec = dataclasses.replace(spec, out=root)
    return Run(root, spec, meta)
