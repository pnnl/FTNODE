"""The experiment specification: everything one ``ftnode-train`` invocation needs.

An experiment file is read into :class:`ExperimentSpec`, and the *resolved* spec is
written back into the run directory as ``run.yaml``.  Those two files are
deliberately distinct.  The file under ``experiments/`` is an input and will drift
as it is edited; the snapshot inside the run is the only record of what actually
produced the checkpoints, and the checkpoints are bare state dicts that carry no
architecture metadata of their own.  :func:`ftnode.experiments.run.load_run` reads
the snapshot and never the source.
"""
from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, replace

import yaml

from ..latent import KappaBudget, LatentModelConfig
from ..systems import DuffingDataConfig
from ..train import TrainConfig
from ..utils import _from_dict
from .registry import Variant, resolve_variants

__all__ = ["ExperimentSpec", "RunPaths", "spec_from_yaml", "spec_from_dict"]


@dataclass(frozen=True)
class RunPaths:
    """Layout of a run directory.  One place that knows the filenames."""

    root: pathlib.Path

    @property
    def run_yaml(self) -> pathlib.Path:
        return self.root / "run.yaml"

    @property
    def rollouts(self) -> pathlib.Path:
        return self.root / "rollouts.npz"

    def variant_dir(self, slug: str) -> pathlib.Path:
        return self.root / "variants" / slug

    def ckpt(self, slug: str, seed: int) -> pathlib.Path:
        return self.variant_dir(slug) / f"seed{seed}.pth"

    def hist(self, slug: str, seed: int) -> pathlib.Path:
        return self.variant_dir(slug) / f"seed{seed}.hist.json"


@dataclass(frozen=True)
class ExperimentSpec:
    """A whole experiment: shared configs, the variants to train, the seeds to use.

    The cross-section constraints below are checked at construction rather than
    left to fail deep inside training.  Every one of them is silent otherwise: a
    ``tau`` mismatch trains an encoder against windows of the wrong width, and a
    ``budget``/``model`` mismatch produces a model whose stated kappa bound is not
    the one it was built with.  The duffing notebooks keep these in sync by
    deriving one from the other in code; an experiment file cannot, so it gets a
    check instead.
    """

    name: str
    out: pathlib.Path
    budget: KappaBudget
    model: LatentModelConfig
    train: TrainConfig
    data_train: DuffingDataConfig
    data_val: DuffingDataConfig
    variants: tuple[Variant, ...]
    seeds: tuple[int, ...]

    def __post_init__(self):
        def bad(msg):
            raise ValueError(f"experiment {self.name!r}: {msg}")

        if self.model.m != self.budget.m:
            bad(f"model.m ({self.model.m}) != budget.m ({self.budget.m})")
        if self.model.operator.sigma_min != self.budget.sigma_min:
            bad(
                f"model.operator.sigma_min ({self.model.operator.sigma_min}) != "
                f"budget.sigma_min ({self.budget.sigma_min}); the operator would be "
                "built with a different contraction floor than the budget assumes"
            )
        tau = self.model.encoder.tau
        for label, data in (("data_train", self.data_train), ("data_val", self.data_val)):
            if data.tau != tau:
                bad(f"{label}.tau ({data.tau}) != model.encoder.tau ({tau})")
        if self.data_train.L != self.train.L:
            bad(f"data_train.L ({self.data_train.L}) != train.L ({self.train.L})")
        if self.data_val.L != self.train.L_eval:
            bad(f"data_val.L ({self.data_val.L}) != train.L_eval ({self.train.L_eval})")
        if self.data_train.seed == self.data_val.seed:
            bad("data_train.seed == data_val.seed; the validation set would repeat training data")
        if not self.seeds:
            bad("no seeds given")
        if len(set(self.seeds)) != len(self.seeds):
            bad(f"duplicate seeds {sorted(self.seeds)}")

    # ------------------------------------------------------------------ derived

    @property
    def paths(self) -> RunPaths:
        return RunPaths(pathlib.Path(self.out))

    def model_for(self, variant: Variant) -> LatentModelConfig:
        """Specialize the shared model config onto one variant's two axes.

        The ``kind`` fields of ``model.operator``/``model.equilibrium`` in the spec
        are placeholders -- the ``variants`` list owns that choice, and this is
        where it is applied.  Everything else in those sections (widths, depths,
        ``sigma_min``, ``R_g``) is shared across variants so the comparison
        isolates structure.
        """
        if variant.is_baseline:
            return self.model
        return replace(
            self.model,
            operator=replace(self.model.operator, kind=variant.operator),
            equilibrium=replace(self.model.equilibrium, kind=variant.equilibrium),
        )

    def jobs(self):
        """Every ``(variant, seed)`` pair, in the notebooks' order: variant then seed."""
        for variant in self.variants:
            for seed in self.seeds:
                yield variant, seed

    def select(self, only=None, seeds=None) -> "ExperimentSpec":
        """A narrowed copy, for sharding a run across processes.

        Each ``(variant, seed)`` is trained from its own fresh seed, so a shard is
        bitwise identical to the corresponding slice of the full run -- which is
        what makes fanning out across tmux panes safe.
        """
        variants = self.variants
        if only:
            missing = set(only) - {v.slug for v in variants}
            if missing:
                raise ValueError(f"--only names variants not in the spec: {sorted(missing)}")
            variants = tuple(v for v in variants if v.slug in set(only))
        chosen = self.seeds
        if seeds is not None:
            missing = set(seeds) - set(chosen)
            if missing:
                raise ValueError(f"--seeds names seeds not in the spec: {sorted(missing)}")
            chosen = tuple(s for s in chosen if s in set(seeds))
        return replace(self, variants=variants, seeds=chosen)

    # ------------------------------------------------------------ serialization

    def to_dict(self) -> dict:
        """Plain-data form, for the ``run.yaml`` snapshot."""
        return {
            "name": self.name,
            "out": str(self.out),
            "budget": dataclasses.asdict(self.budget),
            "model": dataclasses.asdict(self.model),
            "train": dataclasses.asdict(self.train),
            "data_train": dataclasses.asdict(self.data_train),
            "data_val": dataclasses.asdict(self.data_val),
            "variants": [dataclasses.asdict(v) for v in self.variants],
            "seeds": list(self.seeds),
        }


_SECTIONS = {
    "budget": KappaBudget,
    "model": LatentModelConfig,
    "train": TrainConfig,
    "data_train": DuffingDataConfig,
    "data_val": DuffingDataConfig,
}
_REQUIRED = {"name", "out", "variants", "seeds"}


def spec_from_dict(data: dict) -> ExperimentSpec:
    """Build a spec from plain data, rejecting unknown keys at every level.

    Section reconstruction goes through :func:`ftnode.utils._from_dict`, which
    already recurses into nested dataclass fields (``model.encoder`` and friends)
    and raises on unknown keys rather than silently defaulting -- so an experiment
    file written against an older schema fails loudly.
    """
    known = _REQUIRED | set(_SECTIONS)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"experiment spec has no fields {sorted(unknown)}")
    missing = _REQUIRED - set(data)
    if missing:
        raise ValueError(f"experiment spec is missing {sorted(missing)}")

    sections = {
        key: _from_dict(cls, data.get(key) or {}) for key, cls in _SECTIONS.items()
    }
    return ExperimentSpec(
        name=data["name"],
        out=pathlib.Path(data["out"]),
        variants=resolve_variants(data["variants"]),
        seeds=tuple(data["seeds"]),
        **sections,
    )


def spec_from_yaml(path) -> ExperimentSpec:
    """Read an experiment file (or a run's ``run.yaml`` snapshot) into a spec.

    A snapshot carries an extra ``meta`` block, which is provenance rather than
    settings and is dropped here; read it with
    :func:`ftnode.experiments.run.read_meta` if you want it.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    data.pop("meta", None)
    return spec_from_dict(data)
