"""Headless multi-seed training runs, and loading them back for analysis.

Training used to happen only inside notebooks, which meant a run at the paper
settings held a kernel open for hours and left its results in notebook memory --
re-plotting meant re-training.  This package splits the two:

.. code-block:: bash

    nohup uv run ftnode-train experiments/duffing/kappa_full.yaml > run.log 2>&1 &

.. code-block:: python

    run = load_run("runs/kappa_full")     # in a notebook, in seconds
    models, histories = run.models(), run.histories

An experiment file names shared configs, a list of variants and a list of seeds.
A **variant** is a pair of registry keys -- one operator from
:data:`ftnode.latent.A_KINDS`, one equilibrium map from
:data:`ftnode.latent.G_KINDS` -- so a newly written ``g`` can be swept against
every operator without any code change here.

Each ``(variant, seed)`` is trained hermetically, reseeded immediately before its
model is built, which is what lets a run be sharded across processes with bitwise
identical results.
"""
from __future__ import annotations

from .registry import REGISTRY, Variant, resolve_variants, variant_from_spec
from .run import (
    Run,
    build_variant,
    compute_rollouts,
    load_run,
    run_experiment,
    run_metadata,
    train_job,
)
from .spec import ExperimentSpec, RunPaths, spec_from_dict, spec_from_yaml

__all__ = [
    # registry
    "Variant",
    "REGISTRY",
    "variant_from_spec",
    "resolve_variants",
    # spec
    "ExperimentSpec",
    "RunPaths",
    "spec_from_yaml",
    "spec_from_dict",
    # running
    "build_variant",
    "train_job",
    "run_experiment",
    "compute_rollouts",
    "run_metadata",
    # loading
    "load_run",
    "Run",
]
