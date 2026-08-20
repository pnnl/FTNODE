"""``ftnode-train`` -- run an experiment file headlessly.

.. code-block:: bash

    uv run ftnode-train experiments/duffing/kappa_quick.yaml
    uv run ftnode-train <yaml> --only l-ft-k-svd-clamp --seeds 3     # one shard
    uv run ftnode-train <yaml> --skip-existing                       # resume
    nohup uv run ftnode-train <yaml> > runs/kappa_full.log 2>&1 &

Sharding is safe: each ``(variant, seed)`` is reseeded immediately before its model
is built, so a shard produces bitwise the same checkpoints as the corresponding
slice of one long run.  Fan out across tmux panes and merge by pointing every
process at the same ``--out``.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import torch

from .run import load_run, run_experiment
from .spec import spec_from_yaml

__all__ = ["main", "parse_seeds"]

#: Subcommands.  `train` is the default, so the common case is `ftnode-train <yaml>`
#: rather than `ftnode-train train <yaml>`.
_SUBCOMMANDS = ("train", "ctrl", "show")


def parse_seeds(tokens):
    """Parse ``--seeds`` as a mix of integers and inclusive ``lo-hi`` ranges.

    ``--seeds 0 1 2``, ``--seeds 0-9`` and ``--seeds 0-3 7`` all work.  Returns
    ``None`` when nothing was given, meaning "every seed in the spec".
    """
    if not tokens:
        return None
    out = []
    for token in tokens:
        if "-" in token[1:]:  # not a leading minus sign
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return sorted(dict.fromkeys(out))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftnode-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an identification experiment (default)")
    train.add_argument("spec", type=pathlib.Path, help="experiment YAML file")
    train.add_argument("--out", type=pathlib.Path, default=None,
                       help="override the spec's output directory")
    train.add_argument("--only", action="append", metavar="SLUG", default=None,
                       help="restrict to this variant; repeatable")
    train.add_argument("--seeds", nargs="+", metavar="SEED", default=None,
                       help="restrict to these seeds; accepts 0 1 2 or 0-9")
    train.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    train.add_argument("--skip-existing", action="store_true",
                       help="skip jobs whose checkpoint and history already exist")
    train.add_argument("--no-rollouts", action="store_true",
                       help="do not cache validation rollouts afterwards")
    train.add_argument("--quiet", action="store_true", help="suppress per-epoch progress")
    train.add_argument("--dry-run", action="store_true",
                       help="print the resolved jobs and exit without training")

    show = sub.add_parser("show", help="summarize a finished run directory")
    show.add_argument("run", type=pathlib.Path, help="run directory containing run.yaml")

    # The control stage is a different loop (`ftnode.control.train_psi`) with its own
    # config and run-directory shape. The seam is here so it drops in without
    # reshaping the CLI; see the plan in markdown/ for the intended arguments.
    ctrl = sub.add_parser("ctrl", help="train a control policy (not yet implemented)")
    ctrl.add_argument("--plant", type=pathlib.Path, required=True,
                      help="identified-model checkpoint to freeze as the plant")
    return parser


def _cmd_train(args) -> int:
    spec = spec_from_yaml(args.spec)
    if args.out is not None:
        spec = dataclasses.replace(spec, out=args.out)
    seeds = parse_seeds(args.seeds)

    if args.dry_run:
        shard = spec.select(only=args.only, seeds=seeds)
        print(f"{spec.name}: {len(shard.variants)} variants x {len(shard.seeds)} seeds "
              f"-> {spec.out}")
        for variant, seed in shard.jobs():
            print(f"  {variant.slug:20s} seed {seed}  "
                  f"({variant.operator or 'latent-node'} / {variant.equilibrium or '-'})")
        return 0

    # The full spec goes in, the selection narrows only which jobs run here -- so
    # concurrent shards sharing one --out all record the same complete snapshot.
    run_experiment(
        spec,
        only=args.only,
        seeds=seeds,
        device=torch.device(args.device) if args.device else None,
        skip_existing=args.skip_existing,
        rollouts=not args.no_rollouts,
        verbose=not args.quiet,
    )
    return 0


def _cmd_show(args) -> int:
    run = load_run(args.run)
    print(run)
    print(f"  created     {run.meta.get('created')}")
    invocations = run.meta.get("invocations", [])
    print(f"  invocations {len(invocations)}"
          + ("  (sharded or resumed)" if len(invocations) > 1 else ""))
    if invocations:
        last = invocations[-1]
        dirty = "  [DIRTY TREE]" if last.get("git_dirty") else ""
        print(f"  last run    {last.get('at')} on {last.get('device')} "
              f"from {last.get('git_branch')}@{(last.get('git_sha') or '')[:8]}{dirty}")
    histories = run.histories
    print(f"  {'variant':20s} {'seed':>5s} {'best_val':>11s} {'epoch':>6s}  diverged")
    for variant in run.variants:
        for seed, hist in zip(run.seeds, histories[variant.slug]):
            if hist is None:
                print(f"  {variant.slug:20s} {seed:5d} {'-- missing --':>11s}")
                continue
            print(f"  {variant.slug:20s} {seed:5d} {hist['best_val']:11.3e} "
                  f"{str(hist['best_epoch']):>6s}  {hist['diverged_at']}")
    print(f"  rollouts cached: {run.spec.paths.rollouts.exists()}")
    return 0


def main(argv=None) -> int:
    # Line-buffer stdout so `nohup ... > log` shows progress as it happens.
    # train_one's prints have no flush=True, and a block-buffered pipe means a
    # multi-hour run looks hung. Doing it here covers those prints without
    # touching ftnode/train.py.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # not a real stream (captured, piped oddly)
        pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in _SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "train")  # `ftnode-train <yaml>` == `ftnode-train train <yaml>`

    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "show":
        return _cmd_show(args)
    raise SystemExit(
        "`ftnode-train ctrl` is not implemented yet.\n"
        "The identification stage is: `ftnode-train <experiment.yaml>`."
    )


if __name__ == "__main__":
    raise SystemExit(main())
