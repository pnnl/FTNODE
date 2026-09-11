"""Fit a warm potential Phi per seed, for the grad_potential warm-start arm.

The grad_potential arm learns a flat Phi from a cold start, so no saddle forms.  This
script fits a scalar Phi against each frozen incumbent field with the V-only feasibility
fit, projects it onto the arm's spectral caps, and saves the equilibrium state dict.  The
runner then loads it with ``ftnode-train --warm-init``, so the arm starts from a Phi that
already has curvature.

Data hygiene: the fit uses TRAIN latents only.  The warm Phi is the initialization of a
model that is later scored on ``data_val`` (both the reported RMSE and ``best_val`` model
selection read ``data_val``), so any state region taken from ``data_val`` and baked into the
init would leak the evaluation set into the model.  The latents therefore come from rolling
the frozen incumbent over ``data_train``, the same region the arm trains over.  This differs
from the feasibility diagnostic, which reports on ``data_val`` on purpose and trains nothing.

Pre-train gate: after projection, this prints ``max_z lambda_max(sym J_g)`` over the fit
latents and warns when it is below 1.  A saddle of V needs ``lambda_max(J_g) > 1``, so a
warm map below that has been flattened by the cap projection and gives no warm start.  It
does NOT use ``GradPotentialG.curvature_headroom``, which re-seeds the readout and reports
what the cap SET admits rather than what THIS map holds.

    uv run python scripts/make_warm_phi.py
    uv run python scripts/make_warm_phi.py --run runs/sym_jg_full --out runs/sym_jg_full_warmstart/warm_init
"""
from __future__ import annotations

import argparse
import pathlib

import torch

from ftnode.experiments.registry import resolve_variants
from ftnode.experiments.run import build_variant, load_run
from ftnode.latent import fit_potential
from ftnode.train import rollout_y

#: The incumbent whose frozen field each warm Phi is fit against.
INCUMBENT = "l-ft-k-svd-clamp"
#: The two-axis variant the warm map is built for.  Its slug names the warm-file directory.
GRAD_VARIANT = {"operator": "svd_clamp", "equilibrium": "grad_potential"}


def max_sym_curvature(g, Z, U, cap=4096):
    """``max lambda_max(sym J_g)`` of ``g`` over ``Z`` -- the pitchfork gate.

    ``J_g = grad^2 Phi`` is already symmetric, so this is the largest Hessian eigenvalue of
    the potential over the sample.  ``eigvalsh`` is used for its value only, not through a
    gradient, so the repeated eigenvalues the construction produces are harmless here.
    Samples at most ``cap`` latents, because the scan is a diagnostic and the field is
    smooth.
    """
    if Z.shape[0] > cap:
        idx = torch.randperm(Z.shape[0])[:cap]
        Z, U = Z[idx], U[idx]

    def phi_one(z, u):
        return g.phi(z.unsqueeze(0), u.unsqueeze(0)).squeeze()

    hess = torch.func.vmap(torch.func.hessian(phi_one))(Z, U)  # (n, m, m)
    sym = 0.5 * (hess + hess.transpose(-1, -2))
    return float(torch.linalg.eigvalsh(sym)[:, -1].max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=pathlib.Path, default=pathlib.Path("runs/sym_jg_full"),
                    help="run directory holding the frozen incumbent checkpoints")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("runs/sym_jg_full_warmstart/warm_init"),
                    help="directory to write the per-seed warm equilibrium state dicts")
    ap.add_argument("--steps", type=int, default=3000, help="feasibility-fit steps per seed")
    args = ap.parse_args()

    device = torch.device("cpu")
    run = load_run(args.run)
    b = run.spec.budget
    L, h = run.spec.train.L, run.spec.train.h
    m = run.spec.model.m

    gvar = resolve_variants([GRAD_VARIANT])[0]
    models = run.models(device)[INCUMBENT]
    train, _ = run.datasets(device)          # val is deliberately not touched

    out_dir = args.out / gvar.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    for s, model in zip(run.seeds, models):
        if model is None:
            print(f"[seed {s}] incumbent checkpoint missing, skipping")
            continue

        # Visited TRAIN latents: roll the frozen incumbent over the train windows at the
        # training horizon L (NOT the encoder window width), then flatten.  Align the
        # per-trajectory constant input to every latent row, and mask NaN on both together.
        with torch.no_grad():
            _, Zs = rollout_y(model, train.W, train.U, L, h)
        Z = Zs.reshape(-1, m)
        U = train.U.view(-1, 1).expand(-1, L + 1).reshape(-1)
        keep = torch.isfinite(Z).all(dim=-1)
        Z, U = Z[keep], U[keep]

        # V-only feasibility fit, in the ARM's activation so the transfer is a genuine warm
        # start (a tanh-fit Phi loaded into a silu net is a different function).
        res = fit_potential(
            model.dynamics, Z, U, b.sigma_min, b.sigma_max,
            activation=run.spec.model.activation, steps=args.steps, verbose=False,
        )

        # Build the capped target map exactly as the runner will, from the run config, so the
        # warm file's projection target cannot desync from the trained arm.
        g = build_variant(run.spec, gvar).dynamics.equilibrium
        g.load_state_dict(res.potential.state_dict())
        g.project_()

        lam = max_sym_curvature(g, Z, U)
        warn = "  WARN: flat -- no saddle survives the cap projection" if lam < 1.0 else ""
        print(f"[seed {s}] {res.report().splitlines()[-1]}")
        print(f"[seed {s}] max lambda_max(sym J_g) after projection = {lam:.3f}{warn}")

        path = out_dir / f"seed{s}.pth"
        torch.save(g.state_dict(), path)
        print(f"[seed {s}] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
