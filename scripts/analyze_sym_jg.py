"""Report the deciding invariants for the sym_jg_full run, per-arm, across seeds."""
import numpy as np
import torch

from ftnode.experiments import load_run
from ftnode.diagnostics import jg_stats, g_image

torch.manual_seed(0)
run = load_run("runs/sym_jg_full")
_, val = run.datasets()
models = run.models()
hists = run.histories
roll = run.rollouts()

# (z, u) pairs actually visited in the validation rollouts: latents z[seed] (64,601,4)
# paired with each trajectory's constant input U (64,) broadcast across time.
U_traj = val.U.reshape(64, 1).expand(64, 601).reshape(-1)  # (64*601,)

def final_val(h):
    # last recorded validation loss in the history
    for key in ("val_loss", "val", "val_mse"):
        if h and key in h:
            return float(h[key][-1])
    # fall back: scan for any per-epoch val-like list
    for k, v in (h or {}).items():
        if "val" in k and isinstance(v, (list, tuple)) and v:
            return float(v[-1])
    return float("nan")

SUB = 8000
rng = np.random.default_rng(0)

for slug in run.slugs:
    label = run.names[slug]
    per_skew, per_lam, per_gmax, per_val = [], [], [], []
    g_bound = None
    for si, seed in enumerate(run.seeds):
        m = models[slug][si]
        if m is None:
            continue
        eq = m.dynamics.equilibrium
        g_bound = getattr(eq, "g_bound", None)
        Z = torch.from_numpy(roll[slug]["z"][si].reshape(-1, 4)).float()
        U = U_traj.float()
        idx = rng.choice(Z.shape[0], size=min(SUB, Z.shape[0]), replace=False)
        Zs, Us = Z[idx], U[idx]
        skew, lam = jg_stats(m.dynamics, Zs, Us)
        with torch.no_grad():
            gnorm = g_image(m.dynamics, Zs, Us).norm(dim=-1).numpy()
        per_skew.append(skew.mean())
        per_lam.append(lam.max())          # max over samples -> the saddle
        per_gmax.append(gnorm.max())
        per_val.append(final_val(hists[slug][si]))

    print(f"\n=== {slug}   [{label}] ===")
    if not per_skew:
        print("  no models loaded")
        continue
    ps, pl, pg, pv = map(np.array, (per_skew, per_lam, per_gmax, per_val))
    print(f"  skew frac ||skew J_g||/||J_g||  : mean {ps.mean():.2e}  "
          f"max {ps.max():.2e}   (unstructured null ~0.612)")
    print(f"  lambda_max(sym J_g) at saddle   : min {pl.min():.3f}  "
          f"median {np.median(pl):.3f}  max {pl.max():.3f}   (>1 => multistable)")
    if g_bound is not None and np.isfinite(g_bound):
        print(f"  ||g||_2 max realized            : {pg.max():.3f}   "
              f"vs certified g_bound {g_bound:.3f}   "
              f"({'OK within bound' if pg.max() <= g_bound + 1e-4 else 'VIOLATION'})")
    else:
        print(f"  ||g||_2 max realized            : {pg.max():.3f}   (l_inf-box map, no l2 bound)")
    print(f"  final val loss                  : mean {np.nanmean(pv):.4e}  "
          f"min {np.nanmin(pv):.4e}  max {np.nanmax(pv):.4e}")
