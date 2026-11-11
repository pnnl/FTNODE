"""Collapse diagnostic for the latent potential head (``plan_collapse_diagnostic.md``).

Why did the ``grad_potential`` arm (``runs/sym_jg_full``) never form the Duffing saddle --
the spectral caps on Phi, absorption of the fit by the trainable operator, or the rollout
objective itself?  This script holds the three steps of the plan:

``regauge``
    Step B1.  Roll a frozen ``kappa_sat`` incumbent on the training split, refit a pair
    ``(A_rg, Phi_rg)`` with ``A_rg (z - grad Phi_rg) ~ F_inc`` by pointwise regression, and
    gate it (relative error, a saddle, curvature, and the untrained rollout loss).
``train``
    One run.  ``--part A``: the from-scratch protocol of a YAML spec (A-free, A-projinit).
    ``--part B``: the incumbent's encoder and decoder frozen, the operator started at
    ``A_rg`` (arms J-free, F-cold, F-warm, J-cap).  The loop mirrors
    :func:`ftnode.train.train_one` step for step and only adds logging: curvature, operator
    drift, clip activity, and a last-epoch checkpoint.
``score``
    Equilibria and the saddle verdict, curvature, bounds, discrete energy increase, and fit
    (validation, test seeds 1000-1009, settling) for incumbents, the existing capped arm, or
    runs written by ``train``.

Nothing in the package changes.  Every sampling step uses a local ``torch.Generator``, so
the logging never moves the global RNG stream that fixes the batch order.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftnode.experiments.run import build_variant, load_run  # noqa: E402
from ftnode.experiments.spec import spec_from_yaml  # noqa: E402
from ftnode.latent import ClampOperator, GradPotentialG, KappaBudget  # noqa: E402
from ftnode.latent.model import LatentFTNODE, LatentSysID  # noqa: E402
from ftnode.systems import make_dataset, sinks  # noqa: E402
from ftnode.train import rollout_y  # noqa: E402

KAPPA_SAT = ROOT / "runs" / "kappa_sat"
SYM_JG_FULL = ROOT / "runs" / "sym_jg_full"
INC_SLUG = "l-ft-k-svd-clamp"
GRAD_SLUG = "svd_clamp+grad_potential"
OUT = ROOT / "runs" / "collapse_diag"
SPEC_FREE = ROOT / "experiments" / "duffing" / "sym_jg_uncapped.yaml"
SPEC_CAP = ROOT / "experiments" / "duffing" / "sym_jg_full.yaml"

BUDGET = KappaBudget(0.1, 25.0, 0.6, 4)
L_SIGMA = 1.0998            # measured sup |silu'|; the code's table rounds it to 1.10
PROBE_SEED = 12345
N_PROBE = 4096
REGION = 0.5                # data region: within this distance of a training-rollout latent
U_EQ = (-0.2, -0.1, 0.0, 0.1, 0.2)
TEST_SEEDS = tuple(range(1000, 1010))
DECODE_TOL = 0.3

PART_B_ARMS = {
    #          cap,   A trainable, warm Phi
    "J-free": (False, True, False),
    "F-cold": (False, False, False),
    "F-warm": (False, False, True),
    "J-cap": (True, True, False),
}


# ------------------------------------------------------------------ helpers


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _phi(model):
    eq = model.dynamics.equilibrium
    return eq if isinstance(eq, GradPotentialG) else None


@torch.no_grad()
def rollout_latents(model, ds, L, h):
    _, zs = rollout_y(model, ds.W, ds.U, L, h)
    return zs


def probe_set(model, ds, L, h, n=N_PROBE):
    """``n`` latents from the model's own rollouts on ``ds``, at fixed (trajectory, step) picks."""
    zs = rollout_latents(model, ds, L, h)
    g = _gen(PROBE_SEED)
    ti = torch.randint(0, zs.shape[0], (n,), generator=g)
    si = torch.randint(0, zs.shape[1], (n,), generator=g)
    return zs[ti, si].clone(), ds.U[ti].clone()


def region_set(model, ds, L, h):
    """Every training-rollout latent of ``model``: the reference set of the data region."""
    zs = rollout_latents(model, ds, L, h)
    return zs.reshape(-1, zs.shape[-1])


def hess_phi_lmax(phi, Z, U):
    """``lambda_max(grad^2 Phi)`` at every row of ``Z`` (float64 copy; no global RNG)."""
    p = copy.deepcopy(phi).double()
    Z, U = Z.double(), U.double()

    def one(z, u):
        return p.phi(z.unsqueeze(0), u.reshape(1)).squeeze()

    H = torch.func.vmap(torch.func.hessian(one))(Z, U).detach()
    H = 0.5 * (H + H.transpose(-1, -2))
    return torch.linalg.eigvalsh(H)[:, -1]


def operator_drift(op, op_ref, Z):
    with torch.no_grad():
        A, A0 = op(Z), op_ref(Z)
        rel = (A - A0).flatten(1).norm(dim=1) / A0.flatten(1).norm(dim=1)
    return float(rel.median())


def phi_bounds(phi, Z, U):
    """A posteriori Lipschitz bound of Phi from the weight norms, and the realized max."""
    lin = phi._linears()
    with torch.no_grad():
        w1z = lin[0].weight[:, : phi.m]
        norms = [float(torch.linalg.matrix_norm(w1z, 2))] + [
            float(torch.linalg.matrix_norm(l.weight, 2)) for l in lin[1:]
        ]
    bound = L_SIGMA ** (len(lin) - 1) * float(np.prod(norms))
    with torch.no_grad():
        realized = float(phi(Z, U).norm(dim=-1).max())
    return {"weight_norms": norms, "L_phi_bound": bound, "grad_phi_max_probe": realized,
            "nagumo_radius": (BUDGET.sigma_max / BUDGET.sigma_min) * bound}


def _jacobian(F, z):
    """Per-row Jacobian of a row-wise field by ``m`` backward passes (autograd)."""
    z = z.detach().requires_grad_(True)
    f = F(z)
    rows = [torch.autograd.grad(f[:, j].sum(), z, retain_graph=j < f.shape[1] - 1)[0]
            for j in range(f.shape[1])]
    return torch.stack(rows, 1).detach(), f.detach()


def find_equilibria(model, u, starts, region_Z, iters=60, max_step=0.5, tol=1e-6):
    """Damped Newton from ``starts`` at input ``u``; dedupe, classify, decode, locate."""
    m64 = copy.deepcopy(model).double()
    phi = _phi(m64)
    uu = torch.full((starts.shape[0],), float(u), dtype=torch.float64)
    F = lambda z: m64.F(z, uu[: z.shape[0]])  # noqa: E731
    z = starts.double().clone()
    for _ in range(iters):
        J, f = _jacobian(F, z)
        step = torch.linalg.lstsq(J, f.unsqueeze(-1)).solution.squeeze(-1)
        z = (z - torch.nan_to_num(step).clamp(-max_step, max_step)).detach()
    J, f = _jacobian(F, z)
    res = f.norm(dim=-1)
    ok = torch.isfinite(res) & (res < tol)
    roots, Js = [], []
    for i in torch.nonzero(ok).flatten().tolist():
        if all(float((z[i] - r).norm()) >= 1e-3 for r in roots):
            roots.append(z[i])
            Js.append(J[i])
    out = []
    reg = region_Z.double()
    for r, Jr in zip(roots, Js):
        ev = torch.linalg.eigvals(Jr)
        n_pos = int((ev.real > 0).sum())
        kind = "sink" if n_pos == 0 else ("saddle1" if n_pos == 1 else f"other{n_pos}")
        with torch.no_grad():
            y = float(m64.decode(r.unsqueeze(0)))
        rec = {
            "z": [float(v) for v in r],
            "y": y,
            "kind": kind,
            "max_re": float(ev.real.max()),
            "eig": [[float(e.real), float(e.imag)] for e in ev],
            "dist_data": float((reg - r).norm(dim=-1).min()),
        }
        rec["in_region"] = rec["dist_data"] <= REGION
        if phi is not None:
            rec["lmax_hess_phi"] = float(
                hess_phi_lmax(phi, r.unsqueeze(0).float(), torch.tensor([float(u)]))[0])
        out.append(rec)
    return out


def saddle_verdict(roots):
    """The pre-registered criterion at u = 0 (plan section 2.5)."""
    inr = [r for r in roots if r["in_region"]]
    sk = sorted([r for r in inr if r["kind"] == "sink"], key=lambda r: r["y"])
    sd = [r for r in inr if r["kind"] == "saddle1"]
    other = [r for r in inr if r["kind"] not in ("sink", "saddle1")]
    count_ok = len(sk) == 2 and len(sd) == 1 and not other
    pos_ok = count_ok and (abs(sk[0]["y"] + 1) <= DECODE_TOL and abs(sk[1]["y"] - 1) <= DECODE_TOL
                           and abs(sd[0]["y"]) <= DECODE_TOL)
    v = {"forms": bool(pos_ok), "n_sink": len(sk), "n_saddle1": len(sd), "n_other": len(other),
         "n_off_data": len(roots) - len(inr)}
    if sd and "lmax_hess_phi" in sd[0]:
        v["saddle_lmax_hess_phi"] = sd[0]["lmax_hess_phi"]
        # Inertia: an index-1 saddle of F forces lambda_max(grad^2 Phi) > 1.
        v["inertia_consistent"] = bool(sd[0]["lmax_hess_phi"] > 1.0) if count_ok else None
    if sd:
        v["saddle_unstable_eig"] = sd[0]["max_re"]
    return v


def true_saddle_eig(u, delta=0.2):
    qs = float(np.sort(np.roots([1.0, 0.0, -1.0, -u]).real)[1])
    return -delta / 2 + np.sqrt(delta**2 / 4 + 1 - 3 * qs**2)


# ------------------------------------------------------------------ data


_CACHE: dict = {}


def datasets(spec):
    key = ("ds", spec.data_train, spec.data_val)
    if key not in _CACHE:
        _CACHE[key] = (make_dataset(spec.data_train), make_dataset(spec.data_val))
    return _CACHE[key]


def test_sets(spec):
    key = ("test", spec.data_val)
    if key not in _CACHE:
        _CACHE[key] = {ts: make_dataset(dataclasses.replace(spec.data_val, seed=ts))
                       for ts in TEST_SEEDS}
    return _CACHE[key]


def load_incumbent(s):
    key = ("inc", s)
    if key not in _CACHE:
        run = load_run(KAPPA_SAT)
        _CACHE[key] = run.models()[INC_SLUG][s]
    return _CACHE[key]


def regauge_dir(s):
    return OUT / "regauge" / f"inc{s}"


# ------------------------------------------------------------------ models


def build_part_b(s, arm, init_seed):
    """Plan section 2.4: seed, Phi, operator, load A_rg (and Phi_rg), attach frozen ends."""
    cap, a_train, warm = PART_B_ARMS[arm]
    inc = load_incumbent(s)
    rg = regauge_dir(s)
    torch.manual_seed(init_seed)
    phi = GradPotentialG(m=4, q=1, hidden=64, depth=3, R_g=2.0, activation="silu", cap=cap)
    op = ClampOperator(m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=BUDGET)
    op.load_state_dict(torch.load(rg / "A_rg.pth"))
    if warm:
        phi.load_state_dict(torch.load(rg / "phi_rg.pth"))
    model = LatentSysID(copy.deepcopy(inc.encoder), LatentFTNODE(op, phi),
                        copy.deepcopy(inc.decoder))
    frozen = [model.encoder, model.decoder] + ([] if a_train else [model.dynamics.operator])
    for mod in frozen:
        mod.requires_grad_(False)
    return model, frozen


def build_part_a(spec_path, seed, projinit=False):
    spec = spec_from_yaml(spec_path)
    variant = next(v for v in spec.variants if v.slug == GRAD_SLUG)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_variant(spec, variant)
    if projinit:  # rescaled onto the caps at construction, then trained uncapped
        # Only a capped spec rescales at build; flipping an uncapped one would rerun A-free.
        assert model.dynamics.equilibrium.cap, "projinit needs the capped spec"
        model.dynamics.equilibrium.cap = False
    return model, spec


# ------------------------------------------------------------------ training loop


def _state_digest(modules):
    h = hashlib.sha256()
    for mod in modules:
        for k, v in sorted(mod.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_loop(model, train, val, cfg, out_dir, *, probe_fn, frozen=(), op_ref=None,
               max_epochs=None, log_every=10, warmup_epochs=0):
    """``train_one`` step for step, plus logging that touches no global RNG.

    ``warmup_epochs > 0`` replaces the scheduler by a linear warm-up per optimizer step
    over that many epochs, times the same cosine factor.  With 0 the loop is bitwise
    ``train_one``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_digest0 = _state_digest(frozen)
    phi = _phi(model)
    phi_params = list(phi.parameters()) if phi is not None else []
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    projectors = [mod for mod in model.modules() if callable(getattr(mod, "project_", None))]
    hist = {"train": [], "val_extrap": [], "zmax": [], "res": [], "diverged_at": None,
            "best_val": float("inf"), "best_epoch": None, "curv": {}, "drift": {},
            "clip_frac": [], "phi_share": [], "epoch_s": [], "perm0_sha": None}
    n_run = cfg.n_epochs if max_epochs is None else min(max_epochs, cfg.n_epochs)
    diverged = False

    def log_probe(epoch):
        Z, U = probe_fn(model)
        if phi is not None:
            hist["curv"][epoch] = float(hess_phi_lmax(phi, Z, U).max())
        if op_ref is not None:
            hist["drift"][epoch] = operator_drift(model.dynamics.operator, op_ref, Z)

    for epoch in range(n_run):
        t0 = time.time()
        model.train()
        perm = torch.randperm(train.W.shape[0])
        if epoch == 0:
            hist["perm0_sha"] = hashlib.sha256(perm.numpy().tobytes()).hexdigest()[:16]
        ep_losses, ep_res, ep_zmax, n_clip, shares = [], [], 0.0, 0, []
        n_batch = (len(perm) + cfg.batch - 1) // cfg.batch
        for i in range(0, len(perm), cfg.batch):
            if warmup_epochs > 0:
                frac = min(1.0, (epoch + (i // cfg.batch + 1) / n_batch) / warmup_epochs)
                cos = 0.5 * (1 + np.cos(np.pi * epoch / cfg.n_epochs))
                for grp in opt.param_groups:
                    grp["lr"] = cfg.lr * frac * cos
            idx = perm[i: i + cfg.batch]
            w, u, y = train.W[idx], train.U[idx], train.Y[idx]
            yhat, zs = rollout_y(model, w, u, cfg.L, cfg.h)
            loss = ((yhat - y) ** 2).mean()
            res_pen = zs.new_zeros(())
            if cfg.lam_res > 0.0 and hasattr(model.dynamics, "g"):
                u_steps = u.unsqueeze(1).expand(zs.shape[0], zs.shape[1])
                res_pen = ((zs - model.dynamics.g(zs, u_steps)) ** 2).sum(-1).mean()
                loss = loss + cfg.lam_res * res_pen
            if not torch.isfinite(loss):
                diverged = True
                hist["diverged_at"] = epoch
                break
            opt.zero_grad()
            loss.backward()
            phi_sq = sum(float(p.grad.pow(2).sum()) for p in phi_params if p.grad is not None)
            total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip))
            n_clip += int(total > cfg.clip)
            shares.append(np.sqrt(phi_sq) / total if total > 0 else 0.0)
            opt.step()
            for mod in projectors:
                mod.project_()
            ep_losses.append(loss.item())
            ep_res.append(res_pen.item())
            with torch.no_grad():
                ep_zmax = max(ep_zmax, zs.norm(dim=-1).max().item())
        if diverged:
            print(f"[{out_dir.name}] diverged at epoch {epoch}", flush=True)
            break
        if warmup_epochs == 0:
            sched.step()
        model.eval()
        with torch.no_grad():
            yhat_v, _ = rollout_y(model, val.W, val.U, cfg.L_eval, cfg.h)
            val_mse = ((yhat_v - val.Y) ** 2).mean().item()
        hist["train"].append(float(np.mean(ep_losses)))
        hist["val_extrap"].append(val_mse)
        hist["zmax"].append(ep_zmax)
        hist["res"].append(float(np.mean(ep_res)) if ep_res else 0.0)
        hist["clip_frac"].append(n_clip / max(1, len(ep_losses)))
        hist["phi_share"].append(float(np.mean(shares)) if shares else 0.0)
        if np.isfinite(val_mse) and val_mse < hist["best_val"]:
            hist["best_val"] = val_mse
            hist["best_epoch"] = epoch
            torch.save(model.state_dict(), out_dir / "best.pth")
        if epoch % log_every == 0 or epoch == n_run - 1 or n_run <= 12:
            log_probe(epoch)
        hist["epoch_s"].append(time.time() - t0)
        print(f"[{out_dir.name}] ep {epoch:3d} train {hist['train'][-1]:.4e} val {val_mse:.4e} "
              f"curv {hist['curv'].get(epoch, float('nan')):.3f} "
              f"({hist['epoch_s'][-1]:.1f}s)", flush=True)
    torch.save(model.state_dict(), out_dir / "last.pth")
    hist["frozen_unchanged"] = _state_digest(frozen) == frozen_digest0
    with open(out_dir / "hist.json", "w") as fh:
        json.dump(hist, fh, indent=1)
    return hist


# ------------------------------------------------------------------ scoring


def score_model(model, spec, probe, region_Z, op_ref=None):
    train, val = datasets(spec)
    cfg = spec.train
    phi = _phi(model)
    Zp, Up = probe
    rec = {}
    if phi is not None:
        rec["curv_max_probe"] = float(hess_phi_lmax(phi, Zp, Up).max())
        rec["bounds"] = phi_bounds(phi, Zp, Up)
    if op_ref is not None:
        rec["operator_drift"] = operator_drift(model.dynamics.operator, op_ref, Zp)
    zv = rollout_latents(model, val, cfg.L_eval, cfg.h)
    g = _gen(PROBE_SEED + 1)
    flat = zv.reshape(-1, zv.shape[-1])
    starts_data = flat[torch.randint(0, flat.shape[0], (128,), generator=g)]
    starts_box = (torch.rand(128, flat.shape[-1], generator=g) * 2 - 1) * 2.5
    starts = torch.cat([starts_data, starts_box])
    rec["equilibria"] = {}
    for u in U_EQ:
        roots = find_equilibria(model, u, starts, region_Z)
        rec["equilibria"][str(u)] = roots
    rec["verdict_u0"] = saddle_verdict(rec["equilibria"]["0.0"])
    rec["true_saddle_eig_u0"] = true_saddle_eig(0.0)
    with torch.no_grad():
        yv, _ = rollout_y(model, val.W, val.U, cfg.L_eval, cfg.h)
        rec["val_mse"] = float(((yv - val.Y) ** 2).mean())
        if phi is not None:
            uu = val.U.unsqueeze(1).expand(zv.shape[0], zv.shape[1])
            Hs = 0.5 * zv.pow(2).sum(-1) - phi.phi(zv, uu)
            dH = Hs[:, 1:] - Hs[:, :-1]
            rec["max_dH_val"] = float(dH.max())
        tm, wrong, osc, blow = [], 0, 0, 0
        for ts, ds in test_sets(spec).items():
            yt, _ = rollout_y(model, ds.W, ds.U, cfg.L_eval, cfg.h)
            tm.append(float(((yt - ds.Y) ** 2).mean()))
            late = yt[:, -100:]
            blow += int((~torch.isfinite(yt).all(1) | (yt.abs().max(1).values > 10)).sum())
            osc += int((late.std(1) > 0.05).sum())
            qT = ds.Xfull[:, -1, 0]
            for k in range(yt.shape[0]):
                sk = sinks(float(ds.U[k]))
                cands = list(sk.values())
                pred = min(cands, key=lambda c: abs(float(late[k].mean()) - c[0]))
                true = min(cands, key=lambda c: abs(float(qT[k]) - c[0]))
                wrong += int(pred[0] != true[0])
        rec["test_mse_by_seed"] = dict(zip(map(str, TEST_SEEDS), tm))
        rec["test_mse_mean"] = float(np.mean(tm))
        rec["test_wrong_well"] = wrong
        rec["test_late_oscillating"] = osc
        rec["test_blowups"] = blow
        rec["n_test"] = 64 * len(TEST_SEEDS)
    return rec


def _summary(tag, r):
    v = r["verdict_u0"]
    print(f"{tag}: forms={v['forms']} sinks={v['n_sink']} saddles={v['n_saddle1']} "
          f"other={v['n_other']} off_data={v['n_off_data']} "
          f"curv={r.get('curv_max_probe', float('nan')):.3f} val={r['val_mse']:.4f} "
          f"test={r['test_mse_mean']:.4f} wrong={r['test_wrong_well']}/{r['n_test']}", flush=True)


# ------------------------------------------------------------------ commands


def cmd_regauge(a):
    torch.set_num_threads(a.threads)
    s = a.inc
    spec = spec_from_yaml(SPEC_FREE)
    train, val = datasets(spec)
    inc = load_incumbent(s)
    zs = rollout_latents(inc, train, spec.train.L, spec.train.h)
    n_traj, n_step, m = zs.shape
    perm = torch.randperm(n_traj, generator=_gen(2000 + s))
    ho, fit = perm[:64], perm[64:]

    def pairs(ix):
        Z = zs[ix].reshape(-1, m)
        U = train.U[ix].repeat_interleave(n_step)
        with torch.no_grad():
            Ft = torch.cat([inc.F(Z[i:i + 8192], U[i:i + 8192]) for i in range(0, len(Z), 8192)])
        return Z, U, Ft

    Zf, Uf, Ff = pairs(fit)
    Zh, Uh, Fh = pairs(ho)
    A = copy.deepcopy(inc.dynamics.operator).requires_grad_(True)
    torch.manual_seed(1000 + s)
    phi = GradPotentialG(m=4, q=1, hidden=64, depth=3, R_g=2.0, activation="silu", cap=False)
    field = LatentFTNODE(A, phi)
    opt = torch.optim.Adam(field.parameters(), lr=a.lr)
    gb = _gen(3000 + s)

    def heldout():
        num, den = 0.0, 0.0
        for i in range(0, len(Zh), 8192):
            with torch.no_grad():
                d = field.F(Zh[i:i + 8192], Uh[i:i + 8192]) - Fh[i:i + 8192]
            num += float(d.pow(2).sum())
            den += float(Fh[i:i + 8192].pow(2).sum())
        return num / den

    best, best_state, curve = float("inf"), None, []
    t0 = time.time()
    for step in range(1, a.max_steps + 1):
        idx = torch.randint(0, len(Zf), (a.batch,), generator=gb)
        d = field.F(Zf[idx], Uf[idx]) - Ff[idx]
        loss = d.pow(2).sum() / Ff[idx].pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            h = heldout()
            curve.append((step, h))
            if h < best:
                best, best_state = h, copy.deepcopy(field.state_dict())
            print(f"[regauge inc{s}] step {step} heldout_rel {np.sqrt(h):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            past = [c for st, c in curve if st <= step - 2000]
            if past and (past[-1] - h) < 0.01 * past[-1]:
                break
    field.load_state_dict(best_state)
    rg = regauge_dir(s)
    rg.mkdir(parents=True, exist_ok=True)
    torch.save(A.state_dict(), rg / "A_rg.pth")
    torch.save(phi.state_dict(), rg / "phi_rg.pth")
    model_rg = LatentSysID(copy.deepcopy(inc.encoder), field, copy.deepcopy(inc.decoder)).eval()
    probe = probe_set(inc, train, spec.train.L, spec.train.h)
    curv = float(hess_phi_lmax(phi, *probe).max())
    rec = score_model(model_rg, spec, probe, region_set(inc, train, spec.train.L, spec.train.h))
    inc_val = score_val(inc, spec)
    gate = {
        "heldout_rel_rmse": float(np.sqrt(best)),
        "curv_max_probe": curv,
        "forms_saddle": rec["verdict_u0"]["forms"],
        "untrained_val_mse": rec["val_mse"],
        "incumbent_val_mse": inc_val,
        "steps": curve[-1][0],
        "seconds": time.time() - t0,
    }
    gate["pass"] = bool(gate["heldout_rel_rmse"] <= 0.05 and gate["forms_saddle"]
                        and curv > 1.2 and gate["untrained_val_mse"] < 0.25)
    with open(rg / "gate.json", "w") as fh:
        json.dump({"gate": gate, "curve": curve, "score": rec}, fh, indent=1)
    print(json.dumps(gate, indent=1), flush=True)


def cmd_regate(a):
    """Recompute the B1 gate from the saved (A_rg, Phi_rg) under the current region rule."""
    torch.set_num_threads(a.threads)
    s = a.inc
    spec = spec_from_yaml(SPEC_FREE)
    train, _ = datasets(spec)
    inc = load_incumbent(s)
    rg = regauge_dir(s)
    old = json.load(open(rg / "gate.json"))
    A = ClampOperator(m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu", budget=BUDGET)
    A.load_state_dict(torch.load(rg / "A_rg.pth"))
    phi = GradPotentialG(m=4, q=1, hidden=64, depth=3, R_g=2.0, activation="silu", cap=False)
    phi.load_state_dict(torch.load(rg / "phi_rg.pth"))
    model_rg = LatentSysID(copy.deepcopy(inc.encoder), LatentFTNODE(A, phi),
                           copy.deepcopy(inc.decoder)).eval()
    probe = probe_set(inc, train, spec.train.L, spec.train.h)
    rec = score_model(model_rg, spec, probe, region_set(inc, train, spec.train.L, spec.train.h))
    gate = dict(old["gate"])
    gate["forms_saddle"] = rec["verdict_u0"]["forms"]
    gate["untrained_val_mse"] = rec["val_mse"]
    gate["region_rule"] = f"within {REGION} of all training-rollout latents"
    gate["pass"] = bool(gate["heldout_rel_rmse"] <= 0.05 and gate["forms_saddle"]
                        and gate["curv_max_probe"] > 1.2 and gate["untrained_val_mse"] < 0.25)
    with open(rg / "gate.json", "w") as fh:
        json.dump({"gate": gate, "curve": old["curve"], "score": rec, "gate_v1": old["gate"]},
                  fh, indent=1)
    print(json.dumps(gate, indent=1), flush=True)


def score_val(model, spec):
    _, val = datasets(spec)
    with torch.no_grad():
        yv, _ = rollout_y(model, val.W, val.U, spec.train.L_eval, spec.train.h)
    return float(((yv - val.Y) ** 2).mean())


def cmd_train(a):
    torch.set_num_threads(a.threads)
    if a.part == "A":
        spec_path = SPEC_CAP if (a.capped or a.projinit) else SPEC_FREE
        model, spec = build_part_a(spec_path, a.seed, a.projinit)
        arm = "A-projinit" if a.projinit else ("A-cap-replay" if a.capped else "A-free")
        out = pathlib.Path(a.out) if a.out else OUT / "partA" / arm / f"seed{a.seed}"
        train, val = datasets(spec)
        probe_fn = lambda mdl: probe_set(mdl, train, spec.train.L, spec.train.h)  # noqa: E731
        frozen, op_ref = (), None
        meta = {"part": "A", "arm": arm, "seed": a.seed, "spec": str(
            spec_path.relative_to(ROOT)), "projinit": a.projinit}
    else:
        spec = spec_from_yaml(SPEC_FREE)
        init_seed = 10 + a.init
        model, frozen = build_part_b(a.inc, a.arm, init_seed)
        out = pathlib.Path(a.out) if a.out else (
            OUT / "partB" / a.arm / f"inc{a.inc}_init{a.init}")
        train, val = datasets(spec)
        inc = load_incumbent(a.inc)
        probe = probe_set(inc, train, spec.train.L, spec.train.h)
        probe_fn = lambda mdl: probe  # noqa: E731
        op_ref = copy.deepcopy(model.dynamics.operator).requires_grad_(False)  # = A_rg
        meta = {"part": "B", "arm": a.arm, "inc": a.inc, "init": a.init,
                "init_seed": init_seed}
    out.mkdir(parents=True, exist_ok=True)
    meta.update({"threads": a.threads, "torch": torch.__version__,
                 "started": time.strftime("%F %T"), "max_epochs": a.max_epochs,
                 "warmup_epochs": a.warmup_epochs})
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    train_loop(model, train, val, spec.train, out, probe_fn=probe_fn, frozen=frozen,
               op_ref=op_ref, max_epochs=a.max_epochs, warmup_epochs=a.warmup_epochs)


def _rebuild(run_dir):
    meta = json.load(open(run_dir / "meta.json"))
    if meta["part"] == "A":
        model, spec = build_part_a(ROOT / meta["spec"], meta["seed"], meta.get("projinit", False))
        op_ref = None
    else:
        spec = spec_from_yaml(SPEC_FREE)
        model, _ = build_part_b(meta["inc"], meta["arm"], meta["init_seed"])
        op_ref = ClampOperator(m=4, sigma_min=0.1, hidden=64, depth=3, activation="silu",
                               budget=BUDGET)
        op_ref.load_state_dict(torch.load(regauge_dir(meta["inc"]) / "A_rg.pth"))
    return model, spec, meta, op_ref


def cmd_score(a):
    torch.set_num_threads(a.threads)
    results = {}
    spec = spec_from_yaml(SPEC_FREE)
    train, _ = datasets(spec)
    for s in a.incumbent or []:
        inc = load_incumbent(s)
        probe = probe_set(inc, train, spec.train.L, spec.train.h)
        r = score_model(inc, spec, probe, region_set(inc, train, spec.train.L, spec.train.h))
        results[f"incumbent{s}"] = r
        _summary(f"incumbent{s}", r)
    if a.acap:
        models = load_run(SYM_JG_FULL).models()[GRAD_SLUG]
        for s in a.acap:
            mdl = models[s]
            probe = probe_set(mdl, train, spec.train.L, spec.train.h)
            r = score_model(mdl, spec, probe, region_set(mdl, train, spec.train.L, spec.train.h))
            results[f"A-cap/seed{s}"] = r
            _summary(f"A-cap/seed{s}", r)
    for d in a.run or []:
        d = pathlib.Path(d)
        for ck in (["best", "last"] if a.ckpt == "both" else [a.ckpt]):
            if not (d / f"{ck}.pth").exists():
                continue
            model, spec_r, meta, op_ref = _rebuild(d)
            model.load_state_dict(torch.load(d / f"{ck}.pth"))
            model.eval()
            tr, _ = datasets(spec_r)
            ref = load_incumbent(meta["inc"]) if meta["part"] == "B" else model
            probe = probe_set(ref, tr, spec_r.train.L, spec_r.train.h)
            region = region_set(ref, tr, spec_r.train.L, spec_r.train.h)
            r = score_model(model, spec_r, probe, region, op_ref=op_ref)
            r["meta"] = meta
            tag = f"{meta['arm']}/{d.name}/{ck}"
            results[tag] = r
            _summary(tag, r)
    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(results, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regauge")
    r.add_argument("--inc", type=int, required=True)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--batch", type=int, default=4096)
    r.add_argument("--max-steps", type=int, default=20000)
    g = sub.add_parser("regate")
    g.add_argument("--inc", type=int, required=True)
    g.add_argument("--threads", type=int, default=1)
    t = sub.add_parser("train")
    t.add_argument("--part", choices=["A", "B"], required=True)
    t.add_argument("--seed", type=int, default=0, help="Part A model seed")
    t.add_argument("--projinit", action="store_true", help="Part A: rescale at build, train uncapped")
    t.add_argument("--capped", action="store_true",
                   help="Part A: the capped sym_jg_full arm (loop-equivalence check)")
    t.add_argument("--inc", type=int, default=0, help="Part B incumbent seed")
    t.add_argument("--init", type=int, default=0, help="Part B init index (seed 10+init)")
    t.add_argument("--arm", choices=sorted(PART_B_ARMS), default="J-free")
    t.add_argument("--max-epochs", type=int, default=None)
    t.add_argument("--warmup-epochs", type=int, default=0,
                   help="linear learning-rate warm-up over this many epochs (0 = train_one)")
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--out", default=None)
    s = sub.add_parser("score")
    s.add_argument("--incumbent", type=int, nargs="*")
    s.add_argument("--acap", type=int, nargs="*")
    s.add_argument("--run", nargs="*")
    s.add_argument("--ckpt", choices=["best", "last", "both"], default="best")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    {"regauge": cmd_regauge, "regate": cmd_regate, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
