"""The latent port-Hamiltonian head across plants, OOD splits, and measurement noise.

Implements ``plan_ph_head_test.md``.  Four heads share the encoder, the decoder, the data,
the loss, and the training settings:

- ``pH``: ``svd_clamp`` + ``grad_potential(cap=False)``, budget (0.1, 25, 0.6) -- the head
  the collapse diagnostic validated on Duffing (``J = K``, ``R = 0.1 I + P``,
  ``H = |z|^2/2 - Phi``);
- ``pH-floor``: the same head with budget (0.0125, 200, 0.6): the same norm bound, a lower
  dissipation floor, and a sector cap of 119.4 instead of 14.43;
- ``tanh``: the incumbent ``svd_clamp`` + ``tanh_mlp``;
- ``free``: the unstructured ``LatentNODE``, width matched to the pH parameter count.

Subcommands:

``data-check``
    Per plant: critical points and fold table, ``s_y``, every dataset generated and cached
    (with acceptance rates and hashes), and the oracle check that runs the clean truth
    through the rest-outcome classification and picks each test split's horizon.
``train``
    One (plant, training set, noise level, replicate, head).  Seeds the build with the
    replicate, reseeds right before ``train_one`` so every head sees the same batch order,
    and records the dataset hashes and a hash of the first permutation.
``score``
    One trained model: rest outcomes on every test split, MSE, encoder distances,
    equilibrium structure, reached sinks, linearization at matched sinks, and the
    discrete-time checks of the pH heads.  One JSON per model.
``summarize``
    The endpoints E1 to E5 and the outcome triggers of the plan, from the score JSONs.
``jobs``
    The argument lines of one stage, longest runs first, for the launcher.

The training loop is ``ftnode.train.train_one`` unchanged.  Outputs go to ``runs/ph_head/``.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

from ftnode.latent import (
    EncoderConfig,
    EquilibriumConfig,
    KappaBudget,
    LatentModelConfig,
    LinearSkipEncoder,
    OperatorConfig,
    QuarticConfinement,
    build_latent_ftnode,
    build_latent_node,
)
from ftnode.plant_data import PlantDataConfig, PlantDataset, add_noise, dataset_hash, make_clean, output_scale
from ftnode.plants import PLANT_IDS, PLANTS, critical_points, get_plant, separator_output
from ftnode.train import TrainConfig, rk4_step, rollout_y, train_one

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ph_head"
DATA = OUT / "data"

HEADS = {
    "pH": dict(eq="grad_potential", kwargs={"cap": False}, sigma=0.1, kappa=25.0, lam_res=1e-2),
    "pH-floor": dict(eq="grad_potential", kwargs={"cap": False}, sigma=0.0125, kappa=200.0,
                     lam_res=1e-2),
    "tanh": dict(eq="tanh_mlp", kwargs={}, sigma=0.1, kappa=25.0, lam_res=1e-2),
    "free": dict(eq=None, sigma=None, kappa=None, lam_res=0.0),
}
PH_HEADS = ("pH", "pH-floor")
FREE_WIDTH = {4: 95, 8: 105}        # matches the pH parameter count at m = 4 and m = 8
LEVELS = (0.0, 0.01, 0.03, 0.1)
STAGES = {1: (0.0, 0.03), 2: (0.01, 0.1)}
SETS = {"T0": "box", "T1": "upper"}
REPLICATES = 5
N_TRAIN, L_TRAIN, N_VAL, L_VAL, H_STEP = 512, 200, 64, 600, 0.05
TEST_BLOCKS, N_BLOCK, L_TEST_MAX = 5, 64, 4000
HORIZONS = (600, 1200, 2400, 4000)
L_MSE = 600
OSC_FRAC = 0.05                     # oscillation threshold theta = OSC_FRAC * s_y
EPS_CAP, EPS_FRAC, EPS_MONO = 0.3, 0.3, 0.3
CLASSES = ("blowup", "oscillating", "sink", "nonsink", "elsewhere")
N_REACH, L_REACH = 128, 2000
REGION = 0.5                        # in units of s_z
SECTOR_PLANTS = ("duffing_light", "coupled", "duffing")
INSIDE_PLANTS = ("asym_well", "hardening", "toggle", "duffing")
OSCILLATORY = ("duffing", "duffing_light", "asym_well", "hardening", "coupled")
# Single-thread hours per 200-epoch run (feasibility review), for queue order only.
COST = {("pH", 4): 1.36, ("pH", 8): 1.55, ("pH-floor", 4): 1.36, ("pH-floor", 8): 1.55,
        ("tanh", 4): 0.97, ("tanh", 8): 1.20, ("free", 4): 0.27, ("free", 8): 0.29}


# ------------------------------------------------------------------ cells and paths


def level_code(r: float) -> int:
    return int(round(r * 10000))


def sets_for(plant) -> tuple:
    return ("T0", "T1") if plant.sep_index is not None else ("T0",)


def split_regions(plant, set_: str) -> dict:
    """Test split name -> data region, for models trained on ``set_``."""
    if set_ == "T1":
        return {"ID": "upper", "unseen": "lower"}
    splits = {"ID": "box", "far": "far"}
    if plant.u_near:
        splits["input_near"] = "input_near"
    if plant.u_fold:
        splits["input_fold"] = "input_fold"
    return splits


def cell_dir(kind, plant, set_, r, rep):
    return OUT / kind / plant / set_ / f"r{level_code(r):04d}" / f"s{rep}"


def git_head():
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                                  cwd=ROOT).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


# ------------------------------------------------------------------ data (cached)


def _cache_path(cfg: PlantDataConfig) -> pathlib.Path:
    return DATA / cfg.plant / f"{cfg.region}-{cfg.role}-{cfg.index}-n{cfg.n_traj}-L{cfg.L}.pt"


def clean_dataset(cfg: PlantDataConfig) -> PlantDataset:
    """The noise-free dataset of ``cfg``, generated once and cached on disk.

    Concurrent writers produce identical files, and the write is atomic, so a race only
    costs time.
    """
    path = _cache_path(cfg)
    if path.exists():
        return PlantDataset(**torch.load(path))
    ds = make_clean(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save({f.name: getattr(ds, f.name) for f in dataclasses.fields(ds)}, tmp)
    os.replace(tmp, path)
    return ds


def s_y(plant: str) -> float:
    path = DATA / plant / "s_y.json"
    if path.exists():
        return json.loads(path.read_text())["s_y"]
    val = output_scale(plant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"s_y": val}))
    return val


def noisy(cfg: PlantDataConfig, r: float) -> PlantDataset:
    return add_noise(clean_dataset(cfg), cfg, r, s_y(cfg.plant))


def train_cfgs(plant: str, set_: str, rep: int):
    return (PlantDataConfig(plant, SETS[set_], "train", rep, N_TRAIN, L_TRAIN),
            PlantDataConfig(plant, SETS[set_], "val", rep, N_VAL, L_VAL))


def test_cfg(plant: str, region: str, block: int) -> PlantDataConfig:
    return PlantDataConfig(plant, region, "test", block, N_BLOCK, L_TEST_MAX)


def concat(parts) -> PlantDataset:
    fields = {f.name: torch.cat([getattr(p, f.name) for p in parts])
              for f in dataclasses.fields(PlantDataset) if f.name != "accept_rate"}
    return PlantDataset(**fields, accept_rate=float(np.mean([p.accept_rate for p in parts])))


def test_set(plant: str, region: str, r: float) -> PlantDataset:
    """The five test blocks of ``region``, with noise at level ``r`` on the windows."""
    return concat([noisy(test_cfg(plant, region, b), r) for b in range(TEST_BLOCKS)])


# ------------------------------------------------------------------ models


def model_config(plant_name: str, head: str) -> LatentModelConfig:
    plant = get_plant(plant_name)
    spec = HEADS[head]
    eq = EquilibriumConfig(kind=spec["eq"] or "tanh_mlp", hidden=64, depth=3, R_g=2.0,
                           kwargs=dict(spec["kwargs"]) if spec["eq"] else {})
    return LatentModelConfig(
        m=2 * plant.n, q=1, activation="silu",
        encoder=EncoderConfig(hidden=64, depth=2, z_scale=2.0, tau=8),
        operator=OperatorConfig(kind="svd_clamp", hidden=64, depth=3,
                                sigma_min=spec["sigma"] or 0.1),
        equilibrium=eq,
    )


def build_head(plant_name: str, head: str, seed: int, encoder: dict | None = None,
               confine: dict | None = None):
    """Seed, then build: the encoder is drawn first, so it is the same for every head.

    ``encoder = {"kind": "skip", "z_scale": s}`` then wraps the built encoder in a
    :class:`~ftnode.latent.LinearSkipEncoder` that reuses its MLP; the wrap draws no
    global random numbers, so the rest of the build is unchanged.
    ``confine = {"eps": e, "learn": bool}`` wraps a pH head's potential in a
    :class:`~ftnode.latent.QuarticConfinement`, which draws no random numbers either.
    """
    cfg = model_config(plant_name, head)
    spec = HEADS[head]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if spec["eq"] is None:
        model = build_latent_node(cfg, hidden=FREE_WIDTH[cfg.m], depth=4)
    else:
        model = build_latent_ftnode(cfg, KappaBudget(spec["sigma"], spec["kappa"], 0.6, cfg.m))
    if encoder and encoder["kind"] == "skip":
        model.encoder = LinearSkipEncoder.from_encoder(model.encoder, encoder["z_scale"], seed)
    if confine:
        if head not in PH_HEADS:
            raise ValueError("quartic confinement applies to the pH heads only")
        dyn = model.dynamics
        dyn.equilibrium = QuarticConfinement(dyn.equilibrium, confine["eps"], confine["learn"])
    return model


def batch_seed(plant: str, set_: str, r: float, rep: int) -> int:
    ent = [PLANT_IDS[plant], list(SETS).index(set_), level_code(r), rep, 77]
    return int(np.random.SeedSequence(ent).generate_state(1)[0])


def first_perm_hash(n: int) -> str:
    """Hash of the permutation ``train_one`` will draw first, without consuming it."""
    state = torch.get_rng_state()
    perm = torch.randperm(n)
    torch.set_rng_state(state)
    return hashlib.sha256(perm.numpy().tobytes()).hexdigest()[:16]


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_model(rd: pathlib.Path, meta: dict):
    model = build_head(meta["plant"], meta["head"], meta["rep"], meta.get("encoder"),
                       meta.get("confine"))
    model.load_state_dict(torch.load(rd / "best.pth"))
    return model.eval()


# ------------------------------------------------------------------ train


def cmd_train(a):
    torch.set_num_threads(a.threads)
    plant = get_plant(a.plant)
    if a.set not in sets_for(plant):
        raise SystemExit(f"{a.plant} has no training set {a.set}")
    rd = pathlib.Path(a.out) if a.out else cell_dir("train", a.plant, a.set, a.level, a.rep) / a.head
    rd.mkdir(parents=True, exist_ok=True)
    tr_cfg, va_cfg = train_cfgs(a.plant, a.set, a.rep)
    train, val = noisy(tr_cfg, a.level), noisy(va_cfg, a.level)
    encoder = {"kind": "skip", "z_scale": a.skip_scale} if a.encoder == "skip" else None
    confine = None if a.confine == "none" else {"eps": a.eps, "learn": a.confine == "learn"}
    model = build_head(a.plant, a.head, a.rep, encoder, confine)
    bseed = batch_seed(a.plant, a.set, a.level, a.rep)
    torch.manual_seed(bseed)
    cfg = TrainConfig(n_epochs=a.max_epochs or 200, lr=3e-3, batch=64, clip=1.0,
                      lam_res=HEADS[a.head]["lam_res"], L=L_TRAIN, L_eval=L_VAL, h=H_STEP)
    meta = {
        "plant": a.plant, "set": a.set, "level": a.level, "rep": a.rep, "head": a.head,
        "m": 2 * plant.n, "params_dynamics": n_params(model.dynamics),
        "params_total": n_params(model), "head_spec": HEADS[a.head],
        "encoder": encoder, "confine": confine, "model_seed": a.rep, "batch_seed": bseed, "first_perm": first_perm_hash(len(train)),
        "train_hash": dataset_hash(train), "val_hash": dataset_hash(val),
        "s_y": s_y(a.plant), "train_cfg": dataclasses.asdict(cfg), "threads": a.threads,
        "torch": torch.__version__, "git": git_head(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    label = f"{a.plant}/{a.set}/r{level_code(a.level)}/s{a.rep}/{a.head}"
    _, hist = train_one(model, train, val, cfg, ckpt_path=rd / "best.pth", label=label,
                        device=torch.device("cpu"), verbose=True)
    meta["elapsed_s"] = time.time() - t0
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    (rd / "hist.json").write_text(json.dumps(hist))


# ------------------------------------------------------------------ classification


def tolerances(crit_y: np.ndarray, sy: float) -> np.ndarray:
    """``epsilon`` per trajectory from the true critical-point outputs at its input."""
    eps = np.full(crit_y.shape[0], EPS_MONO * sy)
    for i, row in enumerate(crit_y):
        ys = np.sort(row[np.isfinite(row)])
        if len(ys) >= 2:
            eps[i] = min(EPS_CAP, EPS_FRAC * float(np.min(np.diff(ys))))
    return eps


def classify(y: np.ndarray, crit_y: np.ndarray, crit_index: np.ndarray, sy: float):
    """Rest outcome of each row of ``y`` over its last third (plan section 6.1).

    Returns ``(cls, which)``: the class index into :data:`CLASSES` and the column of the
    matched critical point (``-1`` when none).
    """
    n, T = y.shape
    seg = y[:, -(T // 3):]
    finite = np.isfinite(y).all(axis=1)
    mean = np.where(finite, np.nan_to_num(seg).mean(axis=1), np.nan)
    std = np.where(finite, np.nan_to_num(seg).std(axis=1), np.nan)
    eps = tolerances(crit_y, sy)
    cls = np.full(n, 4)
    which = np.full(n, -1)
    for i in range(n):
        if not finite[i]:
            cls[i] = 0
            continue
        if std[i] > OSC_FRAC * sy:
            cls[i] = 1
            continue
        d = np.abs(crit_y[i] - mean[i])
        d[~np.isfinite(d)] = np.inf
        k = int(np.argmin(d))
        if d[k] <= eps[i]:
            which[i] = k
            cls[i] = 2 if crit_index[i, k] == 0 else 3
    return cls, which


def class_fractions(cls) -> dict:
    return {c: float(np.mean(cls == i)) for i, c in enumerate(CLASSES)}


def agreement(cls_m, which_m, cls_t, which_t) -> np.ndarray:
    at_point = np.isin(cls_t, (2, 3))
    return (cls_m == cls_t) & (~at_point | (which_m == which_t))


# ------------------------------------------------------------------ data-check


def fold_table(plant) -> dict:
    us = np.round(np.arange(plant.u_scan[0], plant.u_scan[1] + 1e-9, 0.01), 4)
    counts = []
    for u in us:
        cps = critical_points(plant, float(u))
        counts.append((sum(p.index == 0 for p in cps), len(cps)))
    changes = [{"u_from": float(us[i]), "u_to": float(us[i + 1]),
                "sinks": [counts[i][0], counts[i + 1][0]], "points": [counts[i][1], counts[i + 1][1]]}
               for i in range(len(us) - 1) if counts[i] != counts[i + 1]]
    return {"step": 0.01, "changes": changes}


def crit_record(plant, u: float) -> list:
    out = []
    for p in critical_points(plant, u):
        rec = {"x": [float(v) for v in p.x], "y": p.y, "index": p.index,
               "eig": [[float(e.real), float(e.imag)] for e in p.eig]}
        if p.index == 0:
            ev = p.eig[np.abs(p.eig.real) > 0]
            rec["max_abs_over_re"] = float(np.max(np.abs(ev) / np.abs(ev.real)))
        out.append(rec)
    return out


def oracle(plant_name: str, region: str, sy: float) -> dict:
    """The clean truth through the classification at each candidate horizon."""
    ts = test_set(plant_name, region, 0.0)
    Y = ts.Y_clean.numpy().astype(np.float64)
    cy, ci = ts.crit_y.numpy(), ts.crit_index.numpy()
    per = {}
    for L in HORIZONS:
        cls, _ = classify(Y[:, : L + 1], cy, ci, sy)
        per[L] = {"settled": float(np.mean(np.isin(cls, (2, 3, 4)))), **class_fractions(cls)}
    chosen = next((L for L in HORIZONS if per[L]["settled"] >= 0.99), None)
    return {"L": chosen or HORIZONS[-1], "agreement_only": chosen is None, "by_horizon": per,
            "accept_rate": ts.accept_rate}


def cmd_data_check(a):
    torch.set_num_threads(a.threads)
    for name in a.plants:
        plant = get_plant(name)
        t0 = time.time()
        rec = {"plant": name, "crit_u0": crit_record(plant, 0.0), "folds": fold_table(plant),
               "s_y": s_y(name)}
        if plant.sep_index is not None:
            rec["y_sep_u0"] = separator_output(plant, 0.0)
        data = {}
        for set_ in sets_for(plant):
            for rep in range(REPLICATES):
                for cfg in train_cfgs(name, set_, rep):
                    ds = clean_dataset(cfg)
                    data[f"{set_}/{cfg.role}/{rep}"] = {"hash": dataset_hash(ds),
                                                        "accept_rate": ds.accept_rate}
        regions = sorted({reg for s in sets_for(plant) for reg in split_regions(plant, s).values()})
        rec["horizons"] = {}
        for region in regions:
            for b in range(TEST_BLOCKS):
                ds = clean_dataset(test_cfg(name, region, b))
                data[f"test/{region}/{b}"] = {"hash": dataset_hash(ds), "accept_rate": ds.accept_rate}
            rec["horizons"][region] = oracle(name, region, rec["s_y"])
        rec["datasets"] = data
        rec["elapsed_s"] = time.time() - t0
        path = OUT / "data_check" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1))
        hz = {k: (v["L"], round(v["by_horizon"][v["L"]]["settled"], 3), v["agreement_only"])
              for k, v in rec["horizons"].items()}
        print(f"[{name}] s_y={rec['s_y']:.3f} folds={[(c['u_from'], c['sinks']) for c in rec['folds']['changes']]} "
              f"horizons={hz} ({rec['elapsed_s']:.0f}s)", flush=True)


def horizons(plant_name: str) -> dict:
    return json.loads((OUT / "data_check" / f"{plant_name}.json").read_text())["horizons"]


# ------------------------------------------------------------------ score helpers


def _jacobian(F, z):
    """Per-row Jacobian of a row-wise field by ``m`` backward passes."""
    z = z.detach().requires_grad_(True)
    f = F(z)
    rows = [torch.autograd.grad(f[:, j].sum(), z, retain_graph=j < f.shape[1] - 1)[0]
            for j in range(f.shape[1])]
    return torch.stack(rows, 1).detach(), f.detach()


def newton(m64, u: float, starts, step_cap: float, iters: int = 60, tol: float = 1e-6):
    """Damped Newton in float64 from ``starts``; deduplicated converged roots with spectra."""
    uu = torch.full((starts.shape[0],), float(u), dtype=torch.float64)
    F = lambda z: m64.F(z, uu[: z.shape[0]])  # noqa: E731
    z = starts.double().clone()
    for _ in range(iters):
        J, f = _jacobian(F, z)
        step = torch.linalg.lstsq(J, f.unsqueeze(-1)).solution.squeeze(-1)
        z = (z - torch.nan_to_num(step).clamp(-step_cap, step_cap)).detach()
    J, f = _jacobian(F, z)
    res = f.norm(dim=-1)
    ok = torch.isfinite(res) & (res < tol)
    roots, Js, src = [], [], []
    for i in torch.nonzero(ok).flatten().tolist():
        hit = next((j for j, r in enumerate(roots) if float((z[i] - r).norm()) < 1e-3 * step_cap), None)
        if hit is None:
            roots.append(z[i])
            Js.append(J[i])
            src.append([i])
        else:
            src[hit].append(i)
    out = []
    for r, Jr, s in zip(roots, Js, src):
        ev = torch.linalg.eigvals(Jr)
        out.append({"z": r, "J": Jr, "index": int((ev.real > 1e-9).sum()), "eig": ev,
                    "y": float(m64.decode(r.unsqueeze(0))), "sources": s})
    return out


def min_dist(points, ref) -> np.ndarray:
    return torch.cdist(points.double(), ref.double()).min(dim=1).values.numpy()


@torch.no_grad()
def rollout_fixed_u(model, z0, u: float, L: int, h: float):
    uu = torch.full((z0.shape[0],), float(u))
    z = z0.clone()
    tail = []
    for k in range(L):
        z = rk4_step(model.F, z, uu, h)
        if k >= L - 200:
            tail.append(z)
    return z, torch.stack(tail, 1)


def reached_sinks(model, m64, z0, u: float, s_z: float, plant, sy: float) -> dict:
    """Sinks the model's own trajectories settle at, at constant input ``u``."""
    z_end, tail = rollout_fixed_u(model, z0, u, L_REACH, H_STEP)
    finite = torch.isfinite(z_end).all(dim=1)
    rec = {"u": u, "n_start": int(z0.shape[0]), "n_nonfinite": int((~finite).sum())}
    roots = newton(m64, u, z_end[finite], step_cap=0.5 * s_z, iters=30) if finite.any() else []
    cps = critical_points(plant, u)
    ty = np.array([p.y for p in cps])
    tix = np.array([p.index for p in cps])
    eps = tolerances(ty[None, :] if len(ty) else np.full((1, 1), np.nan), sy)[0]
    sinks_out = []
    zf = z_end[finite].double()
    for r in roots:
        if r["index"] != 0:
            continue
        near = (zf[r["sources"]] - r["z"]).norm(dim=-1) <= 0.25 * s_z
        if not bool(near.any()):
            continue
        d = np.abs(ty - r["y"]) if len(ty) else np.array([np.inf])
        k = int(np.argmin(d))
        kind = ("matched" if tix[k] == 0 else "nonsink") if d[k] <= eps else "spurious"
        sinks_out.append({"y": r["y"], "kind": kind, "true_col": k if d[k] <= eps else -1,
                          "n_reach": int(near.sum()), "z": [float(v) for v in r["z"]],
                          "eig": [[float(e.real), float(e.imag)] for e in r["eig"]]})
    rec["sinks"] = sinks_out
    rec["n_matched"] = sum(s["kind"] == "matched" for s in sinks_out)
    rec["n_spurious"] = sum(s["kind"] == "spurious" for s in sinks_out)
    rec["n_nonsink"] = sum(s["kind"] == "nonsink" for s in sinks_out)
    rec["n_settled"] = int(sum(s["n_reach"] for s in sinks_out))
    return rec


def mode_report(J: np.ndarray, c: np.ndarray, n: int) -> dict:
    """Eigenvalues of ``J`` ranked by output participation ``|c^T v| / |v|``; top ``n``."""
    ev, V = np.linalg.eig(J)
    part = np.abs(c @ V) / np.linalg.norm(V, axis=0)
    order = np.argsort(-part)[:n]
    top = ev[order]
    cplx = top[np.abs(top.imag) > 1e-9]
    return {"eig": [[float(e.real), float(e.imag)] for e in top],
            "participation": [float(p) for p in part[order]],
            "min_zeta": float(np.min(-cplx.real / np.abs(cplx))) if len(cplx) else None,
            "max_abs_over_re": float(np.max(np.abs(ev) / np.abs(ev.real)))}


def structure(model, m64, plant, set_: str, Ztr, s_z: float, mu, reach0: dict, sy: float,
              gen) -> dict:
    """Roots at u = 0, the structure verdict, and the linearization at matched sinks."""
    m = Ztr.shape[1]
    n_box = 128 if m <= 4 else 512
    idx = torch.randperm(Ztr.shape[0], generator=gen)[:128]
    box_a = mu + (2 * torch.rand(n_box, m, generator=gen, dtype=torch.float64) - 1) * 2.5 * s_z
    box_b = mu + (2 * torch.rand(n_box, m, generator=gen, dtype=torch.float64) - 1) * 2.5 * s_z
    roots_a = newton(m64, 0.0, torch.cat([Ztr[idx].double(), box_a]), step_cap=0.5 * s_z)
    roots_b = newton(m64, 0.0, box_b, step_cap=0.5 * s_z)
    ref = Ztr[torch.randperm(Ztr.shape[0], generator=gen)[:8192]]

    def in_region(rs):
        if not rs:
            return []
        d = min_dist(torch.stack([r["z"] for r in rs]), ref) / s_z
        return [dict(r, dist=float(di)) for r, di in zip(rs, d) if di <= REGION]

    ra, rb = in_region(roots_a), in_region(roots_b)
    found_b = sum(any(float((x["z"] - y["z"]).norm()) < 1e-3 * s_z for y in rb) for x in ra)
    cps = critical_points(plant, 0.0)
    ty = np.array([p.y for p in cps])
    eps = tolerances(ty[None, :], sy)[0]
    reached = [s for s in reach0["sinks"]]
    model_pts = [(s["y"], 0) for s in reached] + [(r["y"], r["index"]) for r in ra if r["index"] >= 1]
    true_pts = [(p.y, p.index) for p in cps]
    if set_ == "T1":
        ysep = separator_output(plant, 0.0)
        true_pts = [(y, i) for y, i in true_pts if i == 0 and y > ysep]
        model_pts = [(y, i) for y, i in model_pts if i == 0 and y > ysep]
    used, ok = set(), len(model_pts) == len(true_pts)
    for y, i in true_pts:
        cand = [j for j, (my, mi) in enumerate(model_pts)
                if mi == i and abs(my - y) <= eps and j not in used]
        if len(cand) != 1:
            ok = False
            break
        used.add(cand[0])
    c = model.decoder.c.weight.detach().double().numpy().ravel()
    lin = []
    u0 = torch.zeros(1, dtype=torch.float64)
    for s in reached:
        if s["kind"] != "matched":
            continue
        J, _ = _jacobian(lambda z: m64.F(z, u0), torch.tensor([s["z"]], dtype=torch.float64))
        tev = cps[s["true_col"]].eig
        tc = tev[np.abs(tev.imag) > 1e-9]
        lin.append({"y": s["y"], "model": mode_report(J[0].numpy(), c, plant.n),
                    "true_eig": [[float(e.real), float(e.imag)] for e in tev],
                    "true_min_zeta": float(np.min(-tc.real / np.abs(tc))) if len(tc) else None})
    return {
        "verdict": bool(ok), "n_model_points": len(model_pts), "n_true_points": len(true_pts),
        "roots_in_region": [{"y": r["y"], "index": r["index"], "dist": r["dist"]} for r in ra],
        "coverage": found_b / len(ra) if ra else None, "linearization": lin,
    }


def freq_signature(yhat, ytrue, rest_y, h, n=400) -> dict:
    """Dominant frequency of the first ``n`` steps against the initial amplitude."""
    def peak(x):
        x = x - x.mean(axis=1, keepdims=True)
        X = np.abs(np.fft.rfft(x, n=4096, axis=1))[:, 1:]
        return (np.argmax(X, axis=1) + 1) / (4096 * h)
    amp = np.abs(ytrue[:, 0] - rest_y)
    ok = np.isfinite(yhat[:, :n]).all(axis=1) & (amp > 1e-3)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    fm, ft = peak(yhat[ok, :n]), peak(ytrue[ok, :n])
    la = np.log(amp[ok])
    return {"n": int(ok.sum()), "slope_model": float(np.polyfit(la, np.log(fm), 1)[0]),
            "slope_true": float(np.polyfit(la, np.log(ft), 1)[0]),
            "median_ratio": float(np.median(fm / ft))}


def crossing_levels(plant, ts) -> np.ndarray:
    """Per test trajectory, the output level its large swings cross: the separating
    critical point's output for a multistable plant, the true rest output otherwise."""
    if plant.sep_index is None:
        return ts.rest_y.numpy().astype(np.float64)
    return np.array([separator_output(plant, float(u)) for u in ts.U])


def half_cycles(y, level, h, min_amp):
    """``(amplitude, frequency)`` of every half-cycle of ``y`` about ``level``.

    A half-cycle runs from one crossing of ``level`` to the next; its frequency is
    ``1 / (2 * duration)`` and its amplitude the largest ``|y - level|`` inside it.
    """
    x = y - level
    idx = np.nonzero(np.sign(x[1:]) != np.sign(x[:-1]))[0]
    out = []
    for i, j in zip(idx[:-1], idx[1:]):
        amp = float(np.max(np.abs(x[i:j + 1])))
        if amp >= min_amp:
            out.append((amp, 1.0 / (2.0 * (j - i) * h)))
    return out


def far_frequency(yhat, ytrue, levels, h, sy, n_bins=6) -> dict:
    """Frequency against amplitude over the large swings of far trajectories.

    For a field whose restoring force grows faster than linearly (Duffing, hardening
    spring) the frequency of the large swings rises with their amplitude; for a field
    that grows only linearly far out, it does not.  The comparison is made at matched
    amplitude, so it does not depend on how many small swings each trajectory adds:
    half-cycles are binned by amplitude (log-spaced bins over the truth's range), and per
    bin the median model frequency is divided by the median true frequency.  Slopes are
    fitted to the equal-weight bin medians.  Also reported: the largest amplitude each
    side reaches, and how many model trajectories were non-finite and dropped.
    """
    min_amp = 0.3 * sy
    pm, pt, n_bad = [], [], 0
    for k in range(len(ytrue)):
        if np.isfinite(yhat[k]).all():
            pm += half_cycles(yhat[k], levels[k], h, min_amp)
        else:
            n_bad += 1
        pt += half_cycles(ytrue[k], levels[k], h, min_amp)
    res = {"n_model": len(pm), "n_true": len(pt), "n_nonfinite_dropped": n_bad,
           "max_amp_true": max((a for a, _ in pt), default=None),
           "max_amp_model": max((a for a, _ in pm), default=None)}
    if len(pt) < 10:
        return res
    at, ft = np.array(pt).T
    edges = np.geomspace(at.min(), at.max() * (1 + 1e-9), n_bins + 1)
    am, fm = (np.array(pm).T if pm else (np.array([]), np.array([])))
    rows, mt, mm = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        st, sm = (at >= lo) & (at < hi), (am >= lo) & (am < hi)
        tmed = float(np.median(ft[st])) if st.sum() >= 3 else None
        mmed = float(np.median(fm[sm])) if sm.sum() >= 3 else None
        rows.append({"amp": [float(lo), float(hi)], "n_true": int(st.sum()), "n_model": int(sm.sum()),
                     "f_true": tmed, "f_model": mmed,
                     "ratio": (mmed / tmed) if (tmed and mmed) else None})
        centre = float(np.sqrt(lo * hi))
        if tmed:
            mt.append((centre, tmed))
        if mmed:
            mm.append((centre, mmed))
    res["bins"] = rows
    for key, pts in (("true", mt), ("model", mm)):
        if len(pts) >= 3:
            a, f = np.log(np.array(pts)).T
            res[f"slope_{key}"] = float(np.polyfit(a, f, 1)[0])
    return res


def decay_rate(y, h, start=100):
    """Decay rate of the envelope of ``y`` about its late mean (local maxima, log-linear)."""
    x = np.abs(y - y[-(len(y) // 3):].mean())[start:]
    k = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 1e-3))[0] + 1
    if len(k) < 3:
        return np.nan
    return float(-np.polyfit((k + start) * h, np.log(x[k]), 1)[0])


def ph_checks(model, zs, U, gen, h) -> dict:
    """Discrete-time checks along rollouts: energy increase, sublevel exit, h * rho(DF)."""
    phi = model.dynamics.equilibrium
    with torch.no_grad():
        uu = U[:, None].expand(zs.shape[0], zs.shape[1])
        H = 0.5 * (zs**2).sum(-1) - phi.phi(zs, uu)
    dH = (H[:, 1:] - H[:, :-1]).max().item()
    exit_ = (H - H[:, :1]).max().item()
    pick = torch.randint(0, zs.shape[0] * zs.shape[1], (256,), generator=gen)
    Z = zs.reshape(-1, zs.shape[-1])[pick]
    Uz = uu.reshape(-1)[pick]
    m64 = copy.deepcopy(model).double()
    J, _ = _jacobian(lambda z: m64.F(z, Uz.double()), Z.double())
    rho = torch.linalg.eigvals(J).abs().max(dim=-1).values
    return {"max_dH": dH, "max_H_minus_H0": exit_, "max_h_rho": float(h * rho.max()),
            "rk4_limit": 2.785}


# ------------------------------------------------------------------ score


def cmd_score(a):
    torch.set_num_threads(a.threads)
    rd = pathlib.Path(a.run) if a.run else cell_dir("train", a.plant, a.set, a.level, a.rep) / a.head
    meta = json.loads((rd / "meta.json").read_text())
    if (meta.get("encoder") or meta.get("confine")) and not a.json:
        raise SystemExit("a model with a non-default encoder needs --json: the default score "
                         "path would overwrite the stage-1 record of the same cell")
    out = pathlib.Path(a.json) if a.json else (
        cell_dir("score", meta["plant"], meta["set"], meta["level"], meta["rep"]) / f"{meta['head']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"meta": meta, "run": str(rd)}
    hist_path = rd / "hist.json"
    if not (rd / "best.pth").exists():
        res["no_model"] = True
        out.write_text(json.dumps(res, indent=1))
        return
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        res["diverged_at"] = hist.get("diverged_at")
        res["best_epoch"] = hist.get("best_epoch")
        res["best_val"] = hist.get("best_val")
    t0 = time.time()
    plant = get_plant(meta["plant"])
    sy, r, head = meta["s_y"], meta["level"], meta["head"]
    model = load_model(rd, meta)
    m64 = copy.deepcopy(model).double()
    gen = torch.Generator().manual_seed(4242)
    train = noisy(train_cfgs(meta["plant"], meta["set"], meta["rep"])[0], r)
    with torch.no_grad():
        _, Ztr = rollout_y(model, train.W, train.U, L_TRAIN, H_STEP)
    Ztr = Ztr.reshape(-1, Ztr.shape[-1])
    Ztr = Ztr[torch.isfinite(Ztr).all(dim=1)]
    mu = Ztr.mean(0).double()
    s_z = float(((Ztr - Ztr.mean(0)) ** 2).sum(-1).mean().sqrt())
    ref = Ztr[torch.randperm(Ztr.shape[0], generator=gen)[:8192]]
    res["s_z"] = s_z
    wc = model.decoder.c.weight.detach().numpy().ravel()
    # The bounded encoder confines z0 to (-2, 2)^m; the skip encoder has no such reach.
    res["encoder_reach"] = (None if meta.get("encoder") else
                            float(2 * np.abs(wc).sum() + abs(float(model.decoder.c.bias))))
    hz = horizons(meta["plant"])
    splits = {}
    z0_pool = [model.encode(train.W[:N_REACH // 2]).detach()]
    for split, region in split_regions(plant, meta["set"]).items():
        ts = test_set(meta["plant"], region, r)
        L = hz[region]["L"]
        ys, z0s, H_rec = [], [], []
        for i in range(0, len(ts), N_BLOCK):
            with torch.no_grad():
                yb, zb = rollout_y(model, ts.W[i:i + N_BLOCK], ts.U[i:i + N_BLOCK], L, H_STEP)
            ys.append(yb.numpy().astype(np.float64))
            z0s.append(zb[:, 0])
            if head in PH_HEADS and i == 0 and split in ("ID", "far"):
                H_rec.append(ph_checks(model, zb, ts.U[i:i + N_BLOCK], gen, H_STEP))
        yhat = np.concatenate(ys)
        z0 = torch.cat(z0s)
        truth = ts.Y_clean.numpy().astype(np.float64)[:, : L + 1]
        noisy_y = ts.Y.numpy().astype(np.float64)[:, : L + 1]
        cy, ci = ts.crit_y.numpy(), ts.crit_index.numpy()
        cls_m, wm = classify(yhat, cy, ci, sy)
        cls_t, wt = classify(truth, cy, ci, sy)
        agree = agreement(cls_m, wm, cls_t, wt)
        finite = np.isfinite(yhat).all(axis=1)

        def mse(a_, b_):
            return float(np.mean((a_ - b_) ** 2)) if finite.all() else float("inf")

        tmax = np.abs(truth).max(axis=1)
        rec = {
            "region": region, "L": L, "n": int(len(ts)),
            "agreement_only": hz[region]["agreement_only"],
            "class_model": class_fractions(cls_m), "class_truth": class_fractions(cls_t),
            "disagree": float(1 - agree.mean()),
            "mse600": mse(yhat[:, : L_MSE + 1], truth[:, : L_MSE + 1]),
            "mse600_noisy": mse(yhat[:, : L_MSE + 1], noisy_y[:, : L_MSE + 1]),
            "mse_full": mse(yhat, truth),
            "nonfinite": int((~finite).sum()),
            "overshoot_median": float(np.median(np.where(finite, np.nan_to_num(np.abs(yhat)).max(axis=1) / tmax, np.inf))),
            "overshoot_frac_1p5": float(np.mean(~finite | (np.nan_to_num(np.abs(yhat)).max(axis=1) > 1.5 * tmax))),
            "first_err_median": float(np.nanmedian(np.abs(yhat[:, 0] - truth[:, 0]))),
            "beyond_reach_frac": (None if res["encoder_reach"] is None else
                                  float(np.mean(np.abs(truth[:, 0]) > res["encoder_reach"]))),
            "enc_dist_median": float(np.median(min_dist(z0, ref)) / s_z),
        }
        if split == "far" and meta["plant"] in OSCILLATORY:
            rec["far_signature"] = freq_signature(yhat, truth, ts.rest_y.numpy(), H_STEP)
            rec["far_frequency"] = far_frequency(yhat, truth, crossing_levels(plant, ts), H_STEP, sy)
        if split == "ID" and meta["plant"] in ("duffing_light", "coupled"):
            dm = [decay_rate(y, H_STEP) for y in yhat if np.isfinite(y).all()]
            dt = [decay_rate(y, H_STEP) for y in truth]
            rec["decay_rate_model"] = float(np.nanmedian(dm)) if dm else None
            rec["decay_rate_true"] = float(np.nanmedian(dt))
        if H_rec:
            rec["ph_check"] = H_rec[0]
        splits[split] = rec
        z0_pool.append(z0[: N_REACH // 4])
    res["splits"] = splits
    z0_all = torch.cat(z0_pool)[: N_REACH]
    lo, hi = plant.u_train
    grid = [float(u) for u in np.linspace(lo, hi, 5)]
    bands = [0.5 * (x + y) for x, y in tuple(plant.u_near) + tuple(plant.u_fold)]
    reach = {}
    for u in sorted(set(grid + bands + [0.0])):
        reach[f"{u:+.4f}"] = reached_sinks(model, m64, z0_all, u, s_z, plant, sy)
    res["reached"] = reach
    res["reached_train_grid"] = [f"{u:+.4f}" for u in grid]
    res["structure_u0"] = structure(model, m64, plant, meta["set"], Ztr, s_z, mu,
                                    reach[f"{0.0:+.4f}"], sy, gen)
    res["score_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(f"[score] {rd}: " + " ".join(f"{k}:dis={v['disagree']:.2f},mse={v['mse600']:.3g}"
                                     for k, v in splits.items())
          + f" verdict={res['structure_u0']['verdict']} ({res['score_s']:.0f}s)", flush=True)


# ------------------------------------------------------------------ summarize


def _bootstrap(vals: dict, upper: bool, n=10000, seed=0):
    """Plant-averaged mean and its one-sided 95 % bound, resampling replicates within plants."""
    vals = {p: np.where(np.isfinite(v), v, np.inf if upper else -np.inf)
            for p, v in vals.items() if len(v)}
    if not vals:
        return None, None
    point = float(np.mean([v.mean() for v in vals.values()]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(n)
    for v in vals.values():
        draws += v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    draws /= len(vals)
    with np.errstate(invalid="ignore"):
        bound = float(np.quantile(draws, 0.95 if upper else 0.05))
    return point, bound


def _load_scores():
    recs = {}
    for p in (OUT / "score").rglob("*.json"):
        d = json.loads(p.read_text())
        m = d["meta"]
        recs[(m["plant"], m["set"], m["level"], m["rep"], m["head"])] = d
    return recs


def _id_mse(d):
    return np.inf if d.get("no_model") else d["splits"]["ID"]["mse600"]


def _bad_rest(d, split):
    if d.get("no_model"):
        return 1.0
    c = d["splits"][split]["class_model"]
    return c["elsewhere"] + c["nonsink"] + c["blowup"]


def _spurious(d):
    if d.get("no_model"):
        return np.inf
    grid = d["reached_train_grid"]
    return float(np.mean([d["reached"][u]["n_spurious"] + d["reached"][u]["n_nonsink"] for u in grid]))


def endpoints(recs, head: str, base: str = "tanh") -> dict:
    out = {"head": head, "base": base, "levels": {}}
    plants = sorted({k[0] for k in recs})
    for r in sorted({k[2] for k in recs}):
        lv = {}
        e1, e1_plant = {}, {}
        for p in plants:
            vals = []
            for s in range(REPLICATES):
                ds = [np.log(_id_mse(recs[(p, st, r, s, head)]) / _id_mse(recs[(p, st, r, s, base)]))
                      for st in sets_for(get_plant(p)) if (p, st, r, s, head) in recs and (p, st, r, s, base) in recs]
                if ds:
                    vals.append(np.mean(ds))
            e1[p] = np.array(vals)
            e1_plant[p] = _bootstrap({p: e1[p]}, upper=True)
        pt, bd = _bootstrap(e1, upper=True)
        lv["E1"] = {"mean_log_ratio": pt, "upper": bd, "pass": bd is not None and bd <= np.log(1.25),
                    "per_plant_upper": {p: v[1] for p, v in e1_plant.items()}}
        e2 = {}
        for split in ("far", "input_near", "input_fold"):
            vals = {}
            for p in plants:
                v = [recs[(p, "T0", r, s, head)]["splits"][split]["disagree"] - recs[(p, "T0", r, s, base)]["splits"][split]["disagree"]
                     for s in range(REPLICATES) if (p, "T0", r, s, head) in recs and (p, "T0", r, s, base) in recs
                     and not recs[(p, "T0", r, s, head)].get("no_model") and not recs[(p, "T0", r, s, base)].get("no_model")
                     and split in recs[(p, "T0", r, s, head)]["splits"]]
                if v:
                    vals[p] = np.array(v)
            pt, bd = _bootstrap(vals, upper=True)
            e2[split] = {"mean_diff": pt, "upper": bd, "pass": bd is not None and bd <= 0.05,
                         "per_plant_mean": {p: float(v.mean()) for p, v in vals.items()}}
        lv["E2"] = e2
        sp, ver = {}, {}
        for p in plants:
            a_, b_ = [], []
            for s in range(REPLICATES):
                keys = [(p, st, r, s) for st in sets_for(get_plant(p))
                        if (p, st, r, s, head) in recs and (p, st, r, s, base) in recs]
                if keys:
                    a_.append(np.mean([_spurious(recs[k + (head,)]) - _spurious(recs[k + (base,)]) for k in keys]))
                    b_.append(np.mean([float(not recs[k + (head,)].get("no_model") and recs[k + (head,)]["structure_u0"]["verdict"])
                                       - float(not recs[k + (base,)].get("no_model") and recs[k + (base,)]["structure_u0"]["verdict"])
                                       for k in keys]))
            sp[p], ver[p] = np.array(a_), np.array(b_)
        pts, bds = _bootstrap(sp, upper=True)
        ptv, bdv = _bootstrap(ver, upper=False)
        lv["E3"] = {"spurious_mean_diff": pts, "spurious_upper": bds, "verdict_mean_diff": ptv,
                    "verdict_lower": bdv,
                    "pass": (bds is not None and bds <= 0.5) and (bdv is not None and bdv >= -0.15)}
        vals = {}
        for p in plants:
            v = [_bad_rest(recs[(p, "T1", r, s, head)], "unseen") - _bad_rest(recs[(p, "T1", r, s, base)], "unseen")
                 for s in range(REPLICATES) if (p, "T1", r, s, head) in recs and (p, "T1", r, s, base) in recs]
            if v:
                vals[p] = np.array(v)
        pt, bd = _bootstrap(vals, upper=True)
        lv["E4"] = {"mean_diff": pt, "upper": bd, "pass": bd is not None and bd <= 0.05}
        nf = sum(sum(sp_["nonfinite"] for sp_ in d["splits"].values())
                 for k, d in recs.items() if k[4] == head and k[2] == r and not d.get("no_model"))
        lv["E5"] = {"nonfinite": int(nf), "pass": nf == 0}
        out["levels"][f"{r:g}"] = lv
    lvls = out["levels"].values()
    out["pass"] = {e: all((v[e]["pass"] if e != "E2" else all(x["pass"] for x in v["E2"].values()))
                          for v in lvls) for e in ("E1", "E2", "E3", "E4", "E5")}
    return out


def p2_check(recs, head: str) -> dict:
    """Matched-sink min damping ratio of the head against the truth on the sector plants."""
    res = {}
    for p in ("duffing_light", "coupled"):
        zm, zt = [], []
        for k, d in recs.items():
            if k[0] != p or k[4] != head or d.get("no_model"):
                continue
            for lin in d["structure_u0"]["linearization"]:
                if lin["model"]["min_zeta"] is not None:
                    zm.append(lin["model"]["min_zeta"])
                    zt.append(lin["true_min_zeta"])
        res[p] = {"model_min_zeta_median": float(np.median(zm)) if zm else None,
                  "true_min_zeta": float(np.median(zt)) if zt else None, "n": len(zm)}
    return res


def far_flat(recs, head: str) -> dict:
    sm, st = [], []
    for k, d in recs.items():
        if k[4] != head or k[1] != "T0" or d.get("no_model"):
            continue
        fs = d["splits"].get("far", {}).get("far_signature")
        if fs and "slope_model" in fs:
            sm.append(fs["slope_model"])
            st.append(fs["slope_true"])
    if not sm:
        return {"holds": None}
    m_, t_ = float(np.median(sm)), float(np.median(st))
    return {"slope_model_median": m_, "slope_true_median": t_, "holds": m_ <= 0.5 * t_}


def triggers(ep: dict, recs, head: str) -> dict:
    lv = ep["levels"]
    fail_plants = set()
    for v in lv.values():
        fail_plants |= {p for p, b in v["E1"]["per_plant_upper"].items() if b is None or b > np.log(1.25)}
        for x in v["E2"].values():
            fail_plants |= {p for p, d in x["per_plant_mean"].items() if d > 0.05}
    p2 = p2_check(recs, head)
    p2_ok = all(d["model_min_zeta_median"] is not None and d["model_min_zeta_median"] >= 0.069 > (d["true_min_zeta"] or 1)
                for d in p2.values())
    inside_fail = max((sum(1 for p in INSIDE_PLANTS if (b := v["E1"]["per_plant_upper"].get(p)) is not None and b > np.log(1.25))
                       for v in lv.values()), default=0)
    ff = far_flat(recs, head)
    far_fail = not all(v["E2"]["far"]["pass"] for v in lv.values())
    return {
        "adopt": all(ep["pass"].values()),
        "floor": (not ep["pass"]["E1"] or not ep["pass"]["E2"]) and bool(fail_plants)
                 and fail_plants <= set(SECTOR_PLANTS) and p2_ok,
        "confinement": far_fail and bool(ff.get("holds")),
        "envelope": not ep["pass"]["E4"],
        "stop": inside_fail >= 2,
        "failing_plants": sorted(fail_plants), "p2": p2, "far_signature": ff,
    }


def cmd_summarize(a):
    recs = _load_scores()
    summary = {"n_scores": len(recs), "missing": [], "by_head": {}}
    want = {(p, st, r, s, h) for p, pl in PLANTS.items() for st in sets_for(pl)
            for r in {k[2] for k in recs} for s in range(REPLICATES) for h in HEADS}
    summary["missing"] = sorted("/".join(map(str, k)) for k in want - set(recs))
    for head in PH_HEADS:
        ep = endpoints(recs, head)
        summary["by_head"][head] = {"endpoints": ep, "triggers": triggers(ep, recs, head)}
    path = OUT / "summary" / f"summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, default=float))
    for head, d in summary["by_head"].items():
        print(head, "pass:", d["endpoints"]["pass"], "triggers:",
              {k: v for k, v in d["triggers"].items() if isinstance(v, bool)})
    print(f"missing runs: {len(summary['missing'])}; written {path}")


# ------------------------------------------------------------------ jobs


def cmd_jobs(a):
    lines = []
    for name, plant in PLANTS.items():
        for set_ in sets_for(plant):
            for r in STAGES[a.stage]:
                for rep in range(REPLICATES):
                    for head in a.heads:
                        tag = f"{name}_{set_}_r{level_code(r):04d}_s{rep}_{head}"
                        args = (f"--plant {name} --set {set_} --level {r} --rep {rep} --head {head}")
                        lines.append((COST[(head, 2 * plant.n)], tag, args))
    lines.sort(key=lambda t: -t[0])
    for _, tag, args in lines:
        print(f"{tag} {args}")


# ------------------------------------------------------------------ CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data-check")
    d.add_argument("--plants", nargs="*", default=list(PLANTS))
    d.add_argument("--threads", type=int, default=1)
    for name in ("train", "score"):
        t = sub.add_parser(name)
        t.add_argument("--plant")
        t.add_argument("--set", choices=list(SETS))
        t.add_argument("--level", type=float)
        t.add_argument("--rep", type=int)
        t.add_argument("--head", choices=list(HEADS))
        t.add_argument("--threads", type=int, default=1)
        if name == "train":
            t.add_argument("--max-epochs", type=int, default=None)
            t.add_argument("--out", default=None)
            t.add_argument("--encoder", choices=["tanh", "skip"], default="tanh")
            t.add_argument("--skip-scale", type=float, default=2.0,
                           help="z_scale of the bounded term of the skip encoder")
            t.add_argument("--confine", choices=["none", "learn", "fixed"], default="none",
                           help="quartic confinement of a pH head's energy")
            t.add_argument("--eps", type=float, default=0.05,
                           help="initial (learn) or fixed quartic coefficient")
        else:
            t.add_argument("--run", default=None, help="a run directory instead of the cell args")
            t.add_argument("--json", default=None)
    sub.add_parser("summarize")
    j = sub.add_parser("jobs")
    j.add_argument("--stage", type=int, choices=sorted(STAGES), required=True)
    j.add_argument("--heads", nargs="*", default=list(HEADS))
    a = ap.parse_args(argv)
    {"data-check": cmd_data_check, "train": cmd_train, "score": cmd_score,
     "summarize": cmd_summarize, "jobs": cmd_jobs}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
