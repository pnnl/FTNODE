#!/usr/bin/env python3
"""Apply the agreed budget rule to the learning-curve pilot, and print every number it used.

The pilot, per head: complete cosine schedules at 5000 steps (seeds 0 and 1) and 20000 steps
(seed 0).  The rule (agreed 2026-10-14):

1. **Checks** per run: it stopped on its budget (not deadline, not divergence); at most 1% of
   steps were skipped for non-finite gradients; its final ``val_traj_rel`` is at least 10x
   below its step-0 value.  A head whose pilot fails a check gets no next stage.
2. **Noise**: ``sigma_seed = |v5(s0) - v5(s1)| / sqrt 2``; ``SE_delta = sqrt(SE5^2 + SE20^2)``
   from the per-system standard errors of the validation metric; ``sigma = max`` of the two.
3. **Gain**: ``g = mean(v5) - v20``.  ``g > 2 sigma``: still improving, budget 80000.
   Otherwise saturated, budget 20000.
4. **Both heads get the larger budget** of their two decisions, so the comparison is fair.
5. **Seeds**: two final seeds per head run together on the head's GPU if the predicted finish
   is before ``--two-seed-by``; otherwise one.  The predicted per-process rate with ``n``
   processes on a GPU is ``min(r1, A / n)``, where ``A`` is the measured aggregate rate of
   the pilot's three processes on that GPU and ``r1`` the single-process rate (``--r1``).
   At budget 20000, the 20000-step pilot (seed 0) is reused, so only seed 1 trains.

Writes ``decision.json`` next to the pilot and prints a table.

Usage::

    python scripts/fm_budget_decision.py --pilot runs/fm_v0lite/pilot
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib

HEADS = ("structured", "free")
SMALL, LARGE, NEXT = 5000, 20000, 80000


def load(pilot, head, budget, seed):
    p = pathlib.Path(pilot) / f"{head}_b{budget}_s{seed}" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def checks(s):
    if s is None:
        return ["missing summary.json"]
    bad = []
    if s["stop_reason"] != "budget" or s["steps"] != s["budget"]:
        bad.append(f"stop_reason={s['stop_reason']} steps={s['steps']}/{s['budget']}")
    if s["skipped_frac"] > 0.01:
        bad.append(f"skipped_frac={s['skipped_frac']:.4f} > 0.01")
    v0, v1 = s["first"]["val_traj_rel"], s["final"]["val_traj_rel"]
    if not (math.isfinite(v1) and v1 <= v0 / 10.0):
        bad.append(f"val_traj_rel {v0:.4g} -> {v1:.4g}: not 10x lower")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--r1", type=float, default=5.6, help="single-process steps/s (smoke run)")
    ap.add_argument("--small", type=int, default=SMALL)
    ap.add_argument("--large", type=int, default=LARGE)
    ap.add_argument("--next", type=int, default=NEXT)
    ap.add_argument("--two-seed-by", default="2026-10-15T10:30")
    args = ap.parse_args()
    now = dt.datetime.now()
    two_seed_by = dt.datetime.fromisoformat(args.two_seed_by)
    small, large, nxt = args.small, args.large, args.next

    heads = {}
    for h in HEADS:
        runs = {k: load(args.pilot, h, b, s) for k, (b, s) in
                {"b5_s0": (small, 0), "b5_s1": (small, 1), "b20_s0": (large, 0)}.items()}
        fails = {k: checks(r) for k, r in runs.items()}
        ok = all(not f for f in fails.values())
        d = {"checks_failed": {k: f for k, f in fails.items() if f}, "pass": ok}
        if ok:
            v5 = [runs["b5_s0"]["final"]["val_traj_rel"], runs["b5_s1"]["final"]["val_traj_rel"]]
            se5 = math.sqrt(sum(runs[k]["final"]["val_traj_rel_se"] ** 2 for k in ("b5_s0", "b5_s1")) / 4)
            v20 = runs["b20_s0"]["final"]["val_traj_rel"]
            se20 = runs["b20_s0"]["final"]["val_traj_rel_se"]
            sigma_seed = abs(v5[0] - v5[1]) / math.sqrt(2)
            se_delta = math.sqrt(se5**2 + se20**2)
            sigma = max(sigma_seed, se_delta)
            g = sum(v5) / 2 - v20
            agg = sum(r["steps_per_s"] for r in runs.values())
            d.update({"v5": v5, "v5_mean": sum(v5) / 2, "se5_mean": se5, "v20": v20, "se20": se20,
                      "sigma_seed": sigma_seed, "se_delta": se_delta, "sigma": sigma, "gain": g,
                      "gain_over_sigma": g / sigma if sigma > 0 else float("inf"),
                      "improving": g > 2 * sigma, "budget": nxt if g > 2 * sigma else large,
                      "pilot_aggregate_steps_per_s": agg})
        heads[h] = d

    passing = [h for h in HEADS if heads[h]["pass"]]
    plan = {"made_at": now.isoformat(timespec="seconds"), "heads": heads, "launch": []}
    if passing:
        budget = max(heads[h]["budget"] for h in passing)
        plan["budget"] = budget
        for h in passing:
            A = heads[h]["pilot_aggregate_steps_per_s"]
            if budget == large:
                seeds, n_proc = [1], 1                      # seed 0 exists: the 20000-step pilot
            else:
                r2 = min(args.r1, A / 2)
                end2 = now + dt.timedelta(seconds=budget / r2)
                seeds, n_proc = ([0, 1], 2) if end2 <= two_seed_by else ([0], 1)
            rate = min(args.r1, A / n_proc)
            end = now + dt.timedelta(seconds=budget / rate)
            plan["launch"].append({"head": h, "budget": budget, "seeds": seeds, "procs_per_gpu": n_proc,
                                   "pred_steps_per_s": rate, "pred_end": end.isoformat(timespec="minutes")})

    out = pathlib.Path(args.pilot) / "decision.json"
    out.write_text(json.dumps(plan, indent=2))
    for h in HEADS:
        d = heads[h]
        if not d["pass"]:
            print(f"{h:10s} FAILED checks: {d['checks_failed']}")
            continue
        print(f"{h:10s} v5 = {d['v5'][0]:.5f}, {d['v5'][1]:.5f} (mean {d['v5_mean']:.5f})   v20 = {d['v20']:.5f}   "
              f"sigma_seed {d['sigma_seed']:.5f}  SE_delta {d['se_delta']:.5f}  -> sigma {d['sigma']:.5f}   "
              f"gain {d['gain']:.5f} = {d['gain_over_sigma']:.2f} sigma -> {'IMPROVING' if d['improving'] else 'saturated'}"
              f" (budget {d['budget']})")
    for L in plan["launch"]:
        print(f"launch {L['head']:10s} budget {L['budget']} seeds {L['seeds']} "
              f"({L['procs_per_gpu']} per GPU, ~{L['pred_steps_per_s']:.2f} steps/s, predicted end {L['pred_end']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
