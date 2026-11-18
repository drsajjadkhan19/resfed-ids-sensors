#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypothesis H3: withhold the global update on under-evidenced rounds.

The thin-round failure is confirmed — under a finite energy stock the terminal
model collapses where the trajectory does not. The response first proposed for
it, scaling the server step by round mass, was withdrawn: a constant step beats
it and it is almost never chosen when the choice is free. This study tests the
other response. A round below a mass floor does not move the global model at
all; its client updates are held and merged into the next qualifying round,
discounted for staleness.

Everything the rule can be accused of exploiting is controlled:

*   Clients selected in a withheld round still train and are still charged, so
    the floor buys stability with energy rather than by deleting expensive
    rounds for free.
*   Held work is flushed at the horizon, so nothing paid for is silently lost.
*   The natural alternatives are run under the same tuning budget: declining to
    run an under-filled round at all, buffered aggregation in the manner of
    FedBuff, and the same floor with no staleness discount.

Prediction, fixed before the run: against the same configuration without a
floor, the mass floor improves terminal macro-F1 by at least 0.03 under
energy-matched comparison, at more than one quantum, without reducing delivered
local work by more than 5 %.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch

torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.study_cape_factorial import run_arm, at_budget, paired_stats

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

# Quanta at which thin rounds actually occur; a floor cannot matter where they
# do not, so testing it at w=2 would only dilute the comparison.
QUANTA = [("uniform", 8), ("uniform", 16), ("uniform", 32), ("amortized", 32)]

ARMS = [
    ("no floor", "none", ("drr", None)),
    ("mass floor 0.25", "floor25", ("drr", None)),
    ("mass floor 0.50", "floor50", ("drr", None)),
    ("floor 0.50, no staleness discount", "floor50-nostale", ("drr", None)),
    ("FedBuff", "fedbuff", ("drr", None)),
    ("min participants", "none", ("minparticipants", None)),
    ("mass weighting (withdrawn)", "mass", ("drr", None)),
    ("Adam-style server step", "adam", ("drr", None)),
]

MU0 = 0.01


def label(q, arm):
    kind, bound = q
    q_lab = f"w={bound}" if kind == "uniform" else f"c/a<={bound}"
    return f"{q_lab} | {arm}"


def arm_list():
    return [dict(name=label(q, nm), quantum=q, agg=agg, mu0=MU0, policy=pol)
            for q, (nm, agg, pol) in product(QUANTA, ARMS)]


def job(args):
    seed, regime_name, cfgd = args
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    y = np.concatenate([fed.y[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    cw = 1.0 / np.maximum(f, 1e-9)
    cw = cw / cw.mean()
    rg = REGIMES[regime_name]
    return [run_arm(fed, rg, cfg, a, seed, cw) for a in arm_list()]


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    print(f"mass floor (H3) | regime={regime} seeds={len(seeds)} "
          f"arms={len(arm_list())}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, [(s, regime, cfgd) for s in seeds])
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    idx = {(r["arm"], r["seed"]): r for r in rows}

    def vec(a, k):
        return np.array([idx[(a, s)][k] for s in seeds])

    tests = []
    for q in QUANTA:
        base = label(q, "no floor")
        print(f"  === {label(q, '')[:-2].strip()} ===")
        print(f"  {'arm':<36}{'terminal':>17}{'tail-5':>17}{'J':>8}"
              f"{'work/1e3':>10}{'minw':>6}")
        for nm, _, _ in ARMS:
            a = label(q, nm)
            t = vec(a, "final_macro_f1")
            l = vec(a, "mean_last5")
            print(f"  {nm:<36}{t.mean():>9.4f}±{t.std(ddof=1):<7.4f}"
                  f"{l.mean():>9.4f}±{l.std(ddof=1):<7.4f}"
                  f"{vec(a,'energy_j').mean():>8.1f}"
                  f"{vec(a,'work').mean()/1e3:>10.0f}{vec(a,'min_width').min():>6.0f}")

        # H3 is stated against the no-floor configuration, energy-matched pairwise
        print(f"\n  {'comparison vs no floor':<36}{'diff':>10}{'95% CI':>21}"
              f"{'p':>10}{'work ratio':>12}")
        for nm, _, _ in ARMS:
            if nm == "no floor":
                continue
            a = label(q, nm)
            xs, ys = [], []
            for s in seeds:
                ra, rb = idx[(a, s)], idx[(base, s)]
                cap = min(ra["energy_j"], rb["energy_j"])
                xs.append(at_budget(ra["traj"], "energy_j", cap)[0])
                ys.append(at_budget(rb["traj"], "energy_j", cap)[0])
            m, lo, hi, p = paired_stats(np.array(xs), np.array(ys))
            wr = float(vec(a, "work").mean() / max(vec(base, "work").mean(), 1e-9))
            tests.append({"quantum": label(q, ""), "arm": nm, "diff": m,
                          "ci": [lo, hi], "p": p, "work_ratio": wr})
            print(f"  {nm:<36}{m:>+10.4f} [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"{p:>10.5f}{'*' if p < 0.05 else ' '}{wr:>12.3f}")
        print()

    # ---- the pre-declared decision on H3 ------------------------------------
    print("  H3 decision rule: >= +0.03 energy-matched, at more than one quantum,")
    print("  with delivered local work reduced by no more than 5 %.")
    for nm in ("mass floor 0.25", "mass floor 0.50"):
        ok = [t for t in tests if t["arm"] == nm
              and t["diff"] >= 0.03 and t["p"] < 0.05 and t["work_ratio"] >= 0.95]
        verdict = "MET" if len(ok) > 1 else "NOT MET"
        print(f"    {nm:<20} qualifying quanta: {len(ok)}/{len(QUANTA)}  -> {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"mass_floor_{regime}.json"
    path.write_text(json.dumps({"regime": regime, "seeds": seeds, "config": cfgd,
                                "quanta": [list(q) for q in QUANTA],
                                "arms": [a[0] for a in ARMS], "tests": tests,
                                "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
