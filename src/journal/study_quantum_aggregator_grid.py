#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Work quantum crossed with the server aggregation rule.

The factorial study establishes two things that this grid is built to settle.
A tuned uniform quantum matched or beat the per-client rule derived from the
cost ratio, and evidence weighting was the weakest of the server damping rules
tested. Both comparisons were made with one aggregator and one quantum rule
respectively, so neither is conclusive on its own: the per-client rule might
only pay off under a stronger server step, and the mass signal might only pay
off at a particular quantum.

This grid runs every combination. If the per-client rule never beats the best
uniform value at any aggregator, the per-client claim does not survive. If mass
weighting never beats an ordinary adaptive server optimiser at any quantum, the
evidence-weighting claim does not survive either. Both are stated as
falsifiable in advance and reported whichever way they fall.

The proximal term is held on throughout, since the factorial found it to carry
a significant main effect and a significant interaction with the quantum.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch

torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.study_cape_factorial import run_arm, at_budget, paired_stats

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

QUANTA = [("uniform", 2), ("uniform", 4), ("uniform", 8), ("uniform", 16),
          ("uniform", 32), ("amortized", 16), ("amortized", 32), ("amortized", 64)]
AGGREGATORS = ["none", "mass", "fixed", "decay", "adam"]


def quantum_label(spec):
    kind, bound = spec
    return f"w={bound}" if kind == "uniform" else f"c/a<={bound}"


def grid_arms():
    return [dict(name=f"{quantum_label(q)}|{agg}", quantum=q, agg=agg,
                 mu0=0.01, policy=("drr", None))
            for q in QUANTA for agg in AGGREGATORS]


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
    return [run_arm(fed, rg, cfg, a, seed, cw) for a in grid_arms()]


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    names = [a["name"] for a in grid_arms()]
    print(f"quantum x aggregator | regime={regime} seeds={len(seeds)} "
          f"arms={len(names)}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, [(s, regime, cfgd) for s in seeds])
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    idx = {(r["arm"], r["seed"]): r for r in rows}

    def vec(arm, key):
        return np.array([idx[(arm, s)][key] for s in seeds])

    for endpoint, key in [("terminal model", "final_macro_f1"),
                          ("tail-5 average", "mean_last5")]:
        print(f"  endpoint: {endpoint}")
        print(f"  {'quantum':<12}" + "".join(f"{a:>17}" for a in AGGREGATORS))
        print("  " + "-" * (12 + 17 * len(AGGREGATORS)))
        for q in QUANTA:
            line = f"  {quantum_label(q):<12}"
            for agg in AGGREGATORS:
                v = vec(f"{quantum_label(q)}|{agg}", key)
                line += f"{v.mean():>10.4f}±{v.std(ddof=1):<6.4f}"
            print(line)
        print()

    print(f"  {'arm':<20}{'J':>9}{'updates':>9}{'work/1e3':>10}{'w~':>5}")
    print("  " + "-" * 53)
    for q in QUANTA:
        n = f"{quantum_label(q)}|none"
        print(f"  {quantum_label(q):<20}{vec(n,'energy_j').mean():>9.1f}"
              f"{vec(n,'updates').mean():>9.0f}{vec(n,'work').mean()/1e3:>10.0f}"
              f"{np.median(vec(n,'w_median')):>5.0f}")

    # --- the two falsifiable comparisons ------------------------------------
    tests = []
    print("\n  Q1  Does the per-client quantum beat the best uniform quantum?")
    print(f"  {'aggregator':<12}{'best uniform':<14}{'best per-client':<16}"
          f"{'diff':>10}{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 83)
    for agg in AGGREGATORS:
        uni = [q for q in QUANTA if q[0] == "uniform"]
        amo = [q for q in QUANTA if q[0] == "amortized"]
        bu = max(uni, key=lambda q: vec(f"{quantum_label(q)}|{agg}", "final_macro_f1").mean())
        ba = max(amo, key=lambda q: vec(f"{quantum_label(q)}|{agg}", "final_macro_f1").mean())
        x = vec(f"{quantum_label(ba)}|{agg}", "final_macro_f1")
        y = vec(f"{quantum_label(bu)}|{agg}", "final_macro_f1")
        m, lo, hi, p = paired_stats(x, y)
        tests.append({"question": "per-client vs uniform", "aggregator": agg,
                      "best_uniform": quantum_label(bu), "best_amortized": quantum_label(ba),
                      "diff": m, "ci": [lo, hi], "p": p})
        print(f"  {agg:<12}{quantum_label(bu):<14}{quantum_label(ba):<16}"
              f"{m:>+10.4f} [{lo:>+7.4f},{hi:>+7.4f}]{p:>10.5f}"
              f"{'*' if p < 0.05 else ' '}")

    print("\n  Q2  Does mass-weighted aggregation beat the other server rules?")
    print(f"  {'quantum':<12}{'comparator':<12}{'diff (mass - other)':>22}"
          f"{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 79)
    for q in QUANTA:
        for agg in AGGREGATORS:
            if agg == "mass":
                continue
            x = vec(f"{quantum_label(q)}|mass", "final_macro_f1")
            y = vec(f"{quantum_label(q)}|{agg}", "final_macro_f1")
            m, lo, hi, p = paired_stats(x, y)
            tests.append({"question": "mass vs other aggregator",
                          "quantum": quantum_label(q), "comparator": agg,
                          "diff": m, "ci": [lo, hi], "p": p})
            print(f"  {quantum_label(q):<12}{agg:<12}{m:>+22.4f}"
                  f" [{lo:>+7.4f},{hi:>+7.4f}]{p:>10.5f}{'*' if p < 0.05 else ' '}")

    # --- best cell, and its margin over the published policy ----------------
    best = max(names, key=lambda n: vec(n, "final_macro_f1").mean())
    print(f"\n  best cell on the terminal model: {best} "
          f"{vec(best,'final_macro_f1').mean():.4f}±{vec(best,'final_macro_f1').std(ddof=1):.4f} "
          f"at {vec(best,'energy_j').mean():.1f} J")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"quantum_aggregator_grid_{regime}.json"
    path.write_text(json.dumps(
        {"regime": regime, "seeds": seeds, "config": cfgd,
         "quanta": [list(q) for q in QUANTA], "aggregators": AGGREGATORS,
         "tests": tests, "best_cell": best, "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
