#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the carried results survive the hardware classes assigned by default?

Twenty-four of the corpus's ninety-six devices are bare MAC addresses that match
no vendor-prefix or product-name rule, and each receives the commonest tier by
default. That is a quarter of the federation whose energy per operation, and
therefore whose cost ratio, is an assumption. The honest response is not to
defend the assignment but to change it and report what moves.

The three assignments span the plausible range. Sending the unresolved devices to
the microcontroller tier makes them the most expensive per operation; sending
them to the single-board-computer tier makes them the cheapest, which raises the
median cost ratio from about 12 to about 28 and the communication share from
85.6 % to 93.4 %. If the two results carried forward hold across that span, the
default assignment does not carry them.

Only those two results are tested here, because they are the only ones being
carried forward:

*   the terminal model diverging from the trajectory average at some quanta;
*   the local quantum and the server rule not being separable.

Mechanisms that were withdrawn are not re-examined: a robustness check on a
retracted claim establishes nothing.
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
from src.journal.study_cape_factorial import run_arm, paired_stats

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

DEFAULT_CLASSES = ["camera", "mcu", "sbc"]
QUANTA = [("uniform", 4), ("uniform", 8), ("uniform", 16), ("uniform", 32)]
AGGREGATORS = ["none", "mass", "adam"]


def label(q, agg, dc):
    return f"{dc}|w={q[1]}|{agg}"


def arm_list():
    return [dict(name=label(q, agg, dc), quantum=q, agg=agg, mu0=0.01,
                 policy=("drr", 1.0), default_class=dc)
            for dc, q, agg in product(DEFAULT_CLASSES, QUANTA, AGGREGATORS)]


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
    print(f"hardware-class robustness | regime={regime} seeds={len(seeds)} "
          f"assignments={DEFAULT_CLASSES}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, [(s, regime, cfgd) for s in seeds])
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    idx = {(r["arm"], r["seed"]): r for r in rows}

    def vec(a, k):
        return np.array([idx[(a, s)][k] for s in seeds])

    summary = {}
    for dc in DEFAULT_CLASSES:
        print(f"  === unresolved devices assigned to '{dc}' ===")
        print(f"  {'quantum':<10}" + "".join(f"{a:>22}" for a in AGGREGATORS))
        cells = []
        for q in QUANTA:
            line = f"  w={q[1]:<8}"
            for agg in AGGREGATORS:
                v = vec(label(q, agg, dc), "final_macro_f1")
                cells.append(v.mean())
                line += f"{v.mean():>14.4f}±{v.std(ddof=1):<7.4f}"
            print(line)

        # result 1: does the terminal model diverge from the trajectory average?
        worst_gap, worst_q = 0.0, None
        for q in QUANTA:
            a = label(q, "none", dc)
            gap = float(vec(a, "final_macro_f1").mean() - vec(a, "mean_last5").mean())
            if gap < worst_gap:
                worst_gap, worst_q = gap, q
        if worst_q is not None:
            a = label(worst_q, "none", dc)
            m, lo, hi, p = paired_stats(vec(a, "final_macro_f1"), vec(a, "mean_last5"))
            print(f"\n  endpoint divergence: worst at w={worst_q[1]}, "
                  f"terminal - tail = {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.2g}")
        # result 2: how wide is the surface over the same budget regime?
        rng = (min(cells), max(cells))
        print(f"  grid range over the same budget regime: "
              f"{rng[0]:.4f} to {rng[1]:.4f}  (spread {rng[1]-rng[0]:.4f})\n")
        summary[dc] = {"worst_gap": worst_gap,
                       "worst_quantum": (worst_q[1] if worst_q else None),
                       "range_low": rng[0], "range_high": rng[1],
                       "spread": rng[1] - rng[0]}

    print("  Both carried results, across the three assignments:")
    print(f"  {'assignment':<12}{'worst terminal-tail gap':>26}{'grid spread':>14}")
    for dc in DEFAULT_CLASSES:
        s_ = summary[dc]
        print(f"  {dc:<12}{s_['worst_gap']:>26.4f}{s_['spread']:>14.4f}")
    holds = (all(summary[d]["worst_gap"] < -0.05 for d in DEFAULT_CLASSES)
             and all(summary[d]["spread"] > 0.15 for d in DEFAULT_CLASSES))
    print(f"\n  Both survive every assignment: {holds}")
    if not holds:
        print("  At least one carried result depends on the default assignment, "
              "and that dependence is itself the finding.")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hardware_class_robustness_{regime}.json"
    path.write_text(json.dumps({"regime": regime, "seeds": seeds, "config": cfgd,
                                "default_classes": DEFAULT_CLASSES,
                                "summary": summary, "rows": rows},
                               indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
