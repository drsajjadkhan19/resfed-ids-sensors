#!/usr/bin/env python3
"""
The local work quantum as a scheduling decision.

Motivation
----------
Energy-aware federated schedulers commonly fix the amount of local work done per
participation and optimise only which clients participate. The
per-participation cost, however, decomposes as

    e_k(w) = c_k + a_k * w          c_k = communication (independent of w)
                                    a_k = compute per unit of local work

and in an IoT setting communication dominates: under this cost model and this
federation, c_k is 85.6% of the median client's per-participation cost at w = 2.
That share is a property of the model, the hardware assignment and the local
dataset size, not a measurement of a deployment; at a 2000-row cap on the same
federation it is 78.1%. Total work deliverable for a fixed budget,
n_k(w) * w = w * floor(B_k / (c_k + a_k w)), is therefore strongly increasing in
w -- 16x from w=1 to w=32 in this federation.

The counterweight is client drift: under non-IID data, longer local optimisation
moves clients further apart before averaging, so an optimum exists and depends on
the interaction between energy heterogeneity and statistical heterogeneity.

This experiment measures where that optimum sits, across four pacing policies, and
records the terminal model alongside the tail average. The two endpoints disagree
sharply here: the trajectory can collapse at the final round while the tail
average is still high, so reporting either alone misrepresents the sweep.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import DeficitRoundRobin, LACS, ThresholdGating, FedAvgFull
from src.journal.study_policy_comparison import trajectory, class_weights

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
WORK_QUANTA = [1, 2, 4, 8, 16, 32]


def one(args):
    seed, w, regime_name, base_cfg = args
    cfgd = dict(base_cfg); cfgd["local_epochs"] = w; cfgd["seed"] = seed
    cfg = FedConfig(**cfgd)
    fed = Federation(cfg)
    cw = class_weights(fed)
    rg = REGIMES[regime_name]
    out = []
    for name, pol in [("DRR", DeficitRoundRobin("DRR", cfg.m)),
                      ("LACS-pace", LACS("LACS-pace", cfg.m, use_coverage=False)),
                      ("Threshold-0.3", ThresholdGating("Threshold-0.3", cfg.m, 0.3)),
                      ("FedAvg-full", FedAvgFull("FedAvg-full", cfg.m))]:
        r = trajectory(fed, rg, cfg, pol, seed, cw)
        f1s = [p["macro_f1"] for p in r["traj"]]
        widths = [p.get("width") for p in r["traj"] if p.get("width") is not None]
        out.append({"policy": name, "w": w, "seed": seed,
                    "final_macro_f1": f1s[-1],
                    "mean_last5": float(np.mean(f1s[-5:])),
                    "best_macro_f1": max(f1s),
                    "min_width": (min(widths) if widths else None),
                    "final_width": (widths[-1] if widths else None),
                    "energy_j": r["total_energy_j"], "updates": r["total_updates"],
                    "last_active_round": r["last_active_round"],
                    "stranded": r["stranded_fraction"]})
    return out


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    base = dict(rounds=40, m=12, max_rows_per_client=1200)
    print(f"work-quantum sweep | regime={regime} w={WORK_QUANTA} "
          f"seeds={len(seeds)}", flush=True)

    from multiprocessing import Pool
    jobs = [(s, w, regime, base) for w in WORK_QUANTA for s in seeds]
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(one, jobs)
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    pols = ["FedAvg-full", "Threshold-0.3", "LACS-pace", "DRR"]
    best = {}
    for endpoint, key in [("terminal model", "final_macro_f1"),
                          ("tail-5 average", "mean_last5")]:
        print(f"\n  endpoint: {endpoint}")
        print(f"  {'policy':<15}" + "".join(f"{'w='+str(w):>16}" for w in WORK_QUANTA))
        print("  " + "-" * (15 + 16 * len(WORK_QUANTA)))
        for p in pols:
            line = f"  {p:<15}"
            for w in WORK_QUANTA:
                v = np.array([r[key] for r in rows
                              if r["policy"] == p and r["w"] == w])
                line += f"{v.mean():>9.4f}±{v.std(ddof=1):<6.4f}"
                if key == "final_macro_f1":
                    best.setdefault(p, []).append((w, v.mean()))
            print(line)

    print(f"\n  narrowest round reached, by policy and quantum "
          f"(a width of 1 lets a single client overwrite the global model)")
    print(f"  {'policy':<15}" + "".join(f"{'w='+str(w):>8}" for w in WORK_QUANTA))
    for p in pols:
        line = f"  {p:<15}"
        for w in WORK_QUANTA:
            mw = [r["min_width"] for r in rows
                  if r["policy"] == p and r["w"] == w and r["min_width"] is not None]
            line += f"{(min(mw) if mw else -1):>8}"
        print(line)

    print(f"\n  {'policy':<15} {'best w':>8} {'best F1':>10} {'F1 at w=2':>11} {'gain':>9}")
    print("  " + "-" * 58)
    for p in pols:
        bw, bf = max(best[p], key=lambda t: t[1])
        f2 = dict(best[p])[2]
        print(f"  {p:<15} {bw:>8} {bf:>10.4f} {f2:>11.4f} {bf-f2:>+9.4f}")

    print(f"\n  {'w':>4} {'median e_k (J)':>15} {'updates':>9} {'rounds active':>14} {'stranded':>9}")
    print("  " + "-" * 56)
    for w in WORK_QUANTA:
        sub = [r for r in rows if r["w"] == w and r["policy"] == "DRR"]
        print(f"  {w:>4} {np.mean([r['energy_j'] for r in sub])/max(np.mean([r['updates'] for r in sub]),1):>15.4f} "
              f"{np.mean([r['updates'] for r in sub]):>9.0f} "
              f"{np.mean([r['last_active_round'] for r in sub]):>14.1f} "
              f"{np.mean([r['stranded'] for r in sub]):>9.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"work_quantum_{regime}.json").write_text(
        json.dumps({"regime": regime, "quanta": WORK_QUANTA, "seeds": seeds,
                    "base_config": base, "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {OUT/f'work_quantum_{regime}.json'}")


if __name__ == "__main__":
    main()
