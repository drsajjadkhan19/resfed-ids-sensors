#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the defeat of the loss-guided selectors a property of their settings?

Under an equal tuning budget, deficit round robin beat Oort and power-of-choice
by a wide margin. That result is only meaningful if the two selectors were given
a fair chance, and they were not given one in a specific respect: the shared
grid tuned the local quantum, the server rule and the proximal strength, but
each selector also has a knob of its own — Oort's exploration fraction and
power-of-choice's candidate multiplier — that was held at its default. A defeat
could therefore be an artefact of those defaults rather than of the selection
principle.

This check re-runs each seed's selected configuration across the selector's own
knob, holding everything else at what that seed chose on validation. The
decision rule was fixed before the run: **if either selector improves by more
than 0.05 under a different knob value, the main comparison is void and is
re-run with that knob inside the shared grid.**

Running this is not optional. Publishing a defeat of a published algorithm
without it would be asserting that the algorithm is worse when what was measured
is that one configuration of it is worse.
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
from src.journal.study_cape_factorial import run_arm, paired_stats
from src.journal.simulation import split_validation

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

KNOBS = {
    "Oort": ("oort", [0.1, 0.2, 0.3], 0.2),                 # exploration fraction
    "PowerOfChoice": ("powerofchoice", [2.0, 3.0, 5.0], 3.0),  # candidate multiplier
}
IMPROVEMENT_THRESHOLD = 0.05


def parse_choice(name):
    """Recover the quantum, server rule and proximal strength a seed selected."""
    _, quantum, agg, mu = name.split("|")
    if quantum.startswith("uniform"):
        q = ("uniform", int(quantum[len("uniform"):]))
    else:
        q = ("amortized", int(quantum[len("amortized"):]))
    return q, agg, float(mu[2:])


def job(args):
    seed, regime_name, cfgd, choices = args
    probe = FedConfig(**cfgd)
    probe.seed = seed
    val_dev, _ = split_validation(Federation(probe), 0.2, seed)
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    cfg.exclude_devices = tuple(val_dev)
    fed = Federation(cfg)
    rg = REGIMES[regime_name]

    y = np.concatenate([fed.y[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    cw = 1.0 / np.maximum(f, 1e-9)
    cw = cw / cw.mean()

    out = {"seed": seed}
    for fam, (policy_kind, values, default) in KNOBS.items():
        q, agg, mu0 = parse_choice(choices[fam][seed])
        for v in values:
            arm = dict(name=f"{fam}|knob{v}", quantum=q, agg=agg, mu0=mu0,
                       policy=(policy_kind, v))
            r = run_arm(fed, rg, cfg, arm, seed, cw)
            out[f"{fam}|{v}"] = r["final_macro_f1"]
            out[f"J_{fam}|{v}"] = r["energy_j"]
        out[f"config_{fam}"] = choices[fam][seed]
    return out


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    src = OUT / f"policy_family_selection_{regime}.json"
    if not src.exists():
        print(f"  the family selection result is required first: {src}")
        return
    prior = json.loads(src.read_text())
    seeds = prior["seeds"]
    cfgd = prior["config"]
    choices = {fam: {r["seed"]: r[f"chosen_{fam}"] for r in prior["rows"]}
               for fam in KNOBS}

    print(f"selector knob sensitivity | regime={regime} seeds={len(seeds)}")
    for fam, (_, values, default) in KNOBS.items():
        print(f"  {fam}: sweeping {values} (main comparison used {default})")
    print(flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        rows = pool.map(job, [(s, regime, cfgd, choices) for s in seeds])
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    drr = np.array([r["test_DRR"] for r in prior["rows"]])
    verdict_lines, tests = [], []
    for fam, (_, values, default) in KNOBS.items():
        print(f"  === {fam} ===")
        print(f"  {'knob':>8}{'terminal':>18}{'vs default':>12}{'vs DRR':>12}{'p vs DRR':>11}")
        base = np.array([r[f"{fam}|{default}"] for r in rows])
        best_gain, best_v = -np.inf, default
        for v in values:
            x = np.array([r[f"{fam}|{v}"] for r in rows])
            gain = float(x.mean() - base.mean())
            m, lo, hi, p = paired_stats(x, drr)
            tests.append({"family": fam, "knob": v, "terminal": float(x.mean()),
                          "gain_vs_default": gain, "diff_vs_drr": m, "p_vs_drr": p})
            if gain > best_gain:
                best_gain, best_v = gain, v
            flag = "  <- default" if v == default else ""
            print(f"  {v:>8}{x.mean():>10.4f}±{x.std(ddof=1):<7.4f}"
                  f"{gain:>+12.4f}{m:>+12.4f}{p:>11.5f}{flag}")
        ok = best_gain <= IMPROVEMENT_THRESHOLD
        verdict_lines.append(
            f"  {fam}: best knob {best_v} improves on the default by "
            f"{best_gain:+.4f} -> main comparison "
            f"{'STANDS' if ok else 'VOID, must be re-run with this knob in the grid'}")
        print()

    print("  Pre-declared rule: an improvement above "
          f"{IMPROVEMENT_THRESHOLD:+.2f} voids the main comparison.")
    for line in verdict_lines:
        print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"selector_knob_sensitivity_{regime}.json"
    path.write_text(json.dumps({"regime": regime, "seeds": seeds,
                                "threshold": IMPROVEMENT_THRESHOLD,
                                "tests": tests, "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
