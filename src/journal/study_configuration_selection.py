#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair model selection between competing families on a device-disjoint split.

The identification grid shows that the local quantum and the server rule
interact strongly under a finite stock: over the same budget regime the terminal
score ranges from 0.05 to 0.42 on their pairing alone. That grid cannot settle
which *family* of quantum rule is better, for two reasons. Its comparisons pick
the best level of each family by looking at the test devices and then compute a
paired test on the same data, which is post-selection and cannot be confirmatory.
And it gives no family an explicit tuning budget, so any family with more knobs
is flattered.

This study fixes both. Each competing family is given the **same** number of
trials on the **same** validation devices, chooses one configuration, and is then
evaluated **once** on the test devices with that choice frozen:

    all devices -> TEST devices        never seen during selection or training
                -> VALIDATION devices  held out of training, used only to choose
                -> TRAINING clients     the federation

The families are the **participation policies**, since the open question after
the quantum study is whether the choice of who participates still matters once
the local work and the server rule are tuned. Each family therefore receives the
same grid over local quantum (uniform 4, 16, 32, and the per-client cost-ratio
rule), server rule (none, mass, decay, Adam-style) and proximal strength
(0.01, 0.1) — thirty-two trials each:

    DRR                 capacity-proportional service; quantum scale
    Threshold           the published gating policy; charge threshold
    Oort                loss-guided selection with staleness, exploration and a
                        cost-aware utility term; exploration fraction
    Power-of-choice     size-weighted candidate pool probed on the current global
                        model, with the probe charged; candidate multiplier

An earlier version of this study held each selector's own parameter at a default
while tuning everything around it, and omitted Oort's cost-aware utility term
and power-of-choice's probe entirely. Both omissions favour the policies that
need least information, which is the direction that would manufacture the result
being tested. They are corrected here.

Comparing a tuned DRR against an *untuned* selection algorithm would repeat the
error this study exists to avoid, so every family is tuned identically. The
published and standard configurations are also reported untuned, as references.

The validation devices are removed from the client pool before the feature
transforms are fitted, so they influence neither the training data nor the
training representation. Reported energies are per-seed means over runs whose
budgets vary; comparisons are made within the same budget regime, not at
literally identical joules.
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
from src.journal.simulation import split_validation, split_composition

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

AGGREGATORS = ["none", "mass", "adam"]
MU0 = [0.01, 0.1]
QUANTA = [("uniform", 4), ("uniform", 16), ("uniform", 32), ("amortized", 32)]

# Each family also tunes one parameter of its own, so that no family is compared
# at a fixed setting while its rivals are optimised. The counts are equal by
# construction: 4 quanta x 3 server rules x 2 proximal strengths x 3 own-parameter
# values = 72 trials each.
FAMILIES = {
    "DRR": dict(kind="drr", knob=[0.5, 1.0, 2.0]),                 # quantum scale
    "Threshold": dict(kind="threshold", knob=[0.2, 0.3, 0.5]),     # charge threshold
    "Oort": dict(kind="oort", knob=[0.1, 0.2, 0.3]),               # exploration
    "PowerOfChoice": dict(kind="powerofchoice", knob=[2.0, 3.0, 5.0]),  # candidates
}

# The probe-cost model is an assumption, so it is reported under a low, central
# and high accounting. These are three separate scenarios, NOT three trials to
# choose between: selecting the probing policy's best accounting would hand it
# three times the tuning freedom every other family gets, which is the error
# this study exists to avoid. PROBE_PRIMARY is the declared headline; the others
# appear only as sensitivity, and the policy is never selected across them.
PROBE_PRIMARY = 1.0
PROBE_SCALES = [0.5, 1.0, 2.0]

# Untuned references, reported alongside the selected configurations.
REFERENCES = [
    dict(name="ref Threshold-0.3 (published)", quantum=("uniform", 2), agg="none",
         mu0=0.0, policy=("threshold", 0.3)),
    dict(name="ref FedAvg-full", quantum=("uniform", 2), agg="none", mu0=0.0,
         policy=("fedavg", None)),
    dict(name="ref DRR-base", quantum=("uniform", 2), agg="none", mu0=0.0,
         policy=("drr", None)),
    dict(name="ref CAPE-as-submitted", quantum=("amortized", 16), agg="mass",
         mu0=0.01, policy=("drr", None)),
]


def family_grid(fam):
    """The trials for one family. Every family has the same number."""
    spec = FAMILIES[fam]
    return [dict(name=f"{fam}|{q[0]}{q[1]}|{agg}|mu{mu}|k{kn}", quantum=q,
                 agg=agg, mu0=mu, policy=(spec["kind"], kn), probe_scale=1.0)
            for q, agg, mu, kn in product(QUANTA, AGGREGATORS, MU0, spec["knob"])]


def job(args):
    seed, regime_name, cfgd = args
    probe = FedConfig(**cfgd)
    probe.seed = seed
    val_dev, _ = split_validation(Federation(probe), 0.2, seed)

    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    cfg.exclude_devices = tuple(val_dev)
    fed = Federation(cfg)
    rg = REGIMES[regime_name]
    val_dev = list(fed.excluded_devices)
    val_idx = np.concatenate([fed.client_idx[d] for d in val_dev])

    y = np.concatenate([fed.y[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    cw = 1.0 / np.maximum(f, 1e-9)
    cw = cw / cw.mean()

    out = {"seed": seed,
           "composition": {"train": split_composition(fed, list(fed.client_devices)),
                           "validation": split_composition(fed, val_dev),
                           "test_devices": len(fed.eval_devices),
                           "holdout_label_shift": fed.holdout_label_shift}}

    for fam in FAMILIES:
        grid = family_grid(fam)
        scores = {}
        for arm in grid:
            r = run_arm(fed, rg, cfg, arm, seed, cw, eval_idx=val_idx)
            scores[arm["name"]] = r["final_macro_f1"]
        best = max(scores, key=scores.get)
        chosen = next(a for a in grid if a["name"] == best)
        rt = run_arm(fed, rg, cfg, chosen, seed, cw)      # one test evaluation
        out[f"chosen_{fam}"] = best
        out[f"val_{fam}"] = scores[best]
        out[f"grid_{fam}"] = scores
        out[f"trials_{fam}"] = len(grid)
        out[f"test_{fam}"] = rt["final_macro_f1"]
        out[f"tail_{fam}"] = rt["mean_last5"]
        out[f"J_{fam}"] = rt["energy_j"]
        out[f"probeJ_{fam}"] = rt.get("probe_energy_j", 0.0)
        out[f"updates_{fam}"] = rt["updates"]
        out[f"traj_{fam}"] = [dict(round=p["round"], macro_f1=p["macro_f1"],
                                   energy_j=p["energy_j"], width=p["width"])
                              for p in rt["traj"]]
        out[f"per_class_f1_{fam}"] = rt["per_class_f1"]

    # the selected probing configuration, re-costed under the other accountings
    fam = "PowerOfChoice"
    chosen = next(a for a in family_grid(fam) if a["name"] == out[f"chosen_{fam}"])
    for ps in PROBE_SCALES:
        if ps == 1.0:
            out[f"test_{fam}_probe{ps}"] = out[f"test_{fam}"]
            out[f"J_{fam}_probe{ps}"] = out[f"J_{fam}"]
            out[f"probeJ_{fam}_probe{ps}"] = out.get(f"probeJ_{fam}", 0.0)
            out[f"updates_{fam}_probe{ps}"] = out.get(f"updates_{fam}", 0)
            continue
        alt = dict(chosen, probe_scale=ps)
        ra = run_arm(fed, rg, cfg, alt, seed, cw)
        out[f"test_{fam}_probe{ps}"] = ra["final_macro_f1"]
        out[f"J_{fam}_probe{ps}"] = ra["energy_j"]
        out[f"probeJ_{fam}_probe{ps}"] = ra["probe_energy_j"]
        out[f"updates_{fam}_probe{ps}"] = ra["updates"]

    for ref in REFERENCES:
        rr = run_arm(fed, rg, cfg, ref, seed, cw)
        out[f"test_{ref['name']}"] = rr["final_macro_f1"]
        out[f"tail_{ref['name']}"] = rr["mean_last5"]
        out[f"J_{ref['name']}"] = rr["energy_j"]
        out[f"traj_{ref['name']}"] = [dict(round=p["round"], macro_f1=p["macro_f1"],
                                           energy_j=p["energy_j"], width=p["width"])
                                      for p in rr["traj"]]
    return out


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    n_trials = {f: len(family_grid(f)) for f in FAMILIES}
    print(f"fair family selection | regime={regime} seeds={len(seeds)}")
    print(f"  trials per family (identical by construction): {n_trials}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        rows = pool.map(job, [(s, regime, cfgd) for s in seeds])
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    from collections import Counter
    for fam in FAMILIES:
        print(f"  {fam}: configurations chosen on validation")
        for name, n in Counter(r[f"chosen_{fam}"] for r in rows).most_common(5):
            print(f"    {name:<48} {n}/{len(seeds)}")

    comp = rows[0]["composition"]
    print(f"\n  split composition (seed {rows[0]['seed']}): "
          f"train {comp['train']['n_devices']} dev, "
          f"validation {comp['validation']['n_devices']} dev, "
          f"test {comp['test_devices']} dev")
    l1 = float(np.abs(np.array(comp["train"]["class_mix"])
                      - np.array(comp["validation"]["class_mix"])).sum())
    print(f"    train-vs-validation class-mixture L1 distance: {l1:.4f}")

    idx_probe = {}
    for r in rows:
        for fam in FAMILIES:
            for nm, v in r.get(f"grid_{fam}", {}).items():
                idx_probe[(nm, r["seed"])] = v

    names = list(FAMILIES) + [r["name"] for r in REFERENCES]
    print(f"\n  {'arm':<30}{'terminal':>17}{'tail-5':>17}{'J (mean [min,max])':>24}")
    print("  " + "-" * 88)
    vals = {}
    for n in names:
        t = np.array([r[f"test_{n}"] for r in rows])
        l = np.array([r[f"tail_{n}"] for r in rows])
        j = np.array([r[f"J_{n}"] for r in rows])
        vals[n] = t
        print(f"  {n:<30}{t.mean():>9.4f}±{t.std(ddof=1):<7.4f}"
              f"{l.mean():>9.4f}±{l.std(ddof=1):<7.4f}"
              f"{j.mean():>12.1f} [{j.min():.1f},{j.max():.1f}]")

    print(f"\n  Power-of-choice under alternative probe accountings. These are "
          f"SENSITIVITY scenarios, not\n  candidate configurations: the "
          f"confirmatory comparison uses the declared primary "
          f"(x{PROBE_PRIMARY}) only.")
    print(f"  {'probe cost multiplier':<24}{'terminal':>17}{'J':>10}")
    for ps in PROBE_SCALES:
        t = np.array([r[f"test_PowerOfChoice_probe{ps}"] for r in rows])
        j = np.mean([r[f"J_PowerOfChoice_probe{ps}"] for r in rows])
        print(f"  {('x' + str(ps)):<24}{t.mean():>9.4f}±{t.std(ddof=1):<7.4f}{j:>10.1f}")

    print(f"\n  Confirmatory comparison — selection frozen on validation, "
          f"one test evaluation each (n={len(seeds)})")
    print(f"  {'comparison':<52}{'diff':>10}{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 93)
    tests = []
    pairs = [("DRR", "Oort"), ("DRR", "PowerOfChoice"), ("DRR", "Threshold"),
             ("Oort", "Threshold"), ("PowerOfChoice", "Threshold"),
             ("DRR", "ref DRR-base"), ("DRR", "ref CAPE-as-submitted"),
             ("DRR", "ref Threshold-0.3 (published)")]
    for a, b in pairs:
        m, lo, hi, p = paired_stats(vals[a], vals[b])
        tests.append({"a": a, "b": b, "diff": m, "ci": [lo, hi], "p": p})
        print(f"  {a + '  vs  ' + b:<52}{m:>+10.4f} [{lo:>+7.4f},{hi:>+7.4f}]"
              f"{p:>10.5f}{'*' if p < 0.05 else ' '}")

    # ---- pre-specified policy x server-rule interaction ---------------------
    # Different families choosing different server rules is suggestive, not a
    # test. The contrast below is specified in advance and estimated on cells
    # every family shares, with the proximal strength and each family's own
    # parameter held at their middle values so that only policy, quantum and
    # server rule vary.
    #
    #   gain(p, q) = score(p, q, adam) - score(p, q, none)      per seed
    #   interaction(p1, p2, q) = gain(p1, q) - gain(p2, q)      paired per seed
    #
    # A non-zero interaction means the benefit of the adaptive server rule
    # depends on which participation policy is running it.
    print(f"\n  Pre-specified interaction: does the benefit of the adaptive "
          f"server rule depend on the policy?")
    def cell(fam, q, agg):
        nm = (f"{fam}|{q[0]}{q[1]}|{agg}|mu0.01"
              f"|k{FAMILIES[fam]['knob'][1]}")
        if all((nm, s_) in idx_probe for s_ in seeds):
            return np.array([idx_probe[(nm, s_)] for s_ in seeds])
        return None

    interactions = []
    fams = list(FAMILIES)
    print(f"  {'quantum':<12}{'policy pair':<28}{'interaction':>13}"
          f"{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 84)
    for q in QUANTA:
        gains = {}
        for fam in fams:
            hi, lo = cell(fam, q, "adam"), cell(fam, q, "none")
            if hi is not None and lo is not None:
                gains[fam] = hi - lo
        for i, f1 in enumerate(fams):
            for f2 in fams[i + 1:]:
                if f1 not in gains or f2 not in gains:
                    continue
                d = gains[f1] - gains[f2]
                m, lo_, hi_, pv = paired_stats(d, np.zeros_like(d))
                interactions.append({"quantum": f"{q[0]}{q[1]}", "pair": f"{f1}-{f2}",
                                     "interaction": m, "ci": [lo_, hi_], "p": pv})
                print(f"  {q[0]+str(q[1]):<12}{f1 + ' - ' + f2:<28}{m:>+13.4f}"
                      f" [{lo_:>+7.4f},{hi_:>+7.4f}]{pv:>10.5f}"
                      f"{'*' if pv < 0.05 else ' '}")
    if interactions:
        sig = [i for i in interactions if i["p"] < 0.05]
        print(f"\n  {len(sig)} of {len(interactions)} pairwise interactions "
              f"significant at 0.05 (unadjusted; {len(interactions)} contrasts "
              f"were specified in advance, so read them as a family)")
    else:
        print("  (shared cells unavailable; interaction not estimated)")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"policy_family_selection_balanced_{regime}.json"
    path.write_text(json.dumps({"regime": regime, "seeds": seeds, "config": cfgd,
                                "families": {f: [a["name"] for a in family_grid(f)]
                                             for f in FAMILIES},
                                "tests": tests, "interactions": interactions,
                                "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
