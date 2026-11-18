#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the service order alone is worth, with nothing else changed.

An earlier study in this programme compared the published policy against an arm
that also carried deficit round robin's admission rule, which withholds an
eligible client until it has accrued credit. That arm makes different
participations, so its advantage cannot be attributed to ordering. The present
script measures what remains.

Here the eligibility test is the published one, character for character, and
only the sort key among eligible clients changes. Every client the published
policy would serve in a round remains available in that round, so the arms make
the same participations in the same quantity and differ only in sequence.
Consumed energy and update counts are therefore equal to machine precision,
which the script checks and reports rather than assumes.

Three orders are compared against the published least-trained-first rule:
service by accrued deficit, service by deficit per unit cost, and a random
permutation. The third is the null: it says what an arbitrary order is worth
against the published one, and so bounds how much of any gain is attributable to
the specific rule rather than to the published order being poor.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch

torch.set_num_threads(1)

from scipy import stats

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating, GatePriority
from src.journal.study_policy_comparison import trajectory, class_weights

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
ARMS = ["published", "deficit", "percost", "random"]


def job(args):
    seed, regime, tau, epochs = args
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=epochs)
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    cw = class_weights(fed)
    rg = REGIMES[regime]
    out = []
    for arm in ARMS:
        pol = (ThresholdGating("g", cfg.m, tau) if arm == "published"
               else GatePriority("g", cfg.m, tau, mode=arm, seed=seed))
        tr = trajectory(fed, rg, cfg, pol, seed, cw)
        out.append(dict(regime=regime, tau=tau, epochs=epochs, arm=arm, seed=seed,
                        terminal=tr["traj"][-1]["macro_f1"],
                        energy_j=tr["total_energy_j"], updates=tr["total_updates"]))
    return out


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    tau, epochs = 0.3, 2
    seeds = list(range(1, 11))
    print(f"order only | regime={regime} tau={tau} w={epochs} seeds={len(seeds)}", flush=True)

    from multiprocessing import Pool
    with Pool(workers()) as pool:
        rows = [r for c in pool.map(job, [(s, regime, tau, epochs) for s in seeds]) for r in c]

    get = lambda a, f: np.array([r[f] for r in sorted(
        [x for x in rows if x["arm"] == a], key=lambda x: x["seed"])])
    base = get("published", "terminal")
    print(f"\n  {'arm':<10}{'terminal':>10}{'gain':>10}{'95% CI':>22}{'p':>10}"
          f"{'ahead':>8}{'max|dE|':>9}{'max|dU|':>9}")
    print(f"  {'published':<10}{base.mean():>10.4f}")
    summary = []
    for arm in ARMS[1:]:
        a, d = get(arm, "terminal"), get(arm, "terminal") - base
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        p = float(stats.ttest_rel(get(arm, "terminal"), base).pvalue)
        dE = float(np.abs(get(arm, "energy_j") - get("published", "energy_j")).max())
        dU = int(np.abs(get(arm, "updates") - get("published", "updates")).max())
        print(f"  {arm:<10}{a.mean():>10.4f}{d.mean():>+10.4f}"
              f"{f'[{d.mean()-h:+.4f},{d.mean()+h:+.4f}]':>22}{p:>10.4g}"
              f"{f'{int((d>0).sum())}/10':>8}{dE:>9.3g}{dU:>9}")
        summary.append(dict(arm=arm, terminal=float(a.mean()), gain=float(d.mean()),
                            lo=float(d.mean()-h), hi=float(d.mean()+h), p=p,
                            ahead=int((d > 0).sum()), max_abs_dE=dE, max_abs_dU=dU))

    path = OUT / f"order_only_{regime}.json"
    path.write_text(json.dumps(dict(regime=regime, tau=tau, epochs=epochs, seeds=seeds,
                                    published=float(base.mean()),
                                    summary=summary, rows=rows), indent=1))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
