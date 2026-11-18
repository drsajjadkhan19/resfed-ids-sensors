#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the service order inside the published gate matter, and where?

The published policy makes two decisions. Eligibility, by state of charge
against a threshold, carries its depletion guarantee: Proposition 1 shows that a
threshold above the per-round cost fraction forbids depletion without reference
to any run. Order of service among the eligible carries no guarantee and is
stated without a reason.

Holding eligibility fixed and changing only the order gives a contrast in which
consumed energy, client updates and stranded stock are identical by
construction, split by split, because the same clients are eligible in the same
rounds and each still trains whenever it is served. Whatever the order changes,
it cannot be the budget.

This script runs that contrast across thresholds, across both budget regimes and
across local-work settings, so that the single configuration reported earlier is
either confirmed as a general property of the gate or confined to where it was
measured.
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
from src.journal.participation_policies import ThresholdGating, GatedDeficit
from src.journal.study_policy_comparison import trajectory, class_weights

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

TAUS = [0.2, 0.3, 0.5]
EPOCHS = [2, 8]


def job(args):
    seed, regime, epochs = args
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=epochs)
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    cw = class_weights(fed)
    rg = REGIMES[regime]
    out = []
    for tau in TAUS:
        for order, make in (("published", lambda: ThresholdGating("g", cfg.m, tau)),
                            ("deficit",   lambda: GatedDeficit("g", cfg.m, tau))):
            tr = trajectory(fed, rg, cfg, make(), seed, cw)
            last = tr["traj"][-1]
            out.append(dict(regime=regime, epochs=epochs, tau=tau, order=order,
                            seed=seed, terminal=last["macro_f1"],
                            energy_j=tr["total_energy_j"], updates=tr["total_updates"]))
    return out


def main():
    seeds = list(range(1, 11))
    jobs = [(s, rg, e) for rg in ("R2f-bind", "R2f-slack") for e in EPOCHS for s in seeds]
    print(f"gate order | {len(TAUS)} thresholds x 2 orders x {len(EPOCHS)} work settings "
          f"x 2 regimes x {len(seeds)} seeds", flush=True)

    from multiprocessing import Pool
    with Pool(workers()) as pool:
        rows = [r for chunk in pool.map(job, jobs) for r in chunk]

    print(f"\n  {'regime':<11}{'epochs':>7}{'tau':>6}{'published':>12}{'deficit':>10}"
          f"{'difference, 95% CI':>28}{'p':>9}")
    summary = []
    for rg in ("R2f-bind", "R2f-slack"):
        for e in EPOCHS:
            for tau in TAUS:
                sel = lambda o: np.array([r["terminal"] for r in rows
                                          if r["regime"] == rg and r["epochs"] == e
                                          and r["tau"] == tau and r["order"] == o])
                a, b = sel("deficit"), sel("published")
                d = a - b
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                p = float(stats.ttest_rel(a, b).pvalue) if d.std() > 0 else float("nan")
                print(f"  {rg:<11}{e:>7}{tau:>6.1f}{b.mean():>12.4f}{a.mean():>10.4f}"
                      f"{f'{d.mean():+.4f} [{d.mean()-h:+.4f},{d.mean()+h:+.4f}]':>28}{p:>9.2g}")
                summary.append(dict(regime=rg, epochs=e, tau=tau,
                                    published=float(b.mean()), deficit=float(a.mean()),
                                    diff=float(d.mean()), lo=float(d.mean()-h),
                                    hi=float(d.mean()+h), p=p))

    path = OUT / "gate_order.json"
    path.write_text(json.dumps(dict(taus=TAUS, epochs=EPOCHS, seeds=seeds,
                                    summary=summary, rows=rows), indent=1))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
