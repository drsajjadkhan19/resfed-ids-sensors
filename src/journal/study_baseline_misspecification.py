#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the recommended baseline any safer than the policy it beats?

Deficit round robin leads the published gating policy where the budget binds,
and this article recommends it as the baseline a proposed selector should be
measured against. That recommendation was made on an exact cost model, which is
the one condition the rest of the article argues cannot be assumed.

Deficit round robin is a single-level server policy: it checks affordability
against the charge the *server* believes a client to hold. Under
misspecification it should therefore overdraw exactly as the gating filter does,
and adding the client-side check of Definition 2 should bound it exactly as it
bounds the gating policy, because Theorem 1 does not refer to the selection rule.

This script tests both halves of that prediction. No training is needed: the
schedule is fixed by the energy parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import DeficitRoundRobin, ThresholdGating
from src.journal.study_threshold_characterisation import make_stock
from src.journal.two_level import DualStock, run_round

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
KAPPAS = [1.0, 1.25, 1.5, 2.0, 3.0]


def one(seed, regime, kappa, policy_name, client_check, tau, delta, noise, cfgd):
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    base, _, _ = make_stock(fed, REGIMES[regime], cfg)
    stock = DualStock(base, kappa=kappa, noise=noise, seed=seed)
    pol = (DeficitRoundRobin("drr", cfg.m) if policy_name == "deficit round robin"
           else ThresholdGating("gate", cfg.m, tau))
    for t in range(1, cfg.rounds + 1):
        run_round(pol, t, cfg.rounds, stock, delta, client_check)
    return dict(seed=seed, kappa=kappa, policy=policy_name, client_check=client_check,
                depleted=stock.depleted(),
                participations=int(stock.participations.sum()),
                min_true_fraction=float((stock.true_remaining / stock.initial).min()))


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    tau, delta, noise = 0.3, 0.25, 0.10
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    seeds = list(range(1, 11))
    arms = [("deficit round robin", False), ("deficit round robin", True),
            ("state-of-charge gating", False), ("state-of-charge gating", True)]
    print(f"baseline under misspecification | regime={regime} tau={tau} "
          f"delta={delta} noise={noise} seeds={len(seeds)}", flush=True)

    rows = [one(s, regime, k, p, c, tau, delta, noise, cfgd)
            for k in KAPPAS for p, c in arms for s in seeds]

    print(f"\n  {'kappa':>6}  {'policy':<24}{'level':<12}{'depleted':>10}"
          f"{'worst min':>12}{'parts':>8}")
    summary = []
    for k in KAPPAS:
        for p, c in arms:
            sel = [r for r in rows if r["kappa"] == k and r["policy"] == p
                   and r["client_check"] == c]
            d = float(np.mean([r["depleted"] for r in sel]))
            w = float(min(r["min_true_fraction"] for r in sel))
            pa = float(np.mean([r["participations"] for r in sel]))
            lvl = "two-level" if c else "server only"
            print(f"  {k:>6.2f}  {p:<24}{lvl:<12}{d:>10.1f}{w:>12.3f}{pa:>8.1f}")
            summary.append(dict(kappa=k, policy=p, client_check=c, depleted=d,
                                worst_min_fraction=w, participations=pa))
        print()

    path = OUT / f"baseline_misspecification_{regime}.json"
    path.write_text(json.dumps(dict(regime=regime, tau=tau, delta=delta, noise=noise,
                                    kappas=KAPPAS, seeds=seeds, summary=summary,
                                    rows=rows), indent=1))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
