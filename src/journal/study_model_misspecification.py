#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the client-side check buys when the server's energy model is wrong.

Every guarantee in this article rests on a cost model built from published
device characteristics rather than from measurement. The obvious objection is
that the model may be wrong. The published policy already answers it, in a part
of its design the journal implementation had dropped: the selected client checks
its own measured state before training and refuses the task if it cannot afford
it.

This script makes the objection quantitative. The server keeps its nominal cost
model; the true cost is kappa times larger, with multiplicative noise. Three
designs are compared at each kappa:

    server only     the server filters on its own estimate; the client obeys
    two-level       the client refuses when its measured energy cannot cover
                    the expected cost with a margin delta
    oracle          the server's model is correct, as a reference

Depletion is read from the true stock, so a design that overdraws is visible
even though the server believes everything is fine. No training is needed for
the depletion counts because the schedule is fixed by the energy parameters, so
the sweep is cheap and can be run over many kappa.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating
from src.journal.study_threshold_characterisation import make_stock
from src.journal.two_level import DualStock, run_round

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
KAPPAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
DESIGNS = {"server-only": False, "two-level": True}


def one(seed, regime, kappa, client_check, tau, delta, noise, cfgd):
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    base, _, _ = make_stock(fed, REGIMES[regime], cfg)
    stock = DualStock(base, kappa=kappa, noise=noise, seed=seed)
    pol = ThresholdGating("g", cfg.m, tau)
    for t in range(1, cfg.rounds + 1):
        run_round(pol, t, cfg.rounds, stock, delta, client_check)
    return dict(seed=seed, regime=regime, kappa=kappa,
                design="two-level" if client_check else "server-only",
                depleted=stock.depleted(), drained=stock.drained(0.10),
                participations=int(stock.participations.sum()),
                refusals=int(stock.refusals.sum()),
                min_true_fraction=float((stock.true_remaining / stock.initial).min()))


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    tau, delta, noise = 0.3, 0.25, 0.10
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    seeds = list(range(1, 11))
    print(f"model misspecification | regime={regime} tau={tau} delta={delta} "
          f"noise={noise} seeds={len(seeds)}", flush=True)

    rows = [one(s, regime, k, chk, tau, delta, noise, cfgd)
            for k in KAPPAS for name, chk in DESIGNS.items() for s in seeds]

    print(f"\n  {'kappa':>6}  {'design':<12}{'depleted':>10}{'drained<10%':>13}"
          f"{'participations':>16}{'refusals':>10}{'min true frac':>15}")
    summary = []
    for k in KAPPAS:
        for name in DESIGNS:
            sel = [r for r in rows if r["kappa"] == k and r["design"] == name]
            d = np.mean([r["depleted"] for r in sel])
            dr = np.mean([r["drained"] for r in sel])
            pa = np.mean([r["participations"] for r in sel])
            rf = np.mean([r["refusals"] for r in sel])
            mf = np.mean([r["min_true_fraction"] for r in sel])
            print(f"  {k:>6.2f}  {name:<12}{d:>10.1f}{dr:>13.1f}{pa:>16.1f}{rf:>10.1f}{mf:>15.3f}")
            summary.append(dict(kappa=k, design=name, depleted=float(d), drained=float(dr),
                                participations=float(pa), refusals=float(rf),
                                min_true_fraction=float(mf)))
        print()

    path = OUT / f"model_misspecification_{regime}.json"
    path.write_text(json.dumps(dict(regime=regime, tau=tau, delta=delta, noise=noise,
                                    kappas=KAPPAS, seeds=seeds, summary=summary,
                                    rows=rows), indent=1))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
