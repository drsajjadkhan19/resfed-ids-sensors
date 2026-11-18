#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which clients serve the last rounds, and what that costs the delivered model.

Under a finite stock the schedule is fixed by the energy parameters alone
(Proposition 4), so who participates in each round can be computed without
training anything. That makes one quantity cheap to measure and, as it turns
out, strongly predictive: the number of trailing active rounds served by the
highest-capacity tier and no other.

A policy that excludes on state of charge removes low-capacity clients first,
because charge tracks capacity. Its last rounds are therefore served by a
narrowing subpopulation whose class mixture need not resemble the task. A policy
that paces service by remaining budget keeps the weak tiers alive to the end and
has no such tail.

This script replays every policy's schedule across seeds, records the trailing
single-tier run length and the class mixture of the last rounds, and joins them
to the terminal scores already measured in the policy comparison. It trains
nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from src.journal.federation import Federation, FedConfig
from src.journal.study_threshold_characterisation import make_stock
from src.journal.device_profiles import assign_profiles
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import (ThresholdGating, FedAvgFull, RandomM,
                                                DeficitRoundRobin, TokenBucket, LACS)

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

POLICIES = {
    "FedAvg-full":   lambda m, s: FedAvgFull("FedAvg-full", m),
    "Threshold-0.3": lambda m, s: ThresholdGating("Threshold-0.3", m, 0.3),
    "Random-m":      lambda m, s: RandomM("Random-m", m, seed=s),
    "TokenBucket":   lambda m, s: TokenBucket("TokenBucket", m, burst_rounds=2.0),
    "DRR":           lambda m, s: DeficitRoundRobin("DRR", m),
    "LACS-pace":     lambda m, s: LACS("LACS-pace", m, use_coverage=False),
}


def replay(name, seed, regime, cfgd, tail=3):
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    stock, _, _ = make_stock(fed, REGIMES[regime], cfg)
    pol = POLICIES[name](cfg.m, seed)

    active = []
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        for k in sel:
            stock.charge(k)
        if sel:
            active.append(sel)

    def tiers_of(round_):
        return {assign_profiles([fed.device_names[fed.client_devices[k]]])[0].name
                for k in round_}

    top = "sbc"                      # the highest-capacity tier in Table 3
    run = 0
    for r in reversed(active):
        if tiers_of(r) == {top}:
            run += 1
        else:
            break

    idx = np.concatenate([fed.client_idx[fed.client_devices[k]]
                          for r in active[-tail:] for k in r])
    y = fed.y[idx]
    pool = np.concatenate([fed.y[ix] for ix in fed.client_idx.values()])

    return dict(policy=name, seed=seed,
                trailing_single_tier=run,
                active_rounds=len(active),
                final_width=len(active[-1]),
                benign_share_last=float(np.mean(y == 0)),
                benign_share_pool=float(np.mean(pool == 0)),
                classes_last=int(len(set(y.tolist()))))


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    seeds = list(range(1, 11))
    print(f"late-round survivorship | regime={regime} seeds={len(seeds)}", flush=True)

    rows = [replay(n, s, regime, cfgd) for n in POLICIES for s in seeds]

    terminal = {}
    pc = OUT / f"policy_comparison_{regime}.json"
    if pc.exists():
        per = json.load(open(pc))["per_seed"]
        for r in rows:
            if r["policy"] in per:
                r["terminal"] = per[r["policy"]][r["seed"] - 1]["A_final"]
                terminal[(r["policy"], r["seed"])] = r["terminal"]

    print(f"\n  {'policy':<16}{'trailing single-tier':>22}{'benign last 3':>15}{'terminal':>10}")
    for n in POLICIES:
        rs = [r for r in rows if r["policy"] == n]
        run = [r["trailing_single_tier"] for r in rs]
        ben = np.mean([r["benign_share_last"] for r in rs])
        ter = [r.get("terminal") for r in rs if r.get("terminal") is not None]
        print(f"  {n:<16}{f'{min(run)}-{max(run)}':>22}{100*ben:>14.1f}%"
              f"{(np.mean(ter) if ter else float('nan')):>10.4f}")

    paired = [(r["trailing_single_tier"], r["terminal"]) for r in rows if "terminal" in r]
    if len(paired) > 2:
        x, y = np.array([p[0] for p in paired]), np.array([p[1] for p in paired])
        rho, p = stats.spearmanr(x, y)
        print(f"\n  trailing single-tier rounds vs terminal macro-F1, n={len(x)}: "
              f"Spearman rho = {rho:.3f}, p = {p:.3g}")
        bins = [(1, 2), (3, 4), (5, 6), (7, 99)]
        print(f"  {'trailing':<12}{'n':>4}{'mean terminal':>16}")
        for lo, hi in bins:
            m = (x >= lo) & (x <= hi)
            if m.sum():
                print(f"  {f'{lo}-{hi if hi<99 else chr(43)}':<12}{int(m.sum()):>4}{y[m].mean():>16.4f}")
        summary = dict(spearman_rho=float(rho), spearman_p=float(p), n=int(len(x)))
    else:
        summary = {}

    path = OUT / f"late_round_survivorship_{regime}.json"
    path.write_text(json.dumps(dict(regime=regime, config=cfgd, seeds=seeds,
                                    summary=summary, rows=rows), indent=1))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
