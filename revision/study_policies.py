"""Policies on the full federation and without the single-board-computer tier.

One study answers four review requests with the same runs: per-class detection
(Reviewer 3), detection on the held-out devices of each hardware tier
(Reviewer 2), the order-only contrast without the single-board-computer-tier
partitions (Reviewer 1) and the detection-against-energy curves (Reviewer 2).

Arms, at tau = 0.30, w = 2, binding regime, exact cost model:
    FedAvg-full                 every affordable client, up to m
    Published gate              state-of-charge gating, least-trained first
    Gate, deficit order         the published eligibility, ordered by deficit
    Gate, deficit per cost      the published eligibility, ordered by deficit/cost
    Gate, random order          the published eligibility, random order
    DRR                         deficit round robin

Run from the repository root:  python -m src.revision.study_policies
"""
from __future__ import annotations

from src.journal.participation_policies import (DeficitRoundRobin, FedAvgFull,
                                                GatePriority, ThresholdGating)
from src.revision.common import (SEEDS, class_weights, federation, make_stock,
                                 pool_map, regime, save, single_level, train)

TAU = 0.30
FEDERATIONS = {"full": (), "no-sbc": ("sbc",)}


def arms(m, seed):
    return {
        "FedAvg-full": lambda: FedAvgFull("f", m),
        "Published gate": lambda: ThresholdGating("g", m, TAU),
        "Gate, deficit order": lambda: GatePriority("d", m, TAU, mode="deficit", seed=seed),
        "Gate, deficit per cost": lambda: GatePriority("c", m, TAU, mode="percost", seed=seed),
        "Gate, random order": lambda: GatePriority("r", m, TAU, mode="random", seed=seed),
        "DRR": lambda: DeficitRoundRobin("drr", m),
    }


def job(args):
    seed, fname = args
    fed, cfg = federation(seed, w=2, drop_tiers=FEDERATIONS[fname])
    rg = regime("R2f-bind")
    cw = class_weights(fed)
    out = []
    for name, mk in arms(cfg.m, seed).items():
        stock, hw, P = make_stock(fed, rg, cfg)
        step = single_level(mk(), stock)
        r = train(fed, cfg, rg, lambda t: (step(t, cfg.rounds),
                                           lambda: stock.initial.sum() - stock.remaining.sum()),
                  seed, cw)
        out.append(dict(seed=seed, federation=fname, arm=name,
                        n_clients=len(fed.client_devices),
                        participating=int((stock.participations > 0).sum()),
                        updates=int(stock.participations.sum()),
                        energy_j=float(stock.initial.sum() - stock.remaining.sum()),
                        unspent=float(stock.stranded_fraction()),
                        classes=fed.classes, **r))
    return out


def main():
    jobs = [(s, f) for f in FEDERATIONS for s in SEEDS]
    rows = [r for chunk in pool_map(job, jobs) for r in chunk]
    save("policies_full_and_no_sbc_R2f-bind", dict(tau=TAU, seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
