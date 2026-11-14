"""Deficit round robin across the local work quantum, with and without the
single-board-computer-tier partitions (Table 8 and Figure 3 of the article,
and the sensitivity requested by Reviewer 1).

Run from the repository root:  python -m src.revision.study_quantum
"""
from __future__ import annotations

from src.journal.participation_policies import DeficitRoundRobin
from src.revision.common import (SEEDS, class_weights, federation, make_stock,
                                 pool_map, regime, save, single_level, train)

QUANTA = [1, 2, 4, 8, 16, 32]
FEDERATIONS = {"full": (), "no-sbc": ("sbc",)}


def job(args):
    seed, fname, w = args
    fed, cfg = federation(seed, w=w, drop_tiers=FEDERATIONS[fname])
    rg = regime("R2f-bind")
    cw = class_weights(fed)
    stock, hw, P = make_stock(fed, rg, cfg)
    step = single_level(DeficitRoundRobin("drr", cfg.m), stock)
    r = train(fed, cfg, rg, lambda t: (step(t, cfg.rounds),
                                       lambda: stock.initial.sum() - stock.remaining.sum()),
              seed, cw)
    widths = [p["width"] for p in r["traj"] if p["width"] > 0]
    return dict(seed=seed, federation=fname, w=w, classes=fed.classes,
                last_width=widths[-1] if widths else 0,
                updates=int(stock.participations.sum()),
                energy_j=float(stock.initial.sum() - stock.remaining.sum()), **r)


def main():
    jobs = [(s, f, w) for f in FEDERATIONS for w in QUANTA for s in SEEDS]
    rows = pool_map(job, jobs)
    save("quantum_full_and_no_sbc_R2f-bind", dict(quanta=QUANTA, seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
