"""The training allocation swept (Reviewer 1).

The allocation is the share of each cell budgeted to federated training. It is
swept from half to twice the binding value of the article, through the slack
value. At each allocation:

* energy only, over ten splits: allocations overdrawn by the filter alone and by
  the two-level design at kappa in {1.25, 1.5, 2};
* trained, over ten splits: unconstrained participation, the published gate and
  the gate with deficit order under an exact model, and the filter alone against
  the two-level design at kappa = 1.5.

Run from the repository root:  python -m src.revision.study_allocation
"""
from __future__ import annotations

from src.journal.participation_policies import FedAvgFull, GatePriority, ThresholdGating
from src.revision.common import (SEEDS, class_weights, federation, make_stock,
                                 pool_map, regime, save, single_level, train)
from src.revision.stock import RevStock, run_round_rev

TAU, NOISE, DELTA = 0.30, 0.10, 0.25
ALLOCATIONS = [2.0e-5, 3.0e-5, 4.0e-5, 6.0e-5, 8.6e-5]


def job(args):
    seed, alloc = args
    fed, cfg = federation(seed, w=2)
    rg = regime("R2f-bind", allocation=alloc)
    cw = class_weights(fed)
    rows = []
    # energy only
    for kappa in (1.25, 1.5, 2.0):
        for check in (False, True):
            base, hw, P = make_stock(fed, rg, cfg)
            st = RevStock(base, P, kappa=kappa, noise=NOISE, seed=seed)
            pol = ThresholdGating("g", cfg.m, TAU)
            for t in range(1, cfg.rounds + 1):
                run_round_rev(pol, t, cfg.rounds, st, DELTA, check)
            rows.append(dict(kind="energy", seed=seed, allocation=alloc, kappa=kappa,
                             design="two-level" if check else "server only",
                             overdrawn=int((st.true_remaining < 0).sum()),
                             participating=int((st.participations > 0).sum()),
                             participations=int(st.participations.sum())))
    # trained, exact model
    for name, mk in (("FedAvg-full", lambda: FedAvgFull("f", cfg.m)),
                     ("Published gate", lambda: ThresholdGating("g", cfg.m, TAU)),
                     ("Gate, deficit order",
                      lambda: GatePriority("d", cfg.m, TAU, mode="deficit", seed=seed))):
        stock, hw, P = make_stock(fed, rg, cfg)
        step = single_level(mk(), stock)
        r = train(fed, cfg, rg, lambda t: (step(t, cfg.rounds),
                                           lambda: stock.initial.sum() - stock.remaining.sum()),
                  seed, cw, eval_every=False)
        r.pop("traj")
        rows.append(dict(kind="trained", seed=seed, allocation=alloc, kappa=1.0, arm=name,
                         overdrawn=0, updates=int(stock.participations.sum()),
                         energy_j=float(stock.initial.sum() - stock.remaining.sum()),
                         unspent=float(stock.stranded_fraction()), **r))
    # trained, kappa = 1.5
    for name, check in (("server only", False), ("two-level", True)):
        base, hw, P = make_stock(fed, rg, cfg)
        st = RevStock(base, P, kappa=1.5, noise=NOISE, seed=seed)
        pol = ThresholdGating("g", cfg.m, TAU)
        r = train(fed, cfg, rg,
                  lambda t: (run_round_rev(pol, t, cfg.rounds, st, DELTA, check),
                             lambda: (st.initial - st.true_remaining).sum()),
                  seed, cw, eval_every=False)
        r.pop("traj")
        rows.append(dict(kind="trained", seed=seed, allocation=alloc, kappa=1.5, arm=name,
                         overdrawn=int((st.true_remaining < 0).sum()),
                         updates=int(st.participations.sum()),
                         energy_j=float((st.initial - st.true_remaining).sum()), **r))
    return rows


def main():
    jobs = [(s, a) for a in ALLOCATIONS for s in SEEDS]
    rows = [r for c in pool_map(job, jobs) for r in c]
    save("allocation_sweep_R2f-bind", dict(allocations=ALLOCATIONS, tau=TAU, delta=DELTA,
                                           noise=NOISE, seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
