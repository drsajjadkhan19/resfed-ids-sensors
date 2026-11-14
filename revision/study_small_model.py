"""A backbone small enough for the microcontroller tier to participate (Reviewer 3).

With the backbone of the article (73,544 parameters) one exchange costs more
than the allocation of a microcontroller-tier client, so that tier never trains.
The ``mlp_tiny`` backbone (hidden widths 64, 32, 16) costs about a seventh as
much, and the tier can afford one or two participations. Each arm is run on the
full federation and with the microcontroller-tier clients removed, so the
difference measures what their data contribute.

Run from the repository root:  python -m src.revision.study_small_model
"""
from __future__ import annotations

from src.journal.architectures import build, n_params
from src.journal.participation_policies import DeficitRoundRobin, FedAvgFull, ThresholdGating
from src.revision.common import (SEEDS, class_weights, federation, make_stock,
                                 pool_map, regime, save, single_level, train)

TAU = 0.30
FEDERATIONS = {"full": (), "no-mcu": ("mcu",)}
BACKBONES = ["mlp_tiny", "mlp"]


def job(args):
    seed, fname, backbone = args
    fed, cfg = federation(seed, w=2, drop_tiers=FEDERATIONS[fname])
    rg = regime("R2f-bind", backbone=backbone)
    cw = class_weights(fed)
    out = []
    for name, mk in (("FedAvg-full", lambda: FedAvgFull("f", cfg.m)),
                     ("Published gate", lambda: ThresholdGating("g", cfg.m, TAU)),
                     ("DRR", lambda: DeficitRoundRobin("drr", cfg.m))):
        stock, hw, P = make_stock(fed, rg, cfg)
        step = single_level(mk(), stock)
        r = train(fed, cfg, rg, lambda t: (step(t, cfg.rounds),
                                           lambda: stock.initial.sum() - stock.remaining.sum()),
                  seed, cw, eval_every=False)
        r.pop("traj")
        mcu = [i for i, h in enumerate(hw) if h.name == "mcu"]
        out.append(dict(seed=seed, federation=fname, backbone=backbone, arm=name,
                        n_params=int(P), classes=fed.classes,
                        mcu_clients=len(mcu),
                        mcu_participating=int(sum(stock.participations[i] > 0 for i in mcu)),
                        updates=int(stock.participations.sum()),
                        energy_j=float(stock.initial.sum() - stock.remaining.sum()), **r))
    return out


def main():
    jobs = [(s, f, b) for b in BACKBONES for f in FEDERATIONS for s in SEEDS]
    rows = [r for c in pool_map(job, jobs) for r in c]
    save("small_model_mcu_R2f-bind", dict(tau=TAU, seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
