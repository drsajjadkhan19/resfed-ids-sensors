"""Detection under a misspecified cost model at several client margins, with the
server-side margin, and with a meter error (Reviewers 1 and 2).

Arms at tau = 0.30, w = 2, binding regime, jitter of 10%:
    server only                 the published filter alone
    two-level, delta            delta in {0.10, 0.25, 0.50}
    two-level, server margin    delta = 0.25 and the server filters with it too
    two-level, meter error      delta = 0.25 and a noisy meter of up to 10%

Run from the repository root:  python -m src.revision.study_twolevel_margin
"""
from __future__ import annotations

from src.journal.participation_policies import ThresholdGating
from src.revision.common import (SEEDS, class_weights, federation, make_stock,
                                 pool_map, regime, save, train)
from src.revision.stock import RevStock, run_round_rev

TAU, NOISE = 0.30, 0.10
KAPPAS = [1.0, 1.5, 2.0]
ARMS = [
    ("server only", dict(check=False, delta=0.25)),
    ("two-level, delta 0.10", dict(check=True, delta=0.10)),
    ("two-level, delta 0.25", dict(check=True, delta=0.25)),
    ("two-level, delta 0.50", dict(check=True, delta=0.50)),
    ("two-level, server margin", dict(check=True, delta=0.25, server_margin=0.25)),
    ("two-level, meter error 10%", dict(check=True, delta=0.25, gamma=0.10)),
]


def job(args):
    seed, kappa = args
    fed, cfg = federation(seed, w=2)
    rg = regime("R2f-bind")
    cw = class_weights(fed)
    out = []
    for name, a in ARMS:
        base, hw, P = make_stock(fed, rg, cfg)
        st = RevStock(base, P, kappa=kappa, noise=NOISE, seed=seed,
                      gamma=a.get("gamma", 0.0), server_margin=a.get("server_margin", 0.0))
        pol = ThresholdGating("g", cfg.m, TAU)
        r = train(fed, cfg, rg,
                  lambda t: (run_round_rev(pol, t, cfg.rounds, st, a["delta"], a["check"]),
                             lambda: (st.initial - st.true_remaining).sum()),
                  seed, cw)
        out.append(dict(seed=seed, kappa=kappa, arm=name, classes=fed.classes,
                        overdrawn=int((st.true_remaining < 0).sum()),
                        participations=int(st.participations.sum()),
                        refusals=int(st.refusals.sum()),
                        energy_j=float((st.initial - st.true_remaining).sum()),
                        min_fraction=float((st.true_remaining / st.initial).min()), **r))
    return out


def main():
    jobs = [(s, k) for k in KAPPAS for s in SEEDS]
    rows = [r for c in pool_map(job, jobs) for r in c]
    save("twolevel_margin_R2f-bind", dict(tau=TAU, noise=NOISE, kappas=KAPPAS,
                                          seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
