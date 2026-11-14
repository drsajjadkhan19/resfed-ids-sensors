"""Margin, meter error, refusal cost, server-side margin and error structure.

Energy only: by Proposition 4 the schedule and the charges do not depend on what
the model learns, so no training is needed for these counts. Each sweep is run
on the ten splits of the binding regime at tau = 0.30 and w = 2 unless stated.

    margin        delta in {0, 0.1, 0.25, 0.5, 1.0}, uniform error
    error         client-specific factors (log-normal around kappa) and an error
                  on the compute term only, at delta = 0.25
    meter         additive meter error of up to gamma of the allocation, noisy or
                  always high, at delta = 0.25
    refusal       each refusal costs one model download, or one full exchange
    server        the server's filter uses the client's margin as well
    quantum       refusals under an exact model at w in {1, 2, 4, 8, 16, 32},
                  with and without the server-side margin

Run from the repository root:  python -m src.revision.study_margin_meter
"""
from __future__ import annotations

import numpy as np

from src.journal.participation_policies import ThresholdGating
from src.revision.common import (SEEDS, federation, make_stock, regime, save, pool_map)
from src.revision.stock import RevStock, run_round_rev

TAU, NOISE = 0.30, 0.10
KAPPAS = [1.0, 1.25, 1.5, 2.0, 3.0]


def configs():
    out = []
    for k in KAPPAS:
        out.append(dict(sweep="margin", kappa=k, delta=0.25, check=False))
        for d in (0.0, 0.10, 0.25, 0.50, 1.00):
            out.append(dict(sweep="margin", kappa=k, delta=d, check=True))
        out.append(dict(sweep="server", kappa=k, delta=0.25, check=True, server_margin=0.25))
        for err in ("client", "compute"):
            out.append(dict(sweep="error", kappa=k, delta=0.25, check=True, error=err))
            out.append(dict(sweep="error", kappa=k, delta=0.25, check=False, error=err))
    for k in (1.0, 1.5, 2.0):
        for g in (0.02, 0.05, 0.10, 0.20):
            for mode in ("noisy", "high"):
                out.append(dict(sweep="meter", kappa=k, delta=0.25, check=True,
                                gamma=g, meter=mode))
        for r in ("download", "exchange"):
            out.append(dict(sweep="refusal", kappa=k, delta=0.25, check=True, refusal=r))
    for w in (1, 2, 4, 8, 16, 32):
        for sm in (0.0, 0.25):
            out.append(dict(sweep="quantum", kappa=1.0, delta=0.25, check=True, w=w,
                            server_margin=sm, noise=0.0))
    return out


def one(seed, fed, cfg, c):
    import copy
    rg = regime("R2f-bind")
    cfg = copy.copy(cfg)
    cfg.local_epochs = int(c.get("w", 2))
    base, hw, P = make_stock(fed, rg, cfg)
    K = len(base.initial)
    kvec = None
    if c.get("error") == "client":
        # log-normal per client with median kappa, so a quarter of the clients
        # are at least 18% above it and a quarter at least 15% below
        rng = np.random.default_rng(seed + 104729)
        kvec = c["kappa"] * np.exp(rng.normal(0.0, 0.25, K))
    comm = 2.0 * P * 4 * 6.4e-7
    refusal = {None: 0.0, "download": 0.5 * comm, "exchange": comm}[c.get("refusal")]
    st = RevStock(base, P, kappa=c["kappa"], noise=c.get("noise", NOISE), seed=seed,
                  kappa_vec=kvec, error_on=("compute" if c.get("error") == "compute" else "total"),
                  gamma=c.get("gamma", 0.0), meter=c.get("meter", "noisy"),
                  refusal_cost=refusal, server_margin=c.get("server_margin", 0.0))
    pol = ThresholdGating("g", cfg.m, TAU)
    for t in range(1, cfg.rounds + 1):
        run_round_rev(pol, t, cfg.rounds, st, c["delta"], c["check"])
    part = (st.participations > 0) | (st.true_remaining < st.initial)
    frac = st.true_remaining / st.initial
    bnd = st.bound(c["delta"])
    viol = int(np.sum(st.true_remaining < bnd - 1e-9)) if c["check"] else None
    return dict(c, seed=seed,
                overdrawn=int(np.sum(st.true_remaining < 0.0)),
                participating=int(part.sum()),
                participations=int(st.participations.sum()),
                refusals=int(st.refusals.sum()),
                refused_at_full_charge=int(np.sum(st.has_refused & (st.participations == 0))),
                min_fraction=float(frac[part].min()) if part.any() else 1.0,
                energy_j=float((st.initial - st.true_remaining).sum()),
                refusal_energy_j=float(st.refusal_spent.sum()),
                bound_violations=viol,
                closest_to_bound=(float(np.min((st.true_remaining - bnd) / st.initial))
                                  if c["check"] else None))


def job(seed):
    fed, cfg = federation(seed, w=2)
    return [one(seed, fed, cfg, c) for c in configs()]


def main():
    rows = [r for chunk in pool_map(job, SEEDS) for r in chunk]
    save("margin_meter_R2f-bind", dict(tau=TAU, noise=NOISE, kappas=KAPPAS,
                                       seeds=SEEDS, rows=rows))


if __name__ == "__main__":
    main()
