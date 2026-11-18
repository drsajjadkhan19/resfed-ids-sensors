#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the client-side check costs in detection performance.

The misspecification sweep shows that the published two-level design preserves
every device where a server-side filter alone overdraws batteries. It also
shows that the check refuses a great many rounds once the server's model is
wrong, and a refusal is a participation that does not happen. Whether that costs
detection performance is a separate question, and this script trains the models
needed to answer it.

Four arms are run at each misspecification factor: unconstrained participation,
the published gate with the server-side filter only, the published gate with the
client-side check, and the client-side check with deficit service order. The
endpoint is the terminal model under the budget, evaluated on held-out devices,
which is what a deployment receives.
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

from src.journal.architectures import build
from src.journal.federation import Federation, FedConfig, _train_local, evaluate
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating, FedAvgFull, GatePriority
from src.journal.study_policy_comparison import class_weights
from src.journal.study_threshold_characterisation import make_stock
from src.journal.two_level import DualStock, run_round

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
KAPPAS = [1.0, 1.5, 2.0]


def arms(cfg, tau, seed):
    return [
        ("FedAvg-full",        FedAvgFull("f", cfg.m),                       False),
        ("ResFed server-only", ThresholdGating("t", cfg.m, tau),             False),
        ("ResFed two-level",   ThresholdGating("t", cfg.m, tau),             True),
        ("ResFed two-level + deficit order",
         GatePriority("t", cfg.m, tau, mode="deficit", seed=seed),           True),
    ]


def run(seed, regime, kappa, tau, delta, noise, cfgd):
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    cw = class_weights(fed)
    rg = REGIMES[regime]
    base, _, _ = make_stock(fed, rg, cfg)
    Xe, ye = fed.eval_xy()
    dev = torch.device("cpu")

    out = []
    for name, pol, check in arms(cfg, tau, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        stock = DualStock(base, kappa=kappa, noise=noise, seed=seed)
        model = build(rg.backbone, fed.n_features, fed.n_classes)
        for t in range(1, cfg.rounds + 1):
            trained = run_round(pol, t, cfg.rounds, stock, delta, check)
            ups = []
            for k in trained:
                Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
                loc = build(rg.backbone, fed.n_features, fed.n_classes)
                loc.load_state_dict(model.state_dict())
                _train_local(loc, Xb, yb, cfg, dev, cw)
                ups.append((loc.state_dict(), len(yb)))
            if ups:
                tot = sum(n for _, n in ups)
                model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                       .to(ups[0][0][q].dtype) for q in ups[0][0]})
        out.append(dict(seed=seed, regime=regime, kappa=kappa, arm=name,
                        terminal=evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"],
                        depleted=stock.depleted(), drained=stock.drained(0.10),
                        participations=int(stock.participations.sum()),
                        refusals=int(stock.refusals.sum()),
                        energy_j=float((stock.initial - stock.true_remaining).sum())))
    return out


def job(a):
    return run(*a)


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    tau, delta, noise = 0.3, 0.25, 0.10
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    seeds = list(range(1, 11))
    print(f"two-level accuracy | regime={regime} kappas={KAPPAS} seeds={len(seeds)}", flush=True)

    jobs = [(s, regime, k, tau, delta, noise, cfgd) for k in KAPPAS for s in seeds]
    from multiprocessing import Pool
    with Pool(workers()) as pool:
        rows = [r for c in pool.map(job, jobs) for r in c]

    names = [n for n, _, _ in arms(FedConfig(**cfgd), tau, 1)]
    print(f"\n  {'kappa':>5}  {'arm':<34}{'terminal':>18}{'depleted':>10}"
          f"{'drained':>9}{'parts':>8}{'energy J':>10}")
    summary = []
    for k in KAPPAS:
        for n in names:
            sel = [r for r in rows if r["kappa"] == k and r["arm"] == n]
            v = np.array([r["terminal"] for r in sel])
            print(f"  {k:>5.2f}  {n:<34}{v.mean():>9.4f} +/-{v.std(ddof=1):<7.4f}"
                  f"{np.mean([r['depleted'] for r in sel]):>10.1f}"
                  f"{np.mean([r['drained'] for r in sel]):>9.1f}"
                  f"{np.mean([r['participations'] for r in sel]):>8.0f}"
                  f"{np.mean([r['energy_j'] for r in sel]):>10.1f}")
            summary.append(dict(kappa=k, arm=n, terminal=float(v.mean()),
                                sd=float(v.std(ddof=1)),
                                depleted=float(np.mean([r['depleted'] for r in sel])),
                                drained=float(np.mean([r['drained'] for r in sel])),
                                participations=float(np.mean([r['participations'] for r in sel])),
                                energy_j=float(np.mean([r['energy_j'] for r in sel]))))
        g = lambda n: np.array([r["terminal"] for r in sorted(
            [x for x in rows if x["kappa"] == k and x["arm"] == n], key=lambda z: z["seed"])])
        for a, b in (("ResFed two-level", "ResFed server-only"),
                     ("ResFed two-level", "FedAvg-full")):
            d = g(a) - g(b)
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"         {a} minus {b}: {d.mean():+.4f} "
                  f"[{d.mean()-h:+.4f},{d.mean()+h:+.4f}]")
        print()

    path = OUT / f"two_level_accuracy_{regime}.json"
    path.write_text(json.dumps(dict(regime=regime, tau=tau, delta=delta, noise=noise,
                                    kappas=KAPPAS, seeds=seeds, summary=summary,
                                    rows=rows), indent=1))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
