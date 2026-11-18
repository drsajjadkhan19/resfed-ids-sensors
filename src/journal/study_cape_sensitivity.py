#!/usr/bin/env python3
"""
Sensitivity of CAPE to its three free settings.

The proximal strength mu0, the quantum ceiling w_max and the evidence reference
mass M_ref are fixed a priori. If the reported gain depended on a fortunate
choice of any of them, the result would reflect tuning rather than the method,
so each is swept around its default with the other two held fixed.

A response that is flat over a wide range indicates the mechanism is robust to
the setting. A response peaking exactly at the default would indicate the
defaults had been chosen against the evaluation set, and is reported as such.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig, evaluate
from src.journal.architectures import build, n_params
from src.journal.energy_model import EnergyStock, DeviceProfile, EnergyModel
from src.journal.device_profiles import assign_profiles
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import DeficitRoundRobin
from src.journal.cape import amortized_quantum, EvidenceAggregator, train_local_prox
from src.journal.study_policy_comparison import class_weights

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
E_TX = 6.4e-7
DEFAULTS = dict(mu0=0.01, w_max=16, mref_mult=1.0)


def run(fed, regime, cfg, seed, cw, mu0, w_max, mref_mult):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    P = n_params(build(regime.backbone, fed.n_features, fed.n_classes))
    hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
    n = np.array(fed.client_sizes, float)
    comm = np.full(len(hw), 2.0 * P * 4 * E_TX)
    a = np.array([6.0 * P * ni * h.j_per_flop for ni, h in zip(n, hw)])
    w_vec = amortized_quantum(comm, a, 1, w_max)
    profiles = [DeviceProfile(h.capacity_j * regime.allocation, 1.0, h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(P, 1), fed.client_sizes)
    stock.cost = comm + a * w_vec

    pol = DeficitRoundRobin("DRR", cfg.m)
    m_ref = float(np.sort(n)[::-1][:cfg.m].sum()) * mref_mult
    agg = EvidenceAggregator(m_ref)
    mu = mu0 * (w_vec / max(np.median(w_vec), 1e-9))

    model = build(regime.backbone, fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy(); traj = []
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        ups, mass = [], 0.0
        gp = [p.detach().clone() for p in model.parameters()]
        for k in sel:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(regime.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            train_local_prox(loc, Xb, yb, int(w_vec[k]), cfg.lr, cfg.batch_size,
                             dev, cw, float(mu[k]), gp)
            ups.append((loc.state_dict(), len(yb))); mass += len(yb); stock.charge(k)
        if ups:
            model.load_state_dict(agg.step(model.state_dict(), ups, mass))
        if t % 2 == 0 or t == cfg.rounds:
            traj.append(evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"])
    return float(np.mean(traj[-5:])), float(stock.initial.sum() - stock.remaining.sum()), int(np.median(w_vec))


def job(args):
    seed, knob, val, regime_name, cfgd = args
    cfg = FedConfig(**cfgd); cfg.seed = seed
    fed = Federation(cfg); cw = class_weights(fed)
    kw = dict(DEFAULTS); kw[knob] = val
    f1, j, wmed = run(fed, REGIMES[regime_name], cfg, seed, cw, **kw)
    return {"knob": knob, "value": val, "seed": seed, "f1": f1, "energy_j": j, "w_median": wmed}


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    grid = {"mu0": [0.0, 0.001, 0.01, 0.1, 1.0],
            "w_max": [4, 8, 16, 32],
            "mref_mult": [0.25, 0.5, 1.0, 2.0, 4.0]}
    print(f"CAPE sensitivity | regime={regime} seeds={len(seeds)}")
    print(f"  defaults: {DEFAULTS}\n", flush=True)

    from multiprocessing import Pool
    jobs = [(s, k, v, regime, cfgd) for k, vs in grid.items() for v in vs for s in seeds]
    t0 = time.time()
    with Pool(workers()) as pool:
        rows = pool.map(job, jobs)
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    for k, vs in grid.items():
        print(f"  {k}  (default {DEFAULTS[k]}):")
        print(f"    {'value':>8} {'macro-F1':>18} {'J':>8} {'w_med':>6}")
        best = None
        for v in vs:
            sub = [r for r in rows if r["knob"] == k and r["value"] == v]
            f = np.array([r["f1"] for r in sub])
            mark = "  <- default" if v == DEFAULTS[k] else ""
            print(f"    {str(v):>8} {f.mean():>10.4f}±{f.std(ddof=1):<7.4f} "
                  f"{np.mean([r['energy_j'] for r in sub]):>8.1f} "
                  f"{np.median([r['w_median'] for r in sub]):>6.0f}{mark}")
            if best is None or f.mean() > best[1]:
                best = (v, f.mean())
        spread = max(np.mean([r["f1"] for r in rows if r["knob"] == k and r["value"] == v])
                     for v in vs) - min(np.mean([r["f1"] for r in rows if r["knob"] == k and r["value"] == v])
                                        for v in vs)
        verdict = ("DEFAULT IS OPTIMAL -- check for tuning" if best[0] == DEFAULTS[k]
                   else f"optimum at {best[0]}, not the default")
        print(f"    -> range spread {spread:.4f}; {verdict}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cape_sensitivity_{regime}.json").write_text(json.dumps(
        {"regime": regime, "defaults": DEFAULTS, "grid": grid, "rows": rows},
        indent=1, default=str))
    print(f"Saved -> {OUT/f'cape_sensitivity_{regime}.json'}")


if __name__ == "__main__":
    main()
