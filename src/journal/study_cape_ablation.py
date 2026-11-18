#!/usr/bin/env python3
"""
Ablation of the CAPE mechanisms.

Each mechanism is enabled in turn so that its individual contribution, and any
interaction between mechanisms, can be attributed.

Arms (all under the SAME energy stock, same seeds, same evaluation):
  DRR-base      deficit round robin + vanilla FedAvg + fixed w        (best prior)
  +E            ... + evidence-weighted aggregation
  +C            ... + per-client cost-amortized work quantum
  +C+E          both
  CAPE          +C+E+P (work-proportional proximal)
plus Threshold-0.3 (the published policy) and FedAvg-full as references.
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
from src.journal.energy_model import EnergyStock, EnergyModel, DeviceProfile
from src.journal.device_profiles import assign_profiles
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import DeficitRoundRobin, ThresholdGating, FedAvgFull
from src.journal.cape import amortized_quantum, EvidenceAggregator, train_local_prox
from src.journal.study_policy_comparison import class_weights

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
E_TX = 6.4e-7


def build_stock(fed, regime, P, w_vec):
    """Energy stock with a PER-CLIENT work quantum w_vec."""
    hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
    n = np.array(fed.client_sizes, dtype=float)
    comm = np.full(len(hw), 2.0 * P * 4 * E_TX)            # c_k
    a = np.array([6.0 * P * ni * h.j_per_flop for ni, h in zip(n, hw)])   # a_k per epoch
    profiles = [DeviceProfile(capacity_j=h.capacity_j * regime.allocation,
                              eta=1.0, label=h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(P, 1), fed.client_sizes)
    stock.cost = comm + a * w_vec
    return stock, comm, a


def run_arm(fed, regime, cfg, arm, seed, cw):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    P = n_params(build(regime.backbone, fed.n_features, fed.n_classes))
    K = len(fed.client_devices)

    # --- Mechanism C: per-client work quantum -------------------------------
    _, comm, a = build_stock(fed, regime, P, np.ones(K))
    if arm["C"]:
        w_vec = amortized_quantum(comm, a, w_min=1, w_max=16)
    else:
        w_vec = np.full(K, cfg.local_epochs, dtype=int)
    stock, _, _ = build_stock(fed, regime, P, w_vec)

    # --- policy --------------------------------------------------------------
    pol = arm["policy"](cfg)

    # --- Mechanism E: evidence-weighted aggregation --------------------------
    sizes = np.array(fed.client_sizes, dtype=float)
    m_ref = float(np.sort(sizes)[::-1][:cfg.m].sum())      # a full-capacity round
    agg = EvidenceAggregator(m_ref) if arm["E"] else None

    # --- Mechanism P: work-proportional proximal -----------------------------
    w_ref = float(np.median(w_vec))
    mu_vec = (arm["mu0"] * (w_vec / max(w_ref, 1e-9))) if arm["P"] else np.zeros(K)

    model = build(regime.backbone, fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy()
    traj, last = [], 0
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        ups, mass = [], 0.0
        gparams = [p.detach().clone() for p in model.parameters()]
        for k in sel:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(regime.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            train_local_prox(loc, Xb, yb, int(w_vec[k]), cfg.lr, cfg.batch_size,
                             dev, cw, mu=float(mu_vec[k]), global_params=gparams)
            ups.append((loc.state_dict(), len(yb)))
            mass += len(yb)
            stock.charge(k)
        if ups:
            if agg is not None:
                model.load_state_dict(agg.step(model.state_dict(), ups, mass))
            else:
                tot = sum(n for _, n in ups)
                model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                       .to(ups[0][0][q].dtype) for q in ups[0][0]})
            last = t
        if t % 2 == 0 or t == cfg.rounds:
            traj.append({"round": t, "macro_f1": evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"],
                         "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
                         "width": len(sel)})
    return {"arm": arm["name"], "seed": seed,
            "final_macro_f1": traj[-1]["macro_f1"],
            "mean_last5": float(np.mean([p["macro_f1"] for p in traj[-5:]])),
            "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
            "updates": int(stock.participations.sum()),
            "last_active_round": last,
            "w_median": int(np.median(w_vec)), "w_max": int(w_vec.max()),
            "stranded": float(stock.stranded_fraction()),
            "traj": traj}


def arms(cfg):
    drr = lambda c: DeficitRoundRobin("DRR", c.m)
    return [
        {"name": "Threshold-0.3", "policy": lambda c: ThresholdGating("T", c.m, 0.3),
         "C": False, "E": False, "P": False, "mu0": 0.0},
        {"name": "FedAvg-full", "policy": lambda c: FedAvgFull("F", c.m),
         "C": False, "E": False, "P": False, "mu0": 0.0},
        {"name": "DRR-base", "policy": drr, "C": False, "E": False, "P": False, "mu0": 0.0},
        {"name": "DRR+E", "policy": drr, "C": False, "E": True, "P": False, "mu0": 0.0},
        {"name": "DRR+C", "policy": drr, "C": True, "E": False, "P": False, "mu0": 0.0},
        {"name": "DRR+C+E", "policy": drr, "C": True, "E": True, "P": False, "mu0": 0.0},
        {"name": "CAPE", "policy": drr, "C": True, "E": True, "P": True, "mu0": 0.01},
    ]


def job(args):
    seed, regime_name, cfgd = args
    cfg = FedConfig(**cfgd); cfg.seed = seed
    fed = Federation(cfg); cw = class_weights(fed)
    rg = REGIMES[regime_name]
    return [run_arm(fed, rg, cfg, a, seed, cw) for a in arms(cfg)]


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    print(f"CAPE ablation | regime={regime} seeds={len(seeds)}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, [(s, regime, cfgd) for s in seeds])
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    # ---- three pre-declared controls ---------------------------------------
    # Read off the same trajectory so they cannot be chosen after the fact.
    by_seed = {}
    for c in chunks:
        for r in c:
            by_seed.setdefault(r["seed"], []).append(r)
    for seed, runs in by_seed.items():
        ran = [r for r in runs if r["last_active_round"] > 0]
        Estar = min(r["energy_j"] for r in ran)
        Rstar = min(r["last_active_round"] for r in ran)
        for r in runs:
            tr = r["traj"]
            e = next((p["macro_f1"] for p in tr if p["energy_j"] >= Estar - 1e-9),
                     tr[-1]["macro_f1"])
            ok = [p for p in tr if p["round"] <= Rstar]
            r["B_equal_energy"] = e
            r["C_equal_rounds"] = (ok[-1] if ok else tr[0])["macro_f1"]
            r["E_star"], r["R_star"] = Estar, Rstar

    from scipy import stats
    names = [a["name"] for a in arms(FedConfig(**cfgd))]
    print(f"  {'arm':<15} {'final F1':>17} {'mean last5':>17} {'J':>7} {'upd':>5} {'w':>4} {'strand':>7}")
    print("  " + "-" * 82)
    agg = {}
    for n in names:
        rs = [r for r in rows if r["arm"] == n]
        f = np.array([r["final_macro_f1"] for r in rs])
        l5 = np.array([r["mean_last5"] for r in rs])
        agg[n] = {"final": f, "last5": l5}
        print(f"  {n:<15} {f.mean():>9.4f}±{f.std(ddof=1):<7.4f} {l5.mean():>9.4f}±{l5.std(ddof=1):<7.4f} "
              f"{np.mean([r['energy_j'] for r in rs]):>7.1f} {np.mean([r['updates'] for r in rs]):>5.0f} "
              f"{np.median([r['w_median'] for r in rs]):>4.0f} {np.mean([r['stranded'] for r in rs]):>7.3f}")

    print(f"\n  {'arm':<15} {'A last5':>17} {'B equal energy':>17} {'C equal rounds':>17}")
    print("  " + "-" * 70)
    for n in names:
        rs = [r for r in rows if r["arm"] == n]
        b = np.array([r["B_equal_energy"] for r in rs])
        c = np.array([r["C_equal_rounds"] for r in rs])
        agg[n]["B"], agg[n]["C"] = b, c
        print(f"  {n:<15} {agg[n]['last5'].mean():>9.4f}±{agg[n]['last5'].std(ddof=1):<7.4f} "
              f"{b.mean():>9.4f}±{b.std(ddof=1):<7.4f} {c.mean():>9.4f}±{c.std(ddof=1):<7.4f}")

    print(f"\n  CAPE vs each comparator under all three controls (n={len(seeds)}):")
    print(f"  {'comparator':<15} {'control':<16} {'diff':>10} {'95% CI':>22} {'p':>9}")
    print("  " + "-" * 76)
    for n in names:
        if n == "CAPE":
            continue
        for key, lab in [("last5", "A full horizon"), ("B", "B equal energy"),
                         ("C", "C equal rounds")]:
            d = agg["CAPE"][key] - agg[n][key]
            m_, s_ = d.mean(), d.std(ddof=1)
            h = stats.t.ppf(0.975, len(d) - 1) * s_ / np.sqrt(len(d))
            _, p_ = stats.ttest_rel(agg["CAPE"][key], agg[n][key])
            print(f"  {n:<15} {lab:<16} {m_:>+10.4f} [{m_-h:>+7.4f},{m_+h:>+7.4f}] "
                  f"{p_:>9.5f}{'*' if p_<0.05 else ''}")

    print(f"\n  Ablation, paired vs DRR-base (n={len(seeds)}), metric = mean of last 5 evals:")
    print(f"  {'arm':<15} {'mean diff':>11} {'95% CI':>22} {'p':>9}")
    print("  " + "-" * 60)
    base = agg["DRR-base"]["last5"]
    tests = []
    for n in names:
        if n == "DRR-base":
            continue
        d = agg[n]["last5"] - base
        m_, s_ = d.mean(), d.std(ddof=1)
        h = stats.t.ppf(0.975, len(d) - 1) * s_ / np.sqrt(len(d))
        _, p_ = stats.ttest_rel(agg[n]["last5"], base)
        print(f"  {n:<15} {m_:>+11.4f} [{m_-h:>+7.4f},{m_+h:>+7.4f}] {p_:>9.5f}{'*' if p_<0.05 else ''}")
        tests.append({"arm": n, "diff": float(m_), "ci": [float(m_-h), float(m_+h)], "p": float(p_)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cape_ablation_{regime}.json").write_text(json.dumps(
        {"regime": regime, "seeds": seeds, "config": cfgd,
         "summary": {n: {"final_mean": float(agg[n]["final"].mean()),
                          "final_sd": float(agg[n]["final"].std(ddof=1)),
                          "last5_mean": float(agg[n]["last5"].mean()),
                          "last5_sd": float(agg[n]["last5"].std(ddof=1))} for n in names},
         "tests": tests, "rows": [{k: v for k, v in r.items() if k != "traj"} for r in rows]},
        indent=1, default=str))
    print(f"\nSaved -> {OUT/f'cape_ablation_{regime}.json'}")


if __name__ == "__main__":
    main()
