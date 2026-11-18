#!/usr/bin/env python3
"""
Comparison of participation policies under three pre-declared controls.

Comparing policies under a finite energy budget admits several distinct notions
of a fair comparison, and they do not agree. A policy that paces its spending
may finish the horizon still training while a greedier policy has exhausted
itself, so equal initial stock, equal consumed energy and equal round count each
answer a different question. All three are therefore declared in advance and
reported side by side.

Controls, all read off a single trajectory per policy and seed so that none can
be selected after an outcome is known:

  A  matched initial stock   -- what a policy extracts from its own budget over
                               the full horizon
  B  matched consumed energy -- accuracy at the point where every policy has
                               spent the same number of joules
  C  matched rounds          -- accuracy at the last round reached by every
                               policy, which removes any advantage arising from
                               simply remaining active longer

Token-bucket and deficit-round-robin baselines are included because a cumulative
per-client service curve is a standard rate-control mechanism and must be
measured against the established primitives it resembles.

Each policy is run once per seed for the full horizon while logging
(round, cumulative joules, macro-F1). The three controls are then read off the
same trajectory:

  A. matched initial stock  -- value at the end of the horizon
  B. matched consumed energy -- value at the first round where cumulative
                                energy >= E*, E* = min over policies of their
                                total consumption in that seed
  C. matched rounds          -- value at round R*, R* = min over policies of
                                their last active round in that seed
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch

torch.set_num_threads(1)          # one thread per worker process

from src.journal.federation import Federation, FedConfig, _train_local, evaluate
from src.journal.architectures import build
from src.journal.study_threshold_characterisation import make_stock
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import (ThresholdGating, RandomM, FedAvgFull, LACS,
                                    TokenBucket, DeficitRoundRobin, MinParticipants, MinSelected, GatedDeficit)

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
EVAL_EVERY = 1   # every round, so budget-matched readouts are exact


def policies(cfg, seed):
    return [
        ("FedAvg-full",    FedAvgFull("FedAvg-full", cfg.m)),
        ("Threshold-0.3",  ThresholdGating("Threshold-0.3", cfg.m, 0.3)),
        ("Threshold-0.0",  ThresholdGating("Threshold-0.0", cfg.m, 0.0)),
        ("Random-m",       RandomM("Random-m", cfg.m, seed=seed)),
        ("TokenBucket",    TokenBucket("TokenBucket", cfg.m, burst_rounds=2.0)),
        ("DRR",            DeficitRoundRobin("DRR", cfg.m)),
        ("LACS-pace",      LACS("LACS-pace", cfg.m, use_coverage=False)),
        # The published policy with the one addition the thin-round diagnosis
        # implies: refuse a round that too few clients can afford, rather than
        # aggregating it. Framework defaults already decline rounds below two
        # participants; the published implementation set that bound to one.
        ("Threshold-0.3+minwidth",
         MinParticipants("Threshold-0.3+minwidth", cfg.m, q_min=3,
                         inner=ThresholdGating("thr-inner", cfg.m, 0.3))),
        ("FedAvg-full+minwidth",
         MinParticipants("FedAvg-full+minwidth", cfg.m, q_min=3,
                         inner=FedAvgFull("full-inner", cfg.m))),
        ("DRR+minwidth",
         MinParticipants("DRR+minwidth", cfg.m, q_min=3,
                         inner=DeficitRoundRobin("drr-inner", cfg.m))),
        # Guarding the selected set rather than affordability, which is what a
        # gating rule needs: it can return a thin round while many clients
        # remain affordable but below the charge threshold.
        ("Threshold-0.3+minsel",
         MinSelected("Threshold-0.3+minsel", cfg.m, q_min=3,
                     inner=ThresholdGating("thr-inner2", cfg.m, 0.3))),
        # The published gate, keeping its depletion guarantee, with the order
        # replaced by deficit service. Everything else is identical.
        ("Threshold-0.3+deficit-order",
         GatedDeficit("Threshold-0.3+deficit-order", cfg.m, 0.3)),
        ("FedAvg-full+minsel",
         MinSelected("FedAvg-full+minsel", cfg.m, q_min=3,
                     inner=FedAvgFull("full-inner2", cfg.m))),
    ]


def class_weights(fed):
    y = np.concatenate([fed.y[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    w = 1.0 / np.maximum(f, 1e-9)
    return w / w.mean()


def trajectory(fed, regime, cfg, policy, seed, cw):
    """Run one policy for the full horizon, logging energy and accuracy."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    stock, hw, P = make_stock(fed, regime, cfg)
    model = build(regime.backbone, fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy()
    traj, last_active = [], 0
    for t in range(1, cfg.rounds + 1):
        sel = policy.select(t, cfg.rounds, stock, None)
        ups = []
        for k in sel:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(regime.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            _train_local(loc, Xb, yb, cfg, dev, cw)
            ups.append((loc.state_dict(), len(yb)))
            stock.charge(k)
        if ups:
            tot = sum(n for _, n in ups)
            model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                   .to(ups[0][0][q].dtype) for q in ups[0][0]})
            last_active = t
        if t % EVAL_EVERY == 0 or t == cfg.rounds:
            traj.append({
                "round": t,
                "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
                "updates": int(stock.participations.sum()),
                "width": len(sel),
                "macro_f1": evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"],
            })
    return {"policy": policy.name, "seed": seed, "traj": traj,
            "last_active_round": last_active,
            "total_energy_j": float(stock.initial.sum() - stock.remaining.sum()),
            "total_updates": int(stock.participations.sum()),
            "stranded_fraction": float(stock.stranded_fraction())}


def run_seed(args):
    seed, regime_name, cfgd = args
    cfg = FedConfig(**cfgd)
    cfg.seed = seed                       # M2: hold-out resampled per seed
    fed = Federation(cfg)
    cw = class_weights(fed)
    rg = REGIMES[regime_name]
    out = []
    for _, pol in policies(cfg, seed):
        out.append(trajectory(fed, rg, cfg, pol, seed, cw))
    return seed, out


def at_energy(traj, target):
    for p in traj:
        if p["energy_j"] >= target - 1e-9:
            return p["macro_f1"]
    return traj[-1]["macro_f1"]


def at_round(traj, r):
    ok = [p for p in traj if p["round"] <= r]
    return (ok[-1] if ok else traj[0])["macro_f1"]


def main():
    seeds = list(range(1, 11))            # C5: n = 10
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    print(f"policy comparison | regime={regime} seeds={len(seeds)} cfg={cfgd}",
          flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        results = pool.map(run_seed, [(s, regime, cfgd) for s in seeds])
    print(f"  completed in {(time.time()-t0)/60:.1f} min on {workers()} workers\n", flush=True)

    per_seed = {}
    for seed, runs in results:
        # A policy that never trains would otherwise set E*=R*=0 and collapse
        # both controls to the first trajectory point for every arm.
        ran = [r for r in runs if r["last_active_round"] > 0 and r["total_energy_j"] > 0]
        dead = [r["policy"] for r in runs if r not in ran]
        if dead:
            print(f"  [seed {seed}] policies that never trained, excluded from "
                  f"E*/R*: {dead}", flush=True)
        Estar = min(r["total_energy_j"] for r in ran)
        Rstar = min(r["last_active_round"] for r in ran)
        for r in runs:
            per_seed.setdefault(r["policy"], []).append({
                "seed": seed,
                "A_final": r["traj"][-1]["macro_f1"],
                "B_matched_energy": at_energy(r["traj"], Estar),
                "C_matched_rounds": at_round(r["traj"], Rstar),
                "energy_j": r["total_energy_j"], "updates": r["total_updates"],
                "stranded": r["stranded_fraction"],
                "last_active_round": r["last_active_round"],
                "E_star": Estar, "R_star": Rstar,
            })

    from scipy import stats
    names = list(per_seed)
    print(f"  {'policy':<15} {'A final':>16} {'B @equal energy':>18} {'C @equal rounds':>18} {'J':>7} {'upd':>5}")
    print("  " + "-" * 86)
    summary = []
    for n in names:
        rs = per_seed[n]
        a = np.array([r["A_final"] for r in rs]); b = np.array([r["B_matched_energy"] for r in rs])
        c = np.array([r["C_matched_rounds"] for r in rs])
        print(f"  {n:<15} {a.mean():>8.4f}±{a.std(ddof=1):<7.4f} "
              f"{b.mean():>9.4f}±{b.std(ddof=1):<7.4f} {c.mean():>9.4f}±{c.std(ddof=1):<7.4f} "
              f"{np.mean([r['energy_j'] for r in rs]):>7.1f} "
              f"{np.mean([r['updates'] for r in rs]):>5.0f}")
        summary.append({"policy": n, "A_mean": float(a.mean()), "A_sd": float(a.std(ddof=1)),
                        "B_mean": float(b.mean()), "B_sd": float(b.std(ddof=1)),
                        "C_mean": float(c.mean()), "C_sd": float(c.std(ddof=1)),
                        "energy_j": float(np.mean([r["energy_j"] for r in rs])),
                        "updates": float(np.mean([r["updates"] for r in rs])),
                        "stranded": float(np.mean([r["stranded"] for r in rs]))})

    print("\n  Paired tests, LACS-pace vs each comparator (n=10):")
    print(f"  {'comparator':<15} {'control':<16} {'mean diff':>10} {'95% CI':>22} {'p':>8}")
    print("  " + "-" * 78)
    tests = []
    L = {k: np.array([r[k] for r in per_seed["LACS-pace"]])
         for k in ("A_final", "B_matched_energy", "C_matched_rounds")}
    for n in names:
        if n == "LACS-pace":
            continue
        for key, label in [("A_final", "A initial stock"),
                           ("B_matched_energy", "B equal energy"),
                           ("C_matched_rounds", "C equal rounds")]:
            o = np.array([r[key] for r in per_seed[n]])
            d = L[key] - o
            m_, s_ = d.mean(), d.std(ddof=1)
            h = stats.t.ppf(0.975, len(d) - 1) * s_ / np.sqrt(len(d))
            t_, p_ = stats.ttest_rel(L[key], o)
            star = "*" if p_ < 0.05 else " "
            print(f"  {n:<15} {label:<16} {m_:>+10.4f} [{m_-h:>+7.4f},{m_+h:>+7.4f}] {p_:>8.4f}{star}")
            tests.append({"comparator": n, "control": label, "mean_diff": float(m_),
                          "ci_lo": float(m_-h), "ci_hi": float(m_+h), "p": float(p_)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"policy_comparison_{regime}.json").write_text(json.dumps(
        {"regime": regime, "config": cfgd, "seeds": seeds,
         "summary": summary, "paired_tests": tests, "per_seed": per_seed},
        indent=1, default=str))
    print(f"\nSaved -> {OUT/f'policy_comparison_{regime}.json'}")


if __name__ == "__main__":
    main()
