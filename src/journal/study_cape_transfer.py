#!/usr/bin/env python3
"""Transfer of the CAPE mechanisms to CICIoT2023.

A continuity anchor with the published corpus, and a stress test of the
mechanisms against synthetic Dirichlet heterogeneity rather than natural device
partitions. It answers a narrower question than the primary study: whether the
mechanisms survive a change of corpus, not whether the device-partition
interaction generalises.

Every round is evaluated and the trajectory retained, so the energy-matched
comparison is exact rather than inferred from a ratio of endpoint energies. A
macro-F1-per-joule column is reported alongside, since a method that wins on
score while spending more energy has not established an efficiency claim.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)

from src.journal.preprocessing import load_clean
from src.journal.partitioning import dirichlet_partition
from src.journal.device_profiles import HW_CLASSES
from src.journal.architectures import build, n_params
from src.journal.energy_model import EnergyStock, DeviceProfile, EnergyModel
from src.journal.federation import FedConfig, evaluate
from src.journal.participation_policies import DeficitRoundRobin, ThresholdGating, FedAvgFull
from src.journal.cape import amortized_quantum, EvidenceAggregator, train_local_prox

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "journal"
E_TX, ALLOC = 6.4e-7, 4.0e-5


def federation(alpha, K, seed, max_rows, hw_mode="cyclic"):
    d = load_clean(ROOT / "data" / "clean_ciciot2023")
    parts = dirichlet_partition(d["y_train"], K, alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed)
    parts = [np.asarray(p if len(p) <= max_rows else rng.choice(p, max_rows, replace=False),
                        dtype=np.int64) for p in parts]
    keep = [i for i, p in enumerate(parts) if len(p) >= 8]
    parts = [parts[i] for i in keep]
    hw = assign_hardware(len(keep), hw_mode, seed)
    return d, parts, hw


# Empirical hardware-tier frequencies of the CIC IoT-DIAD 2024 device population,
# used when the transfer study is asked to reproduce a realistic stock
# distribution rather than a uniform one.
DIAD_TIER_FREQUENCIES = {"camera": 28, "mcu": 17, "sbc": 11, "av": 8,
                         "gateway": 4, "appliance": 4}


def assign_hardware(n_clients, mode, seed):
    """Hardware tiers for synthetic clients.

    ``cyclic`` gives every tier an equal share, which is even-handed but produces
    a stock distribution no real device population has: clients deplete together
    and a run ends with a full round. ``diad`` samples the tier frequencies
    measured on the natural device corpus, which is skewed, so a few clients
    outlast the rest. Which of the two is used decides whether the federation can
    end on a thin round at all, and that is the object under test.
    """
    names = list(HW_CLASSES)
    if mode == "cyclic":
        return [HW_CLASSES[names[i % len(names)]] for i in range(n_clients)]
    if mode == "diad":
        tiers = list(DIAD_TIER_FREQUENCIES)
        weights = np.array([DIAD_TIER_FREQUENCIES[t] for t in tiers], dtype=float)
        rng = np.random.default_rng(seed + 9973)
        drawn = rng.choice(tiers, size=n_clients, p=weights / weights.sum())
        return [HW_CLASSES[t] for t in drawn]
    raise ValueError(f"unknown hardware assignment mode: {mode}")


def run(alpha, seed, arm, cfg, K, hw_mode="cyclic"):
    d, parts, hw = federation(alpha, K, seed, cfg.max_rows_per_client, hw_mode)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    n_cls, d_in = len(d["class_names"]), Xtr.shape[1]
    torch.manual_seed(seed); np.random.seed(seed)
    P = n_params(build("mlp", d_in, n_cls)); Kk = len(parts)
    sizes = np.array([len(p) for p in parts], float)
    comm = np.full(Kk, 2.0 * P * 4 * E_TX)
    a = np.array([6.0 * P * s * h.j_per_flop for s, h in zip(sizes, hw)])
    w_vec = (amortized_quantum(comm, a, 1, 16) if arm["C"]
             else np.full(Kk, int(arm.get("w", cfg.local_epochs)), dtype=int))
    profiles = [DeviceProfile(h.capacity_j * ALLOC, 1.0, h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(P, 1), list(sizes.astype(int)))
    stock.cost = comm + a * w_vec

    freq = np.bincount(ytr, minlength=n_cls) / len(ytr)
    cw = 1.0 / np.maximum(freq, 1e-9); cw = cw / cw.mean()
    pol = arm["policy"](cfg)
    m_ref = float(np.sort(sizes)[::-1][:cfg.m].sum())
    agg = EvidenceAggregator(m_ref) if arm["E"] else None
    mu = (arm["mu0"] * (w_vec / max(np.median(w_vec), 1e-9))) if arm["P"] else np.zeros(Kk)

    model = build("mlp", d_in, n_cls); dev = torch.device("cpu"); traj = []
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        ups, mass = [], 0.0
        gp = [p.detach().clone() for p in model.parameters()]
        for k in sel:
            idx = parts[k]
            loc = build("mlp", d_in, n_cls); loc.load_state_dict(model.state_dict())
            train_local_prox(loc, Xtr[idx], ytr[idx], int(w_vec[k]), cfg.lr,
                             cfg.batch_size, dev, cw, float(mu[k]), gp)
            ups.append((loc.state_dict(), len(idx))); mass += len(idx); stock.charge(k)
        if ups:
            if agg is not None:
                model.load_state_dict(agg.step(model.state_dict(), ups, mass))
            else:
                tot = sum(n for _, n in ups)
                model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                       .to(ups[0][0][q].dtype) for q in ups[0][0]})
        traj.append({"round": t,
                     "macro_f1": evaluate(model, Xte, yte, dev, n_cls)["macro_f1"],
                     "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
                     "updates": int(stock.participations.sum()),
                     "width": len(sel)})
    f1 = [q["macro_f1"] for q in traj]
    return {"arm": arm["name"], "alpha": alpha, "seed": seed,
            "final": f1[-1], "last5": float(np.mean(f1[-5:])),
            "energy_j": traj[-1]["energy_j"],
            "updates": traj[-1]["updates"], "K": Kk, "traj": traj}


ARMS = [
    {"name": "Threshold-0.3", "policy": lambda c: ThresholdGating("T", c.m, 0.3),
     "C": False, "E": False, "P": False, "mu0": 0.0},
    {"name": "FedAvg-full", "policy": lambda c: FedAvgFull("F", c.m),
     "C": False, "E": False, "P": False, "mu0": 0.0},
    {"name": "DRR-base", "policy": lambda c: DeficitRoundRobin("D", c.m),
     "C": False, "E": False, "P": False, "mu0": 0.0},
    {"name": "CAPE", "policy": lambda c: DeficitRoundRobin("D", c.m),
     "C": True, "E": True, "P": True, "mu0": 0.01},
    # tuned uniform quantum with the same stabilisers: isolates the per-client
    # rule from the amount of local work it happens to prescribe
    {"name": "DRR-w8+E+P", "policy": lambda c: DeficitRoundRobin("D", c.m),
     "C": False, "w": 8, "E": True, "P": True, "mu0": 0.01},
    {"name": "DRR-w16+E+P", "policy": lambda c: DeficitRoundRobin("D", c.m),
     "C": False, "w": 16, "E": True, "P": True, "mu0": 0.01},
]


def job(args):
    alpha, seed, cfgd, K, hw_mode = args
    cfg = FedConfig(**cfgd)
    return [run(alpha, seed, a, cfg, K, hw_mode) for a in ARMS]


def main():
    hw_mode = sys.argv[1] if len(sys.argv) > 1 else "cyclic"
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    K, seeds, alphas = 72, list(range(1, 11)), [0.1, 0.5, 5.0]
    print(f"transfer to CICIoT2023 | K={K} seeds={len(seeds)} alphas={alphas} "
          f"hardware={hw_mode}\n", flush=True)
    from multiprocessing import Pool
    jobs = [(al, s, cfgd, K, hw_mode) for al in alphas for s in seeds]
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, jobs)
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    from scipy import stats
    names = [a["name"] for a in ARMS]
    idx = {(r["alpha"], r["arm"], r["seed"]): r for r in rows}

    def at_energy(traj, cap):
        ok = [q for q in traj if q["energy_j"] <= cap + 1e-9]
        return (ok[-1] if ok else traj[0])["macro_f1"]

    comparisons = []
    for al in alphas:
        print(f"### Dirichlet alpha={al}")
        print(f"    {'arm':<15}{'terminal':>17}{'tail-5':>17}{'J':>8}{'F1/J x1e3':>11}")
        vals, effs = {}, {}
        for n in names:
            fin = np.array([idx[(al, n, s)]["final"] for s in seeds])
            l5 = np.array([idx[(al, n, s)]["last5"] for s in seeds])
            J = np.array([idx[(al, n, s)]["energy_j"] for s in seeds])
            vals[n], effs[n] = (fin, l5, J), fin / J
            print(f"    {n:<15}{fin.mean():>9.4f}±{fin.std(ddof=1):<7.4f}"
                  f"{l5.mean():>9.4f}±{l5.std(ddof=1):<7.4f}{J.mean():>8.1f}"
                  f"{1000*effs[n].mean():>11.4f}")
        for n in names:
            if n == "CAPE":
                continue
            # exact pairwise equal-energy readout, never overshooting the budget
            eq_a, eq_b = [], []
            for s in seeds:
                ra, rb = idx[(al, "CAPE", s)], idx[(al, n, s)]
                cap = min(ra["energy_j"], rb["energy_j"])
                eq_a.append(at_energy(ra["traj"], cap))
                eq_b.append(at_energy(rb["traj"], cap))
            for lab, x, y in [("terminal", vals["CAPE"][0], vals[n][0]),
                              ("tail-5", vals["CAPE"][1], vals[n][1]),
                              ("equal energy", np.array(eq_a), np.array(eq_b)),
                              ("F1 per joule", effs["CAPE"], effs[n])]:
                d = np.asarray(x) - np.asarray(y)
                p = float(stats.ttest_rel(x, y).pvalue)
                comparisons.append({"alpha": al, "comparator": n, "control": lab,
                                    "diff": float(d.mean()), "p": p})
                print(f"      CAPE vs {n:<15}{lab:<15}{d.mean():>+10.4f}  "
                      f"p={p:.5f}{'*' if p < 0.05 else ''}")
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cape_transfer_{hw_mode}.json").write_text(json.dumps(
        {"config": cfgd, "K": K, "seeds": seeds, "alphas": alphas,
         "hardware_mode": hw_mode, "comparisons": comparisons, "rows": rows},
        indent=1, default=str))
    print(f"Saved -> {OUT/f'cape_transfer_{hw_mode}.json'}")


if __name__ == "__main__":
    main()
