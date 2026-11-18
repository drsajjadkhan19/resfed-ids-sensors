#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full identification study for the CAPE mechanisms.

Supersedes the earlier three-arm ablation. It answers four questions that the
earlier design could not separate, under one energy stock, one seed set and one
evaluation:

1.  Does the per-client quantum beat a *tuned uniform* quantum? Adjusting local
    work to trade computation against communication is long-established; the
    claim under test is narrower, namely that deriving each client's quantum
    from its own cost ratio c_k / a_k beats the best single value of w shared by
    the whole federation. Uniform-w arms are therefore run across a grid, with
    and without the same stabilisers CAPE uses, so the comparison isolates the
    per-client rule rather than the amount of local work.

2.  Is the aggregation gain attributable to the *mass* signal, or to server
    damping in general? Constant relaxation, a decaying mass-agnostic schedule,
    server momentum and an adaptive server optimiser are run in place of
    evidence weighting, holding everything else fixed.

3.  What does each of the three mechanisms contribute, and how do they interact?
    All eight cells of the C/E/P factorial are run, so every main effect and
    every interaction is estimated rather than inferred from a chain of
    increments.

4.  How does the comparison depend on the endpoint and on the budget control?
    Every round is evaluated and the full trajectory is retained, so terminal,
    tail-averaged, matched-energy, matched-round, matched-update and
    matched-work readouts are computed from the same runs.

Controls are computed pairwise: matching against the minimum over *all* arms
lets the cheapest arm dictate the comparison point for every pair, which is not
the question any pair is asking. Readouts never overshoot the matched quantity,
and the value actually attained at the readout is recorded alongside it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from src.journal.runtime import workers
import torch

torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig, evaluate
from src.journal.architectures import build, n_params
from src.journal.energy_model import EnergyStock, EnergyModel, DeviceProfile
from src.journal.device_profiles import assign_profiles, DEFAULT_CLASS
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import (DeficitRoundRobin, ThresholdGating,
                                                FedAvgFull, RandomM, Oort,
                                                PowerOfChoice, MinParticipants)
from src.journal.cape import amortized_quantum, EvidenceAggregator, train_local_prox
from src.journal.server_optimizers import (FixedRelaxation, DecayRelaxation,
                                           ServerMomentum, ServerAdam,
                                           MassFloorAggregator, BufferedAggregator,
                                           weighted_average)

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
E_TX = 6.4e-7            # J per byte transmitted
BYTES_PER_PARAM_REPLY = 1   # a probe returns one scalar loss value
UNIFORM_GRID = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------- #
#  Cost model
# --------------------------------------------------------------------------- #
def cost_terms(fed, regime, P, compute_scale=1.0, comms_scale=1.0,
               default_class=DEFAULT_CLASS):
    """Per-client communication cost c_k and per-epoch compute cost a_k.

    ``default_class`` is the hardware tier given to devices whose identity
    matches no explicit rule. A quarter of this corpus falls into that group, so
    the assignment is an assumption rather than an observation and any conclusion
    has to be shown to survive changing it.
    """
    hw = assign_profiles([fed.device_names[d] for d in fed.client_devices],
                         default=default_class)
    n = np.asarray(fed.client_sizes, dtype=float)
    comm = np.full(len(hw), 2.0 * P * 4 * E_TX * comms_scale)
    a = np.array([6.0 * P * ni * h.j_per_flop * compute_scale
                  for ni, h in zip(n, hw)])
    return comm, a, hw


def build_stock(fed, regime, hw, comm, a, w_vec):
    profiles = [DeviceProfile(capacity_j=h.capacity_j * regime.allocation,
                              eta=1.0, label=h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(1, 1), fed.client_sizes)
    stock.cost = comm + a * np.asarray(w_vec, dtype=float)
    return stock


# --------------------------------------------------------------------------- #
#  Arm specification
# --------------------------------------------------------------------------- #
def make_aggregator(kind, fed, cfg):
    """Server rule. ``mass`` is evidence weighting; the rest are controls."""
    sizes = np.asarray(fed.client_sizes, dtype=float)
    m_ref = float(np.sort(sizes)[::-1][:cfg.m].sum())
    return {
        "none": lambda: None,
        "mass": lambda: EvidenceAggregator(m_ref),
        "fixed": lambda: FixedRelaxation(0.5),
        "decay": lambda: DecayRelaxation(1.0, 10.0),
        "momentum": lambda: ServerMomentum(0.9, 1.0),
        "adam": lambda: ServerAdam(0.01),
        "floor25": lambda: MassFloorAggregator(m_ref, floor_frac=0.25),
        "floor50": lambda: MassFloorAggregator(m_ref, floor_frac=0.50),
        "fedbuff": lambda: BufferedAggregator(k=cfg.m),
        "floor50-nostale": lambda: MassFloorAggregator(m_ref, floor_frac=0.50,
                                                       stale_decay=1.0),
    }[kind]()


def arm_list():
    """Every arm, as (name, quantum-rule, aggregator, proximal-strength, policy)."""
    drr = ("drr", None)
    out = []

    # published and standard references
    out.append(dict(name="Threshold-0.3", quantum=("uniform", 2), agg="none",
                    mu0=0.0, policy=("threshold", 0.3)))
    out.append(dict(name="FedAvg-full", quantum=("uniform", 2), agg="none",
                    mu0=0.0, policy=("fedavg", None)))

    # uniform quantum grid under the same pacing policy: the tuned-global-E arm
    for w in UNIFORM_GRID:
        out.append(dict(name=f"DRR-w{w}", quantum=("uniform", w), agg="none",
                        mu0=0.0, policy=drr))
    # the same grid with CAPE's stabilisers, so the per-client rule is the only
    # thing that differs from CAPE at the best uniform setting
    for w in UNIFORM_GRID:
        out.append(dict(name=f"DRR-w{w}+E+P", quantum=("uniform", w), agg="mass",
                        mu0=0.01, policy=drr))

    # C / E / P factorial, all eight cells
    for c, e, p in product([0, 1], repeat=3):
        tag = "".join(t for t, f in (("C", c), ("E", e), ("P", p)) if f) or "base"
        out.append(dict(name=f"F-{tag}",
                        quantum=("amortized", 16) if c else ("uniform", 2),
                        agg="mass" if e else "none",
                        mu0=0.01 if p else 0.0, policy=drr))

    # aggregation identification: hold C and P fixed, vary the server rule
    for kind in ("fixed", "decay", "momentum", "adam"):
        out.append(dict(name=f"C+P+{kind}", quantum=("amortized", 16),
                        agg=kind, mu0=0.01, policy=drr))
    return out


def make_policy(spec, cfg, fed=None, seed=0):
    kind, arg = spec
    if kind == "drr":
        return DeficitRoundRobin("DRR", cfg.m,
                                 scale=arg if arg is not None else 1.0)
    if kind == "threshold":
        return ThresholdGating("T", cfg.m, arg if arg is not None else 0.3)
    if kind == "random":
        return RandomM("R", cfg.m, seed=seed)
    if kind == "oort":
        return Oort("Oort", cfg.m, eps=arg if arg is not None else 0.2, seed=seed)
    if kind == "minparticipants":
        return MinParticipants("MinPart", cfg.m,
                               q_min=int(arg if arg is not None else cfg.m // 2))
    if kind == "powerofchoice":
        sizes = None if fed is None else np.asarray(fed.client_sizes, float)
        return PowerOfChoice("PoC", cfg.m, d_mult=arg if arg is not None else 3.0,
                             sizes=sizes, seed=seed)
    return FedAvgFull("F", cfg.m)




def client_drift(global_sd, updates) -> float:
    """Normalised distance from the global model to the unweighted client mean.

    D_t = || theta_global - mean_k theta_k ||_2 / || theta_global ||_2

    Computed over the participating set of a single round, so it is available
    from the first rounds of a run rather than only in hindsight.
    """
    if not updates:
        return float("nan")
    keys = [q for q in global_sd if global_sd[q].is_floating_point()]
    num, den = 0.0, 0.0
    n = len(updates)
    for q in keys:
        g = global_sd[q].double()
        mean = sum(sd[q].double() for sd, _ in updates) / n
        num += float(((g - mean) ** 2).sum())
        den += float((g ** 2).sum())
    return float(np.sqrt(num) / max(np.sqrt(den), 1e-12))


def local_loss(model, Xb, yb, device, class_weight=None) -> float:
    """Mean cross-entropy of a locally trained model on its own data."""
    import torch.nn as nn
    model.eval()
    w = (torch.tensor(class_weight, dtype=torch.float32, device=device)
         if class_weight is not None else None)
    lossf = nn.CrossEntropyLoss(weight=w)
    with torch.no_grad():
        Xt = torch.tensor(Xb, device=device)
        yt = torch.tensor(yb, dtype=torch.long, device=device)
        total, n = 0.0, 0
        for i in range(0, len(yt), 4096):
            sl = slice(i, i + 4096)
            total += float(lossf(model(Xt[sl]), yt[sl])) * (sl.stop - sl.start
                                                            if sl.stop <= len(yt)
                                                            else len(yt) - sl.start)
            n += min(4096, len(yt) - i)
    model.train()
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
#  One run
# --------------------------------------------------------------------------- #
def run_arm(fed, regime, cfg, arm, seed, cw, compute_scale=1.0, comms_scale=1.0,
            eval_idx=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    P = n_params(build(regime.backbone, fed.n_features, fed.n_classes))
    K = len(fed.client_devices)
    comm, a, hw = cost_terms(fed, regime, P, compute_scale, comms_scale,
                             arm.get("default_class", DEFAULT_CLASS))

    rule, bound = arm["quantum"]
    if rule == "amortized":
        w_vec = amortized_quantum(comm, a, w_min=1, w_max=bound)
    else:
        w_vec = np.full(K, int(bound), dtype=int)

    stock = build_stock(fed, regime, hw, comm, a, w_vec)
    pol = make_policy(arm["policy"], cfg, fed, seed)
    agg = make_aggregator(arm["agg"], fed, cfg)
    w_ref = float(np.median(w_vec))
    mu_vec = arm["mu0"] * (w_vec / max(w_ref, 1e-9))

    model = build(regime.backbone, fed.n_features, fed.n_classes)

    # A policy that probes candidates before choosing must pay for the probe:
    # the global model is sent to each candidate, a forward pass is run over its
    # data, and a scalar is returned. The reply is negligible beside the model
    # but is included so the ledger is complete. The multiplier makes the
    # accounting a swept assumption rather than a fixed one, since a conclusion
    # that depends on it is a conclusion about the accounting.
    if isinstance(pol, PowerOfChoice):
        scale = float(arm.get("probe_scale", 1.0))
        reply = 4.0 * BYTES_PER_PARAM_REPLY * E_TX
        pol.probe_cost = scale * (0.5 * comm + a / 3.0 + reply)

        def _probe(k, _model=model):
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            return local_loss(_model, Xb, yb, device, cw)

        pol.probe_fn = _probe

    Xe, ye = fed.eval_xy() if eval_idx is None else fed.xy(eval_idx)
    traj, last, work = [], 0, 0.0
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        ups, mass = [], 0.0
        gparams = [p.detach().clone() for p in model.parameters()]
        for k in sel:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(regime.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            train_local_prox(loc, Xb, yb, int(w_vec[k]), cfg.lr, cfg.batch_size,
                             device, cw, mu=float(mu_vec[k]), global_params=gparams)
            ups.append((loc.state_dict(), len(yb)))
            mass += len(yb)
            work += len(yb) * int(w_vec[k])
            stock.charge(k)
            # Loss-guided selection policies need feedback from the clients they
            # served; policies that ignore it simply do not define ``update``.
            if hasattr(pol, "update"):
                pol.update(k, local_loss(loc, Xb, yb, device, cw), len(yb), t)
        drift = client_drift(model.state_dict(), ups) if ups else float("nan")
        if ups:
            if agg is not None:
                model.load_state_dict(agg.step(model.state_dict(), ups, mass))
            else:
                avg = weighted_average(ups)
                model.load_state_dict({q: avg[q].to(ups[0][0][q].dtype) for q in avg})
            last = t
        # every round is evaluated so that budget-matched readouts are exact
        last_eval = evaluate(model, Xe, ye, device, fed.n_classes)
        traj.append({"round": t,
                     "macro_f1": last_eval["macro_f1"],
                     "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
                     "updates": int(stock.participations.sum()),
                     "work": float(work),
                     "width": len(sel),
                     "mass": float(mass),
                     "drift": drift,
                     "lam": (float(agg.last_lambda)
                             if agg is not None and getattr(agg, "last_lambda", None)
                             is not None else None)})
    if agg is not None and hasattr(agg, "flush"):
        model.load_state_dict(agg.flush(model.state_dict()))
        last_eval = evaluate(model, Xe, ye, device, fed.n_classes)
        traj.append({"round": cfg.rounds, "macro_f1": last_eval["macro_f1"],
                     "energy_j": traj[-1]["energy_j"], "updates": traj[-1]["updates"],
                     "work": traj[-1]["work"], "width": 0, "mass": 0.0,
                     "drift": float("nan"), "lam": None, "flush": True})

    f1 = [p["macro_f1"] for p in traj]
    early = [p["drift"] for p in traj[1:5] if p["drift"] == p["drift"]]
    return {"arm": arm["name"], "seed": seed,
            "drift_early": (float(np.mean(early)) if early else float("nan")),
            "final_macro_f1": f1[-1],
            "mean_last5": float(np.mean(f1[-5:])),
            "best_macro_f1": float(np.max(f1)),
            "energy_j": traj[-1]["energy_j"],
            "updates": traj[-1]["updates"],
            "work": traj[-1]["work"],
            "last_active_round": last,
            "w_median": float(np.median(w_vec)),
            "w_mean": float(np.mean(w_vec)),
            "stranded": float(stock.stranded_fraction()),
            "min_width": int(min(p["width"] for p in traj)),
            "probe_scale": float(arm.get("probe_scale", 1.0)),
            "probe_energy_j": float(getattr(pol, "probe_charged", 0.0)),
            "probes": int(getattr(pol, "probes", 0)),
            "per_class_recall": last_eval["per_class_recall"],
            "per_class_precision": last_eval["per_class_precision"],
            "per_class_f1": last_eval["per_class_f1"],
            "traj": traj}


# --------------------------------------------------------------------------- #
#  Pairwise budget-matched readouts
# --------------------------------------------------------------------------- #
def at_budget(traj, key, cap):
    """Last logged point whose ``key`` is at or below ``cap`` (never overshoots)."""
    ok = [p for p in traj if p[key] <= cap + 1e-9]
    p = ok[-1] if ok else traj[0]
    return p["macro_f1"], p[key], p["round"]


def paired_stats(x, y):
    from scipy import stats
    d = np.asarray(x) - np.asarray(y)
    m, n = float(d.mean()), len(d)
    h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    p = float(stats.ttest_rel(x, y).pvalue) if n > 1 else float("nan")
    return m, m - h, m + h, p


def job(args):
    seed, regime_name, cfgd, cscale, tscale = args
    cfg = FedConfig(**cfgd)
    cfg.seed = seed
    fed = Federation(cfg)
    y = np.concatenate([fed.y[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    cw = 1.0 / np.maximum(f, 1e-9)
    cw = cw / cw.mean()
    rg = REGIMES[regime_name]
    return [run_arm(fed, rg, cfg, a, seed, cw, cscale, tscale) for a in arm_list()]


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "R2f-bind"
    cscale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    tscale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tag = "" if (cscale, tscale) == (1.0, 1.0) else f"_c{cscale:g}_t{tscale:g}"
    seeds = list(range(1, 11))
    cfgd = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    names = [a["name"] for a in arm_list()]
    print(f"CAPE factorial | regime={regime} seeds={len(seeds)} arms={len(names)} "
          f"compute_scale={cscale} comms_scale={tscale}", flush=True)

    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers()) as pool:
        chunks = pool.map(job, [(s, regime, cfgd, cscale, tscale) for s in seeds])
    rows = [r for c in chunks for r in c]
    print(f"  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    idx = {(r["arm"], r["seed"]): r for r in rows}

    def vec(arm, key):
        return np.array([idx[(arm, s)][key] for s in seeds])

    print(f"  {'arm':<16}{'terminal':>17}{'tail-5':>17}{'J':>8}{'upd':>6}"
          f"{'work/1e3':>10}{'w~':>5}{'minw':>6}")
    print("  " + "-" * 86)
    for n in names:
        f_, l_ = vec(n, "final_macro_f1"), vec(n, "mean_last5")
        print(f"  {n:<16}{f_.mean():>9.4f}±{f_.std(ddof=1):<7.4f}"
              f"{l_.mean():>9.4f}±{l_.std(ddof=1):<7.4f}"
              f"{vec(n,'energy_j').mean():>8.1f}{vec(n,'updates').mean():>6.0f}"
              f"{vec(n,'work').mean()/1e3:>10.0f}{np.median(vec(n,'w_median')):>5.0f}"
              f"{vec(n,'min_width').min():>6.0f}")

    # ---- pairwise budget-matched comparisons against CAPE -------------------
    cape = "F-CEP"
    controls = [("terminal", None), ("tail5", None),
                ("energy_j", "matched energy"), ("updates", "matched updates"),
                ("work", "matched work"), ("round", "matched rounds")]
    comparisons = []
    print(f"\n  {cape} vs each arm, paired over {len(seeds)} seeds, "
          f"pairwise budget matching")
    print(f"  {'comparator':<16}{'control':<16}{'diff':>10}{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 74)
    for n in names:
        if n == cape:
            continue
        for key, lab in controls:
            if key == "terminal":
                x, y = vec(cape, "final_macro_f1"), vec(n, "final_macro_f1")
            elif key == "tail5":
                x, y = vec(cape, "mean_last5"), vec(n, "mean_last5")
            else:
                xs, ys = [], []
                for s in seeds:
                    ra, rb = idx[(cape, s)], idx[(n, s)]
                    if key == "round":
                        cap = min(ra["last_active_round"], rb["last_active_round"])
                        xs.append(at_budget(ra["traj"], "round", cap)[0])
                        ys.append(at_budget(rb["traj"], "round", cap)[0])
                    else:
                        cap = min(ra[key], rb[key])
                        xs.append(at_budget(ra["traj"], key, cap)[0])
                        ys.append(at_budget(rb["traj"], key, cap)[0])
                x, y = np.array(xs), np.array(ys)
            m, lo, hi, p = paired_stats(x, y)
            comparisons.append({"comparator": n, "control": key, "diff": m,
                                "ci": [lo, hi], "p": p,
                                "cape_mean": float(x.mean()), "other_mean": float(y.mean())})
            star = "*" if p < 0.05 else " "
            print(f"  {n:<16}{lab or key:<16}{m:>+10.4f} [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"{p:>10.5f}{star}")

    # ---- C/E/P factorial effects -------------------------------------------
    print(f"\n  C/E/P factorial effects (paired, n={len(seeds)})")
    print(f"  {'effect':<26}{'endpoint':<12}{'estimate':>10}{'95% CI':>21}{'p':>10}")
    print("  " + "-" * 80)
    cell = {(c, e, p): "F-" + ("".join(t for t, f in (("C", c), ("E", e), ("P", p)) if f) or "base")
            for c, e, p in product([0, 1], repeat=3)}
    effects = []
    for endpoint, key in [("terminal", "final_macro_f1"), ("tail-5", "mean_last5")]:
        V = {k: vec(v, key) for k, v in cell.items()}
        defs = {
            "main C": (lambda C: np.mean([V[(1, e, p)] - V[(0, e, p)]
                                          for e, p in product([0, 1], repeat=2)], axis=0)),
            "main E": (lambda C: np.mean([V[(c, 1, p)] - V[(c, 0, p)]
                                          for c, p in product([0, 1], repeat=2)], axis=0)),
            "main P": (lambda C: np.mean([V[(c, e, 1)] - V[(c, e, 0)]
                                          for c, e in product([0, 1], repeat=2)], axis=0)),
            "interaction C x E": (lambda C: np.mean([V[(1, 1, p)] - V[(1, 0, p)]
                                                     - V[(0, 1, p)] + V[(0, 0, p)]
                                                     for p in [0, 1]], axis=0)),
            "interaction C x P": (lambda C: np.mean([V[(1, e, 1)] - V[(1, e, 0)]
                                                     - V[(0, e, 1)] + V[(0, e, 0)]
                                                     for e in [0, 1]], axis=0)),
            "interaction E x P": (lambda C: np.mean([V[(c, 1, 1)] - V[(c, 1, 0)]
                                                     - V[(c, 0, 1)] + V[(c, 0, 0)]
                                                     for c in [0, 1]], axis=0)),
        }
        for lab, fn in defs.items():
            d = fn(None)
            m, lo, hi, p = paired_stats(d, np.zeros_like(d))
            effects.append({"effect": lab, "endpoint": endpoint, "estimate": m,
                            "ci": [lo, hi], "p": p})
            print(f"  {lab:<26}{endpoint:<12}{m:>+10.4f} [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"{p:>10.5f}{'*' if p < 0.05 else ' '}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cape_factorial_{regime}{tag}.json"
    path.write_text(json.dumps(
        {"regime": regime, "seeds": seeds, "config": cfgd,
         "compute_scale": cscale, "comms_scale": tscale,
         "comparisons": comparisons, "effects": effects,
         "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
