"""Summaries of the revision studies (results/revision), used by analysis/stats.py."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

COLLAPSE = 0.10


def ms(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return dict(mean=float(a.mean()) if len(a) else None, n=int(len(a)))
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), ci=[float(a.mean() - h), float(a.mean() + h)],
                n=int(len(a)))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    try:
        w = stats.wilcoxon(a, b)
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    return dict(diff=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)], wilcoxon_p=wp,
                a_ahead=int((d > 0).sum()), n=int(len(d)))


def fisher(c1, n1, c2, n2):
    return float(stats.fisher_exact([[c1, n1 - c1], [c2, n2 - c2]])[1])


def by(rows, *keys):
    g = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["seed"]):
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def load(rev: Path, name: str):
    p = rev / name
    return json.loads(p.read_text()) if p.exists() else None


def energy_matched(ra, rb):
    """Paired difference in macro-F1 of the models reached at equal energy.

    For each split the energy level is the smaller of the two runs' final
    spends; each run contributes the score of its last round within that level.
    """
    def at(r, e):
        ok = [p["macro_f1"] for p in r["traj"] if p["energy_j"] <= e + 1e-9]
        return ok[-1] if ok else 0.0
    out = []
    for a, b in zip(ra, rb):
        e = min(a["traj"][-1]["energy_j"], b["traj"][-1]["energy_j"])
        out.append(at(a, e) - at(b, e))
    return ms(out)


def summarise(rev: Path) -> dict:
    S = {}
    d = load(rev, "policies_full_and_no_sbc_R2f-bind.json")
    if d:
        g = by(d["rows"], "federation", "arm")
        S["policies"] = {}
        for (fed, arm), rr in g.items():
            t = [r["terminal"] for r in rr]
            pcf1 = np.array([r["per_class_f1"] for r in rr])
            tiers = defaultdict(list)
            for r in rr:
                for tier, v in r["tiers"].items():
                    tiers[tier].append(v["attack_recall"])
            S["policies"][f"{fed}|{arm}"] = dict(
                terminal=ms(t), accuracy=ms([r["accuracy"] for r in rr]), tail5=ms([r["tail5"] for r in rr]),
                collapsed=int(sum(x < COLLAPSE for x in t)), energy_j=ms([r["energy_j"] for r in rr]),
                updates=ms([r["updates"] for r in rr]), unspent=ms([r["unspent"] for r in rr]),
                per_class_f1=dict(zip(rr[0]["classes"], pcf1.mean(0).round(4).tolist())),
                tier_attack_recall={k: ms(v) for k, v in tiers.items()})
        S["policies_tests"] = {}
        for fed in ("full", "no-sbc"):
            T = {arm: g[(fed, arm)] for (f, arm) in g if f == fed}
            for a, b in [("DRR", "Published gate"), ("DRR", "FedAvg-full"), ("Gate, deficit order", "Published gate"),
                         ("Gate, random order", "Published gate"), ("Gate, deficit per cost", "Published gate"),
                         ("Gate, deficit order", "Gate, random order")]:
                if a in T and b in T:
                    ta = [r["terminal"] for r in T[a]]
                    tb = [r["terminal"] for r in T[b]]
                    ca, cb = sum(x < COLLAPSE for x in ta), sum(x < COLLAPSE for x in tb)
                    S["policies_tests"][f"{fed}|{a} vs {b}"] = dict(
                        terminal=paired(ta, tb), fisher_collapse=fisher(ca, len(ta), cb, len(tb)),
                        energy_matched=energy_matched(T[a], T[b]))
    d = load(rev, "quantum_full_and_no_sbc_R2f-bind.json")
    if d:
        g = by(d["rows"], "federation", "w")
        S["quantum"] = {f"{fed}|w={w}": dict(terminal=ms([r["terminal"] for r in rr]),
                                              tail5=ms([r["tail5"] for r in rr]),
                                              collapsed=int(sum(r["terminal"] < COLLAPSE for r in rr)),
                                              last_width_median=float(np.median([r["last_width"] for r in rr])),
                                              energy_j=ms([r["energy_j"] for r in rr]))
                        for (fed, w), rr in sorted(g.items())}
        S["quantum_terminal_minus_tail5"] = {
            f"{fed}|w={w}": paired([r["terminal"] for r in rr], [r["tail5"] for r in rr])
            for (fed, w), rr in sorted(g.items())}
    d = load(rev, "margin_meter_R2f-bind.json")
    if d:
        g = by(d["rows"], "sweep", "kappa", "delta", "check")
        S["margin_meter"] = {f"{s}|k={k}|d={dl}|check={c}": dict(
            overdrawn=ms([r["overdrawn"] for r in rr]), refusals=ms([r["refusals"] for r in rr]),
            refused_at_full_charge=ms([r["refused_at_full_charge"] for r in rr]),
            min_fraction=float(min(r["min_fraction"] for r in rr)),
            bound_violations=int(sum(r["bound_violations"] or 0 for r in rr)),
            participations=ms([r["participations"] for r in rr]))
            for (s, k, dl, c), rr in sorted(g.items(), key=lambda x: str(x[0]))}
    d = load(rev, "twolevel_margin_R2f-bind.json")
    if d:
        g = by(d["rows"], "kappa", "arm")
        S["twolevel_margin"] = {f"k={k}|{a}": dict(terminal=ms([r["terminal"] for r in rr]),
                                                   overdrawn=ms([r["overdrawn"] for r in rr]),
                                                   refusals=ms([r["refusals"] for r in rr]),
                                                   participations=ms([r["participations"] for r in rr]),
                                                   collapsed=int(sum(r["terminal"] < COLLAPSE for r in rr)))
                                for (k, a), rr in sorted(g.items(), key=lambda x: str(x[0]))}
    d = load(rev, "allocation_sweep_R2f-bind.json")
    if d:
        rows = d["rows"]
        keys = [k for k in ("kind", "allocation", "kappa", "design", "arm") if any(k in r for r in rows)]
        g = by(rows, *keys)
        S["allocation"] = {"|".join(f"{k}={v}" for k, v in zip(keys, kk)): {
            m: ms([r[m] for r in rr]) for m in ("overdrawn", "participations", "terminal", "energy_j")
            if all(m in r for r in rr)} for kk, rr in sorted(g.items(), key=lambda x: str(x[0]))}
    d = load(rev, "small_model_mcu_R2f-bind.json")
    if d:
        g = by(d["rows"], "federation", "backbone", "arm")
        S["small_model"] = {f"{f}|{b}|{a}": dict(terminal=ms([r["terminal"] for r in rr]),
                                                  n_params=rr[0]["n_params"],
                                                  mcu_participating=ms([r["mcu_participating"] for r in rr]),
                                                  energy_j=ms([r["energy_j"] for r in rr]))
                            for (f, b, a), rr in sorted(g.items())}
    d = load(rev, "transfer_grid_ciciot2023.json")
    if d:
        g = by(d["rows"], "arm")
        S["transfer_grid"] = {a: dict(final=ms([r["final_macro_f1"] for r in rr]),
                                      last5=ms([r["mean_last5"] for r in rr]),
                                      energy_j=ms([r["energy_j"] for r in rr]))
                              for (a,), rr in sorted(g.items())}
    return S
