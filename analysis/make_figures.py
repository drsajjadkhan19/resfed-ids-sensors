#!/usr/bin/env python3
"""Draw the results figures of the article from the saved run outputs.

    python analysis/make_figures.py            writes figures/*.eps and *.pdf

Every figure is a line graph: the line is the mean over the ten device-disjoint
splits and the band is the 95% t-interval of that mean. The script reads only
files under results/revision and draws nothing it cannot read there.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "revision"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.3, "ps.fonttype": 42, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = {"DRR": "#1f5fa8", "Published gate": "#b2442a", "FedAvg-full": "#555555",
       "Gate, deficit order": "#2e8b57", "Gate, random order": "#c08a00",
       "Gate, deficit per cost": "#7a4fa0", "full": "#1f5fa8", "no-sbc": "#b2442a"}
LABEL = {"full": "full federation", "no-sbc": "SBC tier removed"}


def mean_ci(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = np.sum(~np.isnan(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    s = np.nanstd(a, axis=axis, ddof=1)
    h = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * s / np.sqrt(np.maximum(n, 1))
    return m, m - h, m + h


def unit(lo, hi):
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def light(c, f=0.78):
    """Opaque tint of a colour; EPS has no transparency, so bands are drawn solid."""
    from matplotlib.colors import to_rgb
    return tuple(x + (1 - x) * f for x in to_rgb(c))


def save(fig, name):
    for ext in ("eps", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIG / f"{name}.eps")


def fig_quantum():
    d = json.loads((RES / "quantum_full_and_no_sbc_R2f-bind.json").read_text())
    rows, W = d["rows"], sorted(d["quanta"])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for fed in ("full", "no-sbc"):
        term = np.array([[r["terminal"] for r in rows if r["federation"] == fed and r["w"] == w]
                         for w in W]).T
        width = np.array([[r["last_width"] for r in rows if r["federation"] == fed and r["w"] == w]
                          for w in W]).T
        for k, (a, lab) in enumerate(((term, "Terminal macro-F1"),
                                      (width, "Clients in last updating round"))):
            m, lo, hi = mean_ci(a)
            if k == 0:
                lo, hi = unit(lo, hi)
            ax[k].plot(W, m, marker="o", ms=3, color=COL[fed], label=LABEL[fed])
            ax[k].fill_between(W, lo, hi, color=light(COL[fed]), lw=0, zorder=0)
            ax[k].set_ylabel(lab)
    for k, t in enumerate("ab"):
        ax[k].set_xscale("log", base=2)
        ax[k].set_xticks(W, [str(w) for w in W])
        ax[k].set_xlabel("Work quantum $w$ (local epochs)")
        ax[k].set_title(f"({t})", loc="left", fontsize=8.5)
    ax[0].legend(frameon=False, loc="lower left")
    save(fig, "quantum_terminal")


def fig_joules():
    d = json.loads((RES / "policies_full_and_no_sbc_R2f-bind.json").read_text())
    arms = ["DRR", "Published gate", "FedAvg-full"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for k, fed in enumerate(("full", "no-sbc")):
        runs = [r for r in d["rows"] if r["federation"] == fed]
        emax = max(p["energy_j"] for r in runs for p in r["traj"])
        grid = np.linspace(0, emax, 200)
        for arm in arms:
            curves = []
            for r in runs:
                if r["arm"] != arm:
                    continue
                e = np.array([0.0] + [p["energy_j"] for p in r["traj"]])
                f = np.array([np.nan] + [p["macro_f1"] for p in r["traj"]])
                # score held from the last round reached; undefined past the run's own spend
                idx = np.searchsorted(e, grid, side="right") - 1
                c = f[np.clip(idx, 0, len(f) - 1)]
                c[grid > e[-1]] = np.nan
                curves.append(c)
            m, lo, hi = mean_ci(np.array(curves))
            lo, hi = unit(lo, hi)
            ok = np.sum(~np.isnan(np.array(curves)), axis=0) >= min(3, len(curves))
            ax[k].plot(grid[ok], m[ok], color=COL[arm], label=arm)
            ax[k].fill_between(grid[ok], lo[ok], hi[ok], color=light(COL[arm]), lw=0, zorder=0)
        ax[k].set_xlabel("Cumulative training energy (J)")
        ax[k].set_title(f"({'ab'[k]}) {LABEL[fed]}", loc="left", fontsize=8.5)
    ax[0].set_ylabel("Macro-F1 on held-out devices")
    ax[0].legend(frameon=False, loc="lower right")
    save(fig, "f1_vs_joules")


def fig_margin():
    d = json.loads((RES / "margin_meter_R2f-bind.json").read_text())
    rows = [r for r in d["rows"] if r["sweep"] == "margin"]
    deltas = sorted({r["delta"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    cmap = plt.get_cmap("viridis")
    kap = sorted({r["kappa"] for r in rows})
    for i, k in enumerate(kap):
        m = [np.mean([r["min_fraction"] for r in rows if r["kappa"] == k and r["delta"] == dl
                      and r["check"]]) for dl in deltas]
        ax.plot(deltas, m, marker="o", ms=3, color=cmap(i / max(1, len(kap) - 1)),
                label=f"$\\kappa$ = {k:g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Client margin $\\delta$")
    ax.set_ylabel("Lowest charge / allocation")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "margin_overdraw")


if __name__ == "__main__":
    for f in (fig_quantum, fig_joules, fig_margin):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, "(missing", Path(e.filename).name + ")")
