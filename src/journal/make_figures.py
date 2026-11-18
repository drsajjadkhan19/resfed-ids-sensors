#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the manuscript, generated from saved artefacts.

Every value plotted is read from a results file rather than transcribed, so a
figure cannot drift away from the run that produced it. Each figure declares the
artefact it depends on and is skipped with a message if that artefact is absent,
which keeps the script runnable while studies are still in flight.

Only figures whose content is settled are included here. Anything that depends on
a comparison still running belongs in a later revision of this file, not in a
placeholder that might quietly acquire a number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results" / "journal"
FIGURES = Path(__file__).resolve().parents[3] / "article" / "journal_paper" / "figures"

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"  skipped: {name}.json not present")
        return None
    return json.loads(path.read_text())


def save(fig, name, note=""):
    """Write the vector figure for the manuscript and a raster copy for review.

    The PDF is what the manuscript includes; the PNG exists so the figure can be
    looked at rather than assumed correct.
    """
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png" + (f"  ({note})" if note else ""))


def per_arm(data, key="final_macro_f1"):
    idx = {(r["arm"], r["seed"]): r for r in data["rows"]}
    seeds = data["seeds"]
    arms = list(dict.fromkeys(r["arm"] for r in data["rows"]))
    return {a: np.array([idx[(a, s)][key] for s in seeds]) for a in arms}


# --------------------------------------------------------------------------- #
def fig_endpoint_divergence():
    """The terminal model against the trajectory average, across the quantum."""
    d = load("cape_factorial_R2f-bind")
    if d is None:
        return
    term, tail = per_arm(d, "final_macro_f1"), per_arm(d, "mean_last5")
    quanta = [1, 2, 4, 8, 16, 32]
    arms = [f"DRR-w{w}" for w in quanta]
    x = np.arange(len(quanta))

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for arms_key, series, style in ((arms, tail, dict(marker="o", ls="--",
                                                      color="0.55", label="tail-5 average")),
                                    (arms, term, dict(marker="s", ls="-",
                                                      color="C3", label="terminal model"))):
        m = np.array([series[a].mean() for a in arms_key])
        e = np.array([series[a].std(ddof=1) / np.sqrt(len(d["seeds"])) for a in arms_key])
        ax.errorbar(x, m, yerr=e, capsize=2, lw=1.2, ms=3.5, **style)

    for i, w in enumerate(quanta):
        gap = term[f"DRR-w{w}"].mean() - tail[f"DRR-w{w}"].mean()
        if gap < -0.09:
            ax.annotate("", xy=(i, term[f"DRR-w{w}"].mean()),
                        xytext=(i, tail[f"DRR-w{w}"].mean()),
                        arrowprops=dict(arrowstyle="<->", color="C3", lw=0.8))
            # keep the label inside the axes when the point sits at the edge,
            # and sit it on an opaque patch so the neighbouring line cannot
            # read through it
            side = -0.16 if i >= len(quanta) - 1 else 0.14
            ax.text(i + side, (term[f"DRR-w{w}"].mean() + tail[f"DRR-w{w}"].mean()) / 2,
                    f"{gap:+.2f}", color="C3", fontsize=7, va="center",
                    ha="right" if side < 0 else "left", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(quanta)
    ax.set_xlabel("local work quantum $w$ (epochs)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, max(term[a].mean() for a in arms) * 1.45)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2,
              columnspacing=1.0, handlelength=1.6)
    save(fig, "endpoint_divergence")


# --------------------------------------------------------------------------- #
def fig_quantum_aggregator_grid():
    """The pairing of local work with the server rule, over one budget regime."""
    d = load("quantum_aggregator_grid_R2f-bind")
    if d is None:
        return
    vals = per_arm(d, "final_macro_f1")
    quanta = [q for q in d["quanta"]]
    aggs = d["aggregators"]

    def arm_name(q, agg):
        """Cell key as the grid study writes it."""
        quantum = f"w={q[1]}" if q[0] == "uniform" else f"c/a<={q[1]}"
        return f"{quantum}|{agg}"

    def axis_label(q):
        return f"w={q[1]}" if q[0] == "uniform" else f"c/a\u2264{q[1]}"

    M = np.full((len(quanta), len(aggs)), np.nan)
    for i, q in enumerate(quanta):
        for j, agg in enumerate(aggs):
            key = arm_name(q, agg)
            if key in vals:
                M[i, j] = vals[key].mean()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=float(np.nanmax(M)))
    ax.set_xticks(range(len(aggs)))
    pretty_agg = {"none": "none", "mass": "mass", "fixed": "constant",
                  "decay": "decaying", "adam": "Adam"}
    ax.set_xticklabels([pretty_agg.get(a, a) for a in aggs], rotation=30, ha="right")
    ax.set_yticks(range(len(quanta)))
    ax.set_yticklabels([axis_label(q) for q in quanta])
    ax.set_xlabel("server update rule")
    ax.set_ylabel("local work quantum")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < 0.28 else "black")
    fig.colorbar(im, ax=ax, label="terminal macro-F1", fraction=0.046, pad=0.03)
    save(fig, "quantum_aggregator_grid",
         note=f"range {np.nanmin(M):.4f}-{np.nanmax(M):.4f}")


# --------------------------------------------------------------------------- #
def fig_cost_ratio_portability():
    """How the ranking moves when the communication share is varied."""
    files = [("85.6", "cape_factorial_R2f-bind"),
             ("75", "cape_factorial_R2f-bind_c1.7315_t0.8766"),
             ("50", "cape_factorial_R2f-bind_c3.463_t0.5844"),
             ("30", "cape_factorial_R2f-bind_c4.8481_t0.3506")]
    series = {}
    shares = []
    for share, name in files:
        d = load(name)
        if d is None:
            return
        vals = per_arm(d, "final_macro_f1")
        shares.append(float(share))
        for arm in ("DRR-w16+E+P", "DRR-w32+E+P", "F-CP", "F-CEP", "F-base"):
            if arm in vals:
                series.setdefault(arm, []).append(vals[arm].mean())

    pretty = {"DRR-w16+E+P": "uniform $w$=16", "DRR-w32+E+P": "uniform $w$=32",
              "F-CP": "per-client $c/a$", "F-CEP": "per-client $c/a$ + mass",
              "F-base": "baseline $w$=2"}
    # The four settings are the points that were run, not samples of a continuum,
    # so they are spaced evenly. A linear axis would compress 85.6 against 75 --
    # the interval where the ranking actually changes -- into almost nothing.
    x = np.arange(len(shares))
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    for arm, ys in series.items():
        ax.plot(x, ys, marker="o", ms=3.5, lw=1.2, label=pretty.get(arm, arm))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sh:g}" for sh in shares])
    ax.set_xlabel("communication share of per-participation cost (%)")
    ax.set_ylabel("terminal macro-F1")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, columnspacing=1.2)
    save(fig, "cost_ratio_portability")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("figures from saved artefacts:")
    fig_endpoint_divergence()
    fig_quantum_aggregator_grid()
    fig_cost_ratio_portability()


if __name__ == "__main__":
    main()
