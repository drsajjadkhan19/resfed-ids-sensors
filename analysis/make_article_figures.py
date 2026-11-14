#!/usr/bin/env python3
"""Figures 3 and 4 of the article, drawn only from the saved run outputs.

    python analysis/make_article_figures.py [results/journal] [figures]

Both figures are line graphs. Figure 3 draws one thin line per device-disjoint
split and a bold line for the mean (or median); Figure 4 draws the mean
macro-F1 reached at each level of cumulative energy, with a 95% band. Bands are
drawn as opaque tints because EPS has no transparency.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "journal"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "ps.fonttype": 42, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
RED, GREY, BLUE, ORANGE = "#B2182B", "#555555", "#2166AC", "#E08214"


def tint(c, f=0.8):
    return tuple(x + (1 - x) * f for x in to_rgb(c))


def save(fig, name):
    for ext in ("eps", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.eps")


d = json.loads((RES / "cape_factorial_R2f-bind.json").read_text())
seeds = d["seeds"]
rows = {(r["arm"], r["seed"]): r for r in d["rows"]}

# ---------------------------------------------------------------- Figure 3 --
W = [1, 2, 4, 8, 16, 32]
term = np.array([[rows[(f"DRR-w{w}", s)]["final_macro_f1"] for w in W] for s in seeds])
tail = np.array([[rows[(f"DRR-w{w}", s)]["mean_last5"] for w in W] for s in seeds])
width = np.array([[[p["width"] for p in rows[(f"DRR-w{w}", s)]["traj"] if p["width"] > 0][-1]
                   for w in W] for s in seeds])
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.28})
for i in range(len(seeds)):
    a.plot(W, term[i], color=tint(RED, 0.65), lw=0.6, zorder=1,
           label="terminal, each split" if i == 0 else None)
    b.plot(W, width[i] + (i - 4.5) * 0.04, color=tint(GREY, 0.55), lw=0.6, zorder=1,
           label="each split" if i == 0 else None)
a.plot(W, term.mean(0), "-s", color=RED, lw=1.5, ms=3.2, mfc="white", zorder=3, label="terminal, mean")
a.plot(W, tail.mean(0), "--o", color=GREY, lw=1.3, ms=3.2, mfc="white", zorder=3, label="tail-5, mean")
a.axhline(0.10, color="0.35", lw=0.6, ls=":")
a.text(1.05, 0.105, "collapse threshold", fontsize=6.5, color="0.3", va="bottom")
a.set_ylim(0, 0.35)
a.set_ylabel("Macro-F1 on held-out devices")
b.plot(W, np.median(width, 0), "-s", color=BLUE, lw=1.5, ms=3.2, mfc="white", zorder=3, label="median")
b.set_ylim(0, 13)
b.set_yticks([0, 2, 4, 6, 8, 10, 12])
b.set_ylabel("Clients in last updating round")
for ax, lab in ((a, "(a)"), (b, "(b)")):
    ax.set_xscale("log", base=2)
    ax.set_xticks(W, [str(w) for w in W])
    ax.set_xlabel("Local work quantum $w$ (epochs)")
    ax.set_title(lab, loc="left", fontsize=8.5, fontweight="bold")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
         columnspacing=1.0, handlelength=1.8)
b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handlelength=1.8)
save(fig, "endpoint_per_split")

# ---------------------------------------------------------------- Figure 4 --
arms = [("DRR-w2", "deficit round robin", BLUE, "-"),
        ("FedAvg-full", "full participation", ORANGE, "--"),
        ("Threshold-0.3", r"state-of-charge gating, $\tau=0.30$", RED, "-.")]
grid = np.linspace(0, 120, 241)


def readout(traj, cap):
    """Score of the last evaluated model whose cumulative energy is within cap."""
    val = np.nan
    for p in traj:
        if p["energy_j"] <= cap + 1e-9:
            val = p["macro_f1"]
        else:
            break
    return val


fig, ax = plt.subplots(figsize=(4.4, 2.7))
for arm, label, col, ls in arms:
    M = np.array([[readout(rows[(arm, s)]["traj"], c) for c in grid] for s in seeds])
    ok = ~np.isnan(M).any(0)
    m = np.nanmean(M, 0)
    h = stats.t.ppf(0.975, len(seeds) - 1) * np.nanstd(M, 0, ddof=1) / np.sqrt(len(seeds))
    ax.fill_between(grid[ok], np.clip(m - h, 0, 1)[ok], np.clip(m + h, 0, 1)[ok],
                    color=tint(col, 0.82), lw=0, zorder=0)
for arm, label, col, ls in arms:
    M = np.array([[readout(rows[(arm, s)]["traj"], c) for c in grid] for s in seeds])
    ok = ~np.isnan(M).any(0)
    ax.plot(grid[ok], np.nanmean(M, 0)[ok], ls, color=col, lw=1.4, label=label, zorder=2)
ax.set_xlabel("Cumulative training energy (J)")
ax.set_ylabel("Macro-F1 on held-out devices")
ax.set_xlim(0, 120)
ax.set_ylim(0, None)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.2)
save(fig, "f1_vs_joules")
