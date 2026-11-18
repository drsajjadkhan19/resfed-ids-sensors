#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypothesis H1: does early client drift predict the terminal score?

The quantum × server-rule surface spans a wide range within one budget regime,
and that is so far an observation with no predictor attached. The candidate is
client drift: a large local quantum moves client models away from the global
model, and a server rule that follows them too confidently amplifies the
disagreement.

H1 claims prediction, not explanation, and the distinction is load-bearing.
Quantum, server rule, round mass and effective training length all vary together
across the cells of this grid, so a correlation cannot identify a cause. What it
can establish is whether a quantity observable in the first few rounds carries
information about an outcome only visible at the horizon — which is what a
practitioner would need.

The indicator was defined before it was measured:

    D_t = || theta_global - mean_k theta_k ||_2 / || theta_global ||_2

averaged over rounds 2 to 5. The decision rule was fixed at the same time:
Spearman |rho| >= 0.6 across grid cells, and every collapsing cell (terminal
macro-F1 < 0.10) above the grid median of D.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"

RHO_THRESHOLD = 0.6
COLLAPSE = 0.10


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "quantum_aggregator_grid_R2f-bind"
    path = OUT / f"{name}.json"
    data = json.loads(path.read_text())
    rows = data["rows"]
    seeds = data["seeds"]
    idx = {(r["arm"], r["seed"]): r for r in rows}
    arms = list(dict.fromkeys(r["arm"] for r in rows))

    missing = [a for a in arms
               if not np.isfinite(idx[(a, seeds[0])].get("drift_early", np.nan))]
    if missing:
        print(f"  {len(missing)} of {len(arms)} arms carry no drift record; "
              f"re-run the grid with drift logging enabled.")
        return

    cells = []
    for a in arms:
        d = np.array([idx[(a, s)]["drift_early"] for s in seeds])
        f = np.array([idx[(a, s)]["final_macro_f1"] for s in seeds])
        cells.append({"arm": a, "drift": float(np.nanmean(d)),
                      "terminal": float(f.mean()), "terminal_sd": float(f.std(ddof=1))})

    D = np.array([c["drift"] for c in cells])
    F = np.array([c["terminal"] for c in cells])
    rho, p_rho = stats.spearmanr(D, F)
    med = float(np.median(D))
    collapsing = [c for c in cells if c["terminal"] < COLLAPSE]
    above = [c for c in collapsing if c["drift"] > med]

    print(f"H1 — early drift as a predictor of the terminal score  ({name})")
    print(f"  cells: {len(cells)}   seeds per cell: {len(seeds)}\n")
    print(f"  {'cell':<20}{'drift (rounds 2-5)':>20}{'terminal':>18}")
    print("  " + "-" * 58)
    for c in sorted(cells, key=lambda c: c["drift"]):
        mark = "  <- collapsed" if c["terminal"] < COLLAPSE else ""
        print(f"  {c['arm']:<20}{c['drift']:>20.5f}"
              f"{c['terminal']:>10.4f}±{c['terminal_sd']:<6.4f}{mark}")

    print(f"\n  Spearman rho = {rho:+.3f}  (p = {p_rho:.4g})")
    print(f"  grid median drift = {med:.5f}")
    print(f"  collapsing cells (terminal < {COLLAPSE}): {len(collapsing)}; "
          f"of these, {len(above)} lie above the median drift")

    crit_rho = abs(rho) >= RHO_THRESHOLD
    crit_collapse = (len(collapsing) == 0) or (len(above) == len(collapsing))
    verdict = "SUPPORTED" if (crit_rho and crit_collapse) else "NOT SUPPORTED"
    print(f"\n  pre-declared rule: |rho| >= {RHO_THRESHOLD} and every collapsing "
          f"cell above the median drift")
    print(f"  |rho| >= {RHO_THRESHOLD}: {crit_rho}   all collapses above median: "
          f"{crit_collapse}   ->  H1 {verdict}")
    if verdict == "NOT SUPPORTED":
        print("  Reported outcome: the interaction has no early predictor among "
              "the quantities declared, and that negative result stands.")

    (OUT / f"drift_prediction_{name}.json").write_text(json.dumps(
        {"source": name, "cells": cells, "spearman_rho": float(rho),
         "spearman_p": float(p_rho), "median_drift": med,
         "n_collapsing": len(collapsing), "n_collapsing_above_median": len(above),
         "verdict": verdict}, indent=1))
    print(f"\nSaved -> {OUT/f'drift_prediction_{name}.json'}")


if __name__ == "__main__":
    main()
