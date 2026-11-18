#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the equal-trial-count policy result survive when the budget stops binding?

The confirmatory comparison of the article is run in the binding regime, where
the energy stock rather than the round horizon is what stops training. That is
the motivated case, but it leaves a question a reader is entitled to ask: if
the budget does not bind, does the ranking hold, collapse to equivalence, or
reverse?

This script answers it from the two artefacts directly. It re-derives, for each
regime, the terminal score of every family, the paired difference against
deficit round robin with a 95% interval, and the 90% interval that the
article's pre-declared equivalence test reads against a +/-0.05 margin. Nothing
is recomputed from the models: the artefacts hold one frozen test score per
family per seed, which is the quantity the comparison is defined on.

The statistics are the article's own: paired within seed, ten seeds, Student
intervals. No correction for multiplicity is applied here because the contrast
set is the three comparators against deficit round robin, declared in the
article, not a search.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
FAMILIES = ["DRR", "Threshold", "Oort", "PowerOfChoice"]
LABEL = {"DRR": "Deficit round robin", "Threshold": "Threshold gating",
         "Oort": "Energy-adapted Oort", "PowerOfChoice": "Power-of-choice"}
MARGIN = 0.05


def load(regime):
    p = OUT / f"policy_family_selection_balanced_{regime}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    scores = {f: np.array([r[f"test_{f}"] for r in d["rows"]]) for f in FAMILIES}
    energy = {f: np.array([r[f"J_{f}"] for r in d["rows"]]) for f in FAMILIES}
    return d, scores, energy


def interval(diff, level):
    n = len(diff)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return float(diff.mean() - h), float(diff.mean() + h)


def report(regime):
    got = load(regime)
    if got is None:
        print(f"{regime}: artefact not present yet")
        return None
    d, scores, energy = got
    print("=" * 78)
    print(f"{regime}   seeds={len(d['seeds'])}   config={d['config']}")
    print(f"  {'family':<22}{'terminal':>18}{'energy J':>10}"
          f"{'vs DRR (95% CI)':>30}{'p':>9}")
    rows = []
    for f in FAMILIES:
        a, e = scores[f], energy[f]
        cell = f"{a.mean():.4f} +/- {a.std(ddof=1):.4f}"
        if f == "DRR":
            print(f"  {LABEL[f]:<22}{cell:>18}{e.mean():>10.1f}{'--':>30}{'':>9}")
            continue
        diff = scores["DRR"] - a
        lo, hi = interval(diff, 0.95)
        lo90, hi90 = interval(diff, 0.90)
        p = float(stats.ttest_rel(scores["DRR"], a).pvalue)
        print(f"  {LABEL[f]:<22}{cell:>18}{e.mean():>10.1f}"
              f"{f'{diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]':>30}{p:>9.4g}")
        # The article declares equivalence only if the 90% interval lies wholly
        # inside the margin; outside it in one direction is a decided difference.
        verdict = ("equivalent" if -MARGIN < lo90 and hi90 < MARGIN else
                   "DRR ahead beyond margin" if lo90 > MARGIN else
                   "DRR behind beyond margin" if hi90 < -MARGIN else
                   "undecided")
        rows.append((f, diff.mean(), (lo, hi), (lo90, hi90), p, verdict))
    print(f"\n  equivalence at +/-{MARGIN} on the 90% interval:")
    for f, m, _, (lo90, hi90), _, v in rows:
        print(f"    DRR vs {LABEL[f]:<22} 90% [{lo90:+.4f},{hi90:+.4f}]  {v}")
    return rows


def main():
    b = report("R2f-bind")
    s = report("R2f-slack")
    if b and s:
        print("=" * 78)
        print("  sign of the DRR margin, by regime")
        print(f"  {'comparator':<24}{'binding':>12}{'slack':>12}{'reverses':>11}")
        for (f, mb, *_), (_, ms, *_) in zip(b, s):
            print(f"  {LABEL[f]:<24}{mb:>+12.4f}{ms:>+12.4f}"
                  f"{('yes' if mb * ms < 0 else 'no'):>11}")


if __name__ == "__main__":
    main()
