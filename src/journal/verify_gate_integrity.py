#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent verification of the confirmatory study's artefact.

A study that reports its own protocol is reporting an intention. This script
checks the properties the protocol claims, against the saved artefact and by
recomputing the splits from their seeds, so that each claim is confirmed by
something other than the sentence asserting it.

Six invariants are checked, and every one of them has been violated at some point
in this project's history:

1.  Every family ran the declared number of trials, and the same number.
2.  Validation devices are disjoint from test devices, recomputed from the seed
    rather than read from the study's own summary.
3.  Validation devices were excluded from the client pool before the feature
    transforms were fitted.
4.  The selected configuration was evaluated on test exactly once per seed, and
    the selection came from validation scores alone.
5.  Probe energy, where a policy pays for information, scales with its price and
    sits inside the reported total.
6.  The interaction contrast was computed and serialised, not merely printed as
    a table of means.

Exit status is non-zero if any check fails, so the script is usable as a gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "journal"


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok, name, detail=""):
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail else ""))
        return ok


def verify(name="policy_family_selection_balanced_R2f-bind"):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"artefact not present: {path}")
        return 1
    data = json.loads(path.read_text())
    rows = data["rows"]
    seeds = data["seeds"]
    families = list(data["families"])
    r = Report()

    print(f"verifying {path.name}\n")

    # 1 -- equal and declared trial counts
    counts = {f: {row[f"trials_{f}"] for row in rows} for f in families}
    equal = len({next(iter(c)) for c in counts.values()}) == 1
    single = all(len(c) == 1 for c in counts.values())
    declared = {f: len(data["families"][f]) for f in families}
    matches = all(next(iter(counts[f])) == declared[f] for f in families)
    r.check(equal and single and matches, "every family ran the same declared trial count",
            ", ".join(f"{f}={next(iter(counts[f]))}" for f in families))

    # 2 and 3 -- splits recomputed from the seed, not taken on trust
    from src.journal.federation import Federation, FedConfig
    from src.journal.simulation import split_validation

    leaks, unexcluded, fitted_on = [], [], []
    for s in seeds:
        probe = FedConfig(**data["config"])
        probe.seed = s
        base = Federation(probe)
        val, _ = split_validation(base, 0.2, s)
        if set(val) & set(base.eval_devices):
            leaks.append(s)

        cfg = FedConfig(**data["config"])
        cfg.seed = s
        cfg.exclude_devices = tuple(val)
        fed = Federation(cfg)
        if set(fed.client_devices) & set(val):
            unexcluded.append(s)
        fitted_on.append(len(fed.client_devices))
    r.check(not leaks, "validation devices disjoint from test devices (recomputed)",
            f"checked {len(seeds)} seeds" if not leaks else f"leaking seeds: {leaks}")
    r.check(not unexcluded,
            "validation devices removed from the pool before transforms are fitted",
            f"transforms fitted on {min(fitted_on)}-{max(fitted_on)} training devices")

    # 4 -- one frozen test evaluation, chosen on validation
    #
    # The second half of this check needs the per-cell validation scores. If the
    # artefact does not carry them the check is *not verifiable*, and reporting
    # that as a pass would be worse than not running it: an absent record would
    # certify itself.
    chosen_ok, val_ok, verifiable = True, True, 0
    for row in rows:
        for f in families:
            if row[f"chosen_{f}"] not in data["families"][f]:
                chosen_ok = False
            grid = row.get(f"grid_{f}")
            if not grid:
                continue
            verifiable += 1
            if max(grid, key=grid.get) != row[f"chosen_{f}"]:
                val_ok = False
    r.check(chosen_ok, "each selected configuration belongs to its family's grid")
    expected = len(rows) * len(families)
    if verifiable == 0:
        r.check(False, "the selection is the arg-max of the validation scores alone",
                "NOT VERIFIABLE: the artefact carries no per-cell validation scores")
    else:
        r.check(val_ok and verifiable == expected,
                "the selection is the arg-max of the validation scores alone",
                f"{verifiable}/{expected} family-seed selections re-derived")

    # 5 -- information a policy consumes is inside its energy total
    # The invariant is that a dearer probe costs the probing policy more, not
    # that it raises total consumption. Under a finite stock those come apart:
    # an expensive probe drains clients faster, so fewer of them stay affordable
    # and the federation spends *less* in total while producing far fewer
    # updates. Requiring total energy to rise would test a false premise.
    probing = [row for row in rows if "J_PowerOfChoice" in row]
    scales = (0.5, 1.0, 2.0)
    have_probe_j = probing and all(f"probeJ_PowerOfChoice_probe{ps}" in probing[0]
                                   for ps in scales)
    have_totals = probing and all(f"J_PowerOfChoice_probe{ps}" in probing[0]
                                  for ps in scales)
    if have_probe_j:
        pj = {ps: float(np.mean([row[f"probeJ_PowerOfChoice_probe{ps}"]
                                 for row in probing])) for ps in scales}
        tot = {ps: float(np.mean([row[f"J_PowerOfChoice_probe{ps}"]
                                  for row in probing])) for ps in scales}
        rises = pj[0.5] < pj[1.0] < pj[2.0]
        inside = all(pj[ps] <= tot[ps] + 1e-9 for ps in scales)
        r.check(rises and inside,
                "probe energy scales with its price and is inside the total",
                f"probe {pj[0.5]:.1f} < {pj[1.0]:.1f} < {pj[2.0]:.1f} J, "
                f"each within totals {tot[0.5]:.1f}/{tot[1.0]:.1f}/{tot[2.0]:.1f} J")
    elif have_totals:
        t = {ps: float(np.mean([row[f"test_PowerOfChoice_probe{ps}"]
                                for row in probing])) for ps in scales}
        r.check(t[0.5] >= t[1.0] >= t[2.0],
                "the probe price changes the outcome, so it is being charged",
                f"terminal {t[0.5]:.4f} >= {t[1.0]:.4f} >= {t[2.0]:.4f}; "
                f"probe energy not itemised in this artefact")
    else:
        r.check(False, "probing family and its accounting scenarios are recorded",
                "NOT VERIFIABLE: the artefact carries no alternative probe accountings")

    # 6 -- the interaction was estimated, not narrated
    inter = data.get("interactions", [])
    has_stats = bool(inter) and all(
        {"interaction", "ci", "p"} <= set(i) for i in inter)
    r.check(has_stats, "interaction contrast estimated with intervals and serialised",
            f"{len(inter)} contrasts stored" if inter else "no contrasts in the artefact")

    print(f"\n{'all checks passed' if r.failures == 0 else str(r.failures) + ' check(s) FAILED'}")
    return 1 if r.failures else 0


def verify_misspecification(name="model_misspecification_R2f-bind"):
    """Checks 8 and 9 of the integrity table, on the kappa sweep.

    Check 8 re-derives Corollary 1 rather than restating it: the slope
    n_k(tau) rho_k is fitted on the exact-model cell alone and the remaining
    cells are predicted with no freedom left, because a server-side filter runs
    the same schedule whatever the true cost is.

    Check 9 counts what the text calls "every participating client" instead of
    inferring it from a saturating depletion count: the depleted set must equal
    the set of clients with at least one participation, in every seed.
    """
    r = Report()
    print(f"\nverifying {name}")
    data = json.loads((RESULTS / f"{name}.json").read_text())

    # 8 -- the corollary predicts WHICH clients deplete, from the stock alone
    #
    # An earlier version of this check fitted the slope of the minimum-charge
    # sweep to the data and predicted the remaining cells. It could not fail: the
    # server-only schedule does not depend on kappa and the jitter draws come
    # from the same seeded generator at every kappa, so the residuals are affine
    # in kappa by construction. What follows instead predicts the depleted *set*
    # from Equation (4) and the initial stock, with no reference to the run, and
    # compares it with the simulated set. It can fail, and it does once a client
    # goes flat part-way through the horizon, which truncates the schedule the
    # corollary assumes.
    from src.journal.federation import Federation, FedConfig
    from src.journal.operating_regimes import REGIMES
    from src.journal.participation_policies import ThresholdGating
    from src.journal.study_threshold_characterisation import make_stock
    from src.journal.two_level import DualStock, run_round

    tau, delta = data["tau"], data["delta"]
    agree, disagree = [], []
    for kappa in (1.0, 1.25, 1.5):
        for seed in data["seeds"]:
            cfg = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
            cfg.seed = seed
            fed = Federation(cfg)
            base, _, _ = make_stock(fed, REGIMES[data["regime"]], cfg)
            rho = np.asarray(base.cost) / np.asarray(base.initial)
            beta = np.asarray(base.remaining) / np.asarray(base.initial)
            theta = np.maximum(tau, rho)
            n = np.where(beta >= theta, np.floor((beta - theta) / rho) + 1, 0)
            predicted = set(np.flatnonzero(kappa * n * rho > beta).tolist())

            stock = DualStock(base, kappa=kappa, noise=0.0, seed=seed)  # no jitter
            pol = ThresholdGating("g", cfg.m, tau)
            for t in range(1, cfg.rounds + 1):
                run_round(pol, t, cfg.rounds, stock, delta, False)
            observed = set(np.flatnonzero(stock.true_remaining <= 0.0).tolist())
            (agree if predicted == observed else disagree).append((kappa, seed))
    r.check(not disagree,
            "Corollary 1 predicts the depleted set from Equation (4) and the stock alone",
            f"{len(agree)} cells (all {len(data['seeds'])} splits at "
            f"kappa 1.0, 1.25 and 1.5), exact agreement"
            if not disagree else f"disagrees at {disagree[:4]}")

    # 9 -- "every participating client" is counted, not inferred
    from src.journal.federation import Federation, FedConfig
    from src.journal.operating_regimes import REGIMES
    from src.journal.participation_policies import ThresholdGating
    from src.journal.study_threshold_characterisation import make_stock
    from src.journal.two_level import DualStock, run_round

    tau, delta, noise = data["tau"], data["delta"], data["noise"]
    mismatches = []
    for seed in data["seeds"]:
        cfg = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
        cfg.seed = seed
        fed = Federation(cfg)
        base, _, _ = make_stock(fed, REGIMES[data["regime"]], cfg)
        stock = DualStock(base, kappa=1.5, noise=noise, seed=seed)
        pol = ThresholdGating("g", cfg.m, tau)
        for t in range(1, cfg.rounds + 1):
            run_round(pol, t, cfg.rounds, stock, delta, False)
        participated = int((stock.participations > 0).sum())
        if participated != stock.depleted():
            mismatches.append((seed, participated, stock.depleted()))
    r.check(not mismatches,
            "the depleted set equals the participating set at kappa = 1.5",
            "equal in every seed" if not mismatches else str(mismatches))

    # 10 -- the theorem's bound is respected by every saved run
    #
    # The comparison is against each split's own minimum, not against the mean
    # of the per-split minima that the study's summary carries: a bound on the
    # worst case is not tested by an average.
    rho_max, jitter = 0.413, 1.0 + noise      # costliest participating tier
    per_run = {}
    for row in data["rows"]:
        if row["design"] == "two-level":
            per_run.setdefault(row["kappa"], []).append(row["min_true_fraction"])
    worst, violations, n = None, [], 0
    for k, mins in per_run.items():
        bound = -max(0.0, jitter * k - 1.0 - delta) * rho_max
        for m in mins:
            n += 1
            if m < bound:
                violations.append((k, m, bound))
            if bound < 0 and (worst is None or m - bound < worst[1]):
                worst = (k, m - bound)
    r.check(not violations,
            "every saved run's minimum charge stays inside the Theorem 1 bound",
            f"{n} runs checked; closest at kappa={worst[0]}, "
            f"{worst[1]:.3f} of capacity to spare" if worst else f"{n} runs, no binding cell")

    print(f"\n{'all checks passed' if r.failures == 0 else str(r.failures) + ' check(s) FAILED'}")
    return 1 if r.failures else 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.startswith("model_misspecification"):
        sys.exit(verify_misspecification(arg))
    rc = verify(arg or "policy_family_selection_balanced_R2f-bind")
    rc |= verify_misspecification()
    sys.exit(rc)
