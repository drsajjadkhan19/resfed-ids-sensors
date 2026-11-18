#!/usr/bin/env python3
"""Run every study behind the article, in order, and log what was run.

    python reproduce.py --list                 list the studies and their outputs
    python reproduce.py                        run everything that has no output yet
    python reproduce.py --only order-only selection
    python reproduce.py --force                rerun even when the output exists

Each study is run as ``python -m <module> <args>`` from the repository root with
``PYTHONHASHSEED=0``. Its console output goes to ``logs/<name>.log``, and one
line per run is appended to ``results/RUN_LOG.jsonl`` with the command, the exit
code, the wall time, and the SHA-256 of every output file. ``RESFED_WORKERS``
sets the number of worker processes (default: all cores); results do not depend
on it, because every job seeds itself from its split index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
J = "results/journal"

# name, phase, module, args, outputs, what the output supports in the article
STUDIES = [
    # ------------------------------------------------------------ article tables
    ("policy-comparison-bind", "article", "src.journal.study_policy_comparison", ["R2f-bind"],
     [f"{J}/policy_comparison_R2f-bind.json"], "Tables 8, 9 and 19"),
    ("policy-comparison-slack", "article", "src.journal.study_policy_comparison", ["R2f-slack"],
     [f"{J}/policy_comparison_R2f-slack.json"], "Table 19"),
    ("misspecification", "article", "src.journal.study_model_misspecification", ["R2f-bind"],
     [f"{J}/model_misspecification_R2f-bind.json"], "Table 10"),
    ("baseline-misspecification", "article", "src.journal.study_baseline_misspecification",
     ["R2f-bind"], [f"{J}/baseline_misspecification_R2f-bind.json"], "Tables 10 and 13"),
    ("two-level-accuracy", "article", "src.journal.study_two_level_accuracy", ["R2f-bind"],
     [f"{J}/two_level_accuracy_R2f-bind.json"], "Tables 11 and 12"),
    ("order-only", "article", "src.journal.study_order_only", ["R2f-bind"],
     [f"{J}/order_only_R2f-bind.json"], "Table 15"),
    ("gate-order", "article", "src.journal.study_gate_order", [],
     [f"{J}/gate_order.json"], "Table 9 (threshold sweep) and Section 5.5"),
    ("tier-decoupling", "article", "src.journal.study_order_tier_decoupling", [],
     [f"{J}/order_tier_decoupling_R2f-bind.json"], "Table 17"),
    ("clean-vs-full", "article", "src.journal.study_twolevel_preservation", ["A"],
     [f"{J}/twolevel_preservation_clean_vs_full_federation.json"], "Table 16"),
    ("threshold-central", "article", "src.journal.study_threshold_central_reference", [],
     [f"{J}/threshold_central_reference.json"], "Table 9 (partitions removed) and the centralised reference of Section 5"),
    ("factorial", "article", "src.journal.study_cape_factorial", ["R2f-bind"],
     [f"{J}/cape_factorial_R2f-bind.json"], "Figures 3 and 4"),
    ("selection", "article", "src.journal.study_configuration_selection", ["R2f-bind"],
     [f"{J}/policy_family_selection_balanced_R2f-bind.json"], "Table 18; Table 20, H0"),
    # ------------------------------------------------------- supporting studies
    ("factorial-share-75", "supporting", "src.journal.study_cape_factorial",
     ["R2f-bind", "1.7315", "0.8766"], [f"{J}/cape_factorial_R2f-bind_c1.7315_t0.8766.json"],
     "Section 4.4, communication-share sweep, 75%"),
    ("factorial-share-50", "supporting", "src.journal.study_cape_factorial",
     ["R2f-bind", "3.463", "0.5844"], [f"{J}/cape_factorial_R2f-bind_c3.463_t0.5844.json"],
     "Section 4.4, communication-share sweep, 50%"),
    ("factorial-share-30", "supporting", "src.journal.study_cape_factorial",
     ["R2f-bind", "4.8481", "0.3506"], [f"{J}/cape_factorial_R2f-bind_c4.8481_t0.3506.json"],
     "Section 4.4, communication-share sweep, 30%"),
    ("quantum-grid", "supporting", "src.journal.study_quantum_aggregator_grid", ["R2f-bind"],
     [f"{J}/quantum_aggregator_grid_R2f-bind.json"], "Table 20, H1; mass weighting (Section 5.8)"),
    ("drift", "supporting", "src.journal.analyse_drift_prediction",
     ["quantum_aggregator_grid_R2f-bind"],
     [f"{J}/drift_prediction_quantum_aggregator_grid_R2f-bind.json"], "Table 20, H1"),
    ("mass-floor", "supporting", "src.journal.study_mass_floor", ["R2f-bind"],
     [f"{J}/mass_floor_R2f-bind.json"], "Table 20, H3"),
    ("hardware-robustness", "supporting", "src.journal.study_hardware_class_robustness",
     ["R2f-bind"], [f"{J}/hardware_class_robustness_R2f-bind.json"],
     "Section 5.1, hardware assignment"),
    ("transfer-cyclic", "supporting", "src.journal.study_cape_transfer", ["cyclic"],
     [f"{J}/cape_transfer_cyclic.json"], "Table 20, H4; per-client quantum on the transfer corpus"),
    ("transfer-diad", "supporting", "src.journal.study_cape_transfer", ["diad"],
     [f"{J}/cape_transfer_diad.json"], "Table 20, H4, sampled tiers"),
    ("work-quantum", "supporting", "src.journal.study_work_quantum", ["R2f-bind"],
     [f"{J}/work_quantum_R2f-bind.json"], "Table 7"),
    ("cost-model", "supporting", "src.journal.study_cost_model_sensitivity", [],
     [f"{J}/cost_model_sensitivity.json"], "cost-model sensitivity"),
    ("threshold-characterisation", "supporting", "src.journal.study_threshold_characterisation",
     [], [f"{J}/threshold_characterisation.json"], "Propositions 2 and 3 against runs"),
    ("stopping-time", "supporting", "src.journal.study_stopping_time_independence", [],
     [f"{J}/stopping_time_independence.json"], "Proposition 4 against runs"),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--phase", choices=["article", "supporting"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    todo = [s for s in STUDIES
            if (not a.only or s[0] in a.only) and (not a.phase or s[1] == a.phase)]
    if a.list:
        for name, phase, mod, args, outs, what in todo:
            done = all((ROOT / o).exists() for o in outs)
            print(f"{'done' if done else 'todo':<5} {phase:<11} {name:<28} {what}")
        return

    for d in ("data/diad_federated/X.npy",):
        if not (ROOT / d).exists() and not a.dry_run:
            sys.exit(f"missing {d}: run `python -m src.journal.dataset_builder` first (see data/README.md)")

    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
    (ROOT / "logs").mkdir(exist_ok=True)
    for name, phase, mod, args, outs, what in todo:
        if not a.force and all((ROOT / o).exists() for o in outs):
            print(f"skip  {name} (output exists)")
            continue
        cmd = [sys.executable, "-m", mod, *args]
        print(f"run   {name}: {' '.join(cmd[1:])}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        with open(ROOT / "logs" / f"{name}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rec = dict(name=name, phase=phase, command=" ".join(cmd[1:]), exit_code=rc,
                   minutes=round((time.time() - t0) / 60, 2),
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   outputs={o: (sha(ROOT / o) if (ROOT / o).exists() else None) for o in outs})
        with open(ROOT / "results" / "RUN_LOG.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"      exit {rc} after {rec['minutes']} min", flush=True)
        if rc != 0:
            sys.exit(f"{name} failed; see logs/{name}.log")


if __name__ == "__main__":
    main()
