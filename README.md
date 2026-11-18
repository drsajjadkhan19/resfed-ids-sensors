# ResFed-IDS: federated intrusion detection under a finite energy stock

Code and run outputs for the article

> S. Kante, K. Kucuk, S. A. Khan, M. Z. Konyar, A. Ünlü, M. Asshad.
> *Preserving Devices Without Calibrating Energy: Bounded Guarantees and Service
> Order in Two-Level Resource-Aware Federated Intrusion Detection for the
> Internet of Things.* Sensors (under review).

Every number, table and results figure in the article comes from the runs whose
outputs are saved in `results/journal/`. The statistics and Figures 3 and 4 are
recomputed from those files by the scripts in `analysis/`, without training.

## Layout

| Path | Contents |
|---|---|
| `src/journal/` | federation, energy model, participation policies, two-level control and the study scripts |
| `src/utils/` | preprocessing helpers |
| `reproduce.py` | runs every study in order and logs commands, exit codes and output checksums |
| `analysis/` | statistics (`stats_journal.py`) and Figures 3 and 4 (`make_article_figures.py`), computed only from `results/journal/` |
| `results/journal/` | outputs of the runs behind the article and the Supplementary Materials (JSON) |
| `results/exploratory/` | outputs of exploratory runs that the article does not report (see below) |
| `figures/` | Figures 1 to 4 of the article |
| `tools/` | synthetic data and a smoke test that runs every study end to end |
| `data/` | where the corpora go (see `data/README.md`) |

## Where each table and figure comes from

Table and figure numbers follow the revised article.

| Article | Saved output (`results/journal/`) | Study script (`src/journal/`) |
|---|---|---|
| Table 7 (terminal against tail-5) | `work_quantum_R2f-bind.json` | `study_work_quantum.py` |
| Figures 3 and 4 | `cape_factorial_R2f-bind.json` | `study_cape_factorial.py` |
| Table 8 (declining thin rounds) | `policy_comparison_R2f-bind.json` | `study_policy_comparison.py` |
| Table 9 (unspent energy, threshold sweep) | `policy_comparison_R2f-bind.json`, `gate_order.json`, `threshold_central_reference.json` | `study_policy_comparison.py`, `study_gate_order.py`, `study_threshold_central_reference.py` |
| Table 10 (overdrawn allocations by κ) | `model_misspecification_R2f-bind.json`, `baseline_misspecification_R2f-bind.json` | `study_model_misspecification.py`, `study_baseline_misspecification.py` |
| Tables 11 and 12 (two-level accuracy, ablation) | `two_level_accuracy_R2f-bind.json` | `study_two_level_accuracy.py` |
| Table 13 (deficit round robin under misspecification) | `baseline_misspecification_R2f-bind.json` | `study_baseline_misspecification.py` |
| Table 14 (bound against δ) | none: values of Equation (8), computed analytically | |
| Table 15 (service order only) | `order_only_R2f-bind.json` | `study_order_only.py` |
| Table 16 (single-board-computer partitions removed) | `twolevel_preservation_clean_vs_full_federation.json` | `study_twolevel_preservation.py A` |
| Table 17 (partitions given the camera tier) | `order_tier_decoupling_R2f-bind.json` | `study_order_tier_decoupling.py` |
| Table 18 (families at equal tuning) | `policy_family_selection_balanced_R2f-bind.json` | `study_configuration_selection.py` |
| Table 19 (binding and slack regimes) | `policy_comparison_R2f-bind.json`, `policy_comparison_R2f-slack.json` | `study_policy_comparison.py` |
| Table 20 (planned hypotheses) | `quantum_aggregator_grid_R2f-bind.json`, `drift_prediction_quantum_aggregator_grid_R2f-bind.json`, `mass_floor_R2f-bind.json`, `quantum_family_selection_R2f-bind.json`, `cape_transfer_cyclic.json`, `cape_transfer_diad.json` | `study_quantum_aggregator_grid.py`, `analyse_drift_prediction.py`, `study_mass_floor.py`, `study_cape_transfer.py` |
| Class profiles of the devices (Section 4.1) | `device_class_counts.tsv`, `device_class_concentration.json` | `extract_device_class_counts.sh`, `device_class_analysis.py` |
| Supplementary Section S1 (readings withdrawn before submission) | `policy_family_selection_R2f-bind.json` (earlier comparison), `tiebreak_probe.json` (tie-break reversal), `late_round_survivorship_R2f-bind.json` (late-round tier composition) | `study_configuration_selection.py`, `study_late_round_survivorship.py` |

`python reproduce.py --list` prints the same mapping for every registered study.
The client-side check uses the margin δ = 0.25 of the article in every table
(`two_level.py`, function `run_round`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on CPU. Python 3.13 with the pinned versions was used for the
reported runs.

## Reproducing the article

1. Download the corpora and build the processed arrays as described in
   `data/README.md`.
2. Check that the code runs on your machine (synthetic data):
   `bash tools/smoke_test.sh`
3. Run the studies:

```bash
python reproduce.py --list                  # each study and the table or figure it supports
python reproduce.py                         # everything without an output yet
RESFED_WORKERS=8 python reproduce.py --phase article
python reproduce.py --only order-only selection --force
```

4. Recompute the statistics and draw the figures:

```bash
python analysis/stats_journal.py results/journal results/statistics_journal.json
python analysis/make_article_figures.py     # Figures 3 and 4 (EPS and PDF)
```

Steps 1 to 3 are needed only to train again. Step 4 runs on the saved outputs
alone.

When studies are run with `reproduce.py`, `results/RUN_LOG.jsonl` records the
command, the exit code, the wall time and the SHA-256 of each output, and each
new output file carries the package versions, the code hash and the checksums of
the processed data. A full run takes about 35 to 45 hours on four CPU cores.

## Exploratory outputs

`results/exploratory/` keeps outputs of runs made while the study was designed.
The article and the Supplementary Materials report none of them: earlier
configurations of the per-client quantum (`cape_ablation_*`, `cape_sensitivity_*`,
`cape_validation_selection_*`), the development controls of the order contrast
(`order_controls_R2f-bind.json`, superseded by Tables 15 and 17), a first transfer anchor on CICIoT2023
(`anchor_transfer_ciciot2023.json`), and experiment B of
`study_twolevel_preservation.py` (`twolevel_preservation_cost_misspecification.json`),
which tests a different client-side check in which the client learns its true
cost after the first participation and refuses below 1.1 times that estimate.

## Determinism

Each job seeds NumPy and PyTorch from its split index (seeds 1 to 10), PyTorch
runs single-threaded with deterministic kernels (`src/journal/runtime.py`), and
`reproduce.py` sets `PYTHONHASHSEED=0`. The number of worker processes
(`RESFED_WORKERS`) changes the wall time only. Results are bit-identical on the
same platform and package versions; across CPU architectures, floating-point
summation order can move the last digits.

## Main settings

| Setting | Value |
|---|---|
| Rounds, server capacity | T = 40, m = 12 |
| Rows per client (cap), local epochs | 1200, w = 2 unless swept |
| Device-disjoint splits | 10 (seeds 1 to 10), 25% of devices held out |
| Budget regimes | `R2f-bind` (allocation 4e-5 of the cell), `R2f-slack` (8.6e-5) |
| Gate | server threshold 0.30 on state of charge, client margin 0.25 |
| Backbone | MLP (256, 128, 64), 73,544 parameters |

## Licence

Code: MIT (see `LICENSE`). The corpora remain under the terms of their
publisher, the Canadian Institute for Cybersecurity, University of New Brunswick.
