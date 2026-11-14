# ResFed-IDS: federated intrusion detection under a finite energy stock

Code and run outputs for the article

> S. Kante, K. Kucuk, M. Z. Konyar, A. Ünlü, M. Asshad, S. A. Khan.
> *Preserving Devices Without Calibrating Energy: Bounded Guarantees and Service
> Order in Two-Level Resource-Aware Federated Intrusion Detection for the
> Internet of Things.* Sensors (under revision).

Every number, table and results figure in the article comes from the runs whose
outputs are saved in `results/journal/`. The statistics and Figures 3 and 4 are
recomputed from those files by the scripts in `analysis/`, without training.

## Layout

| Path | Contents |
|---|---|
| `src/journal/` | federation, energy model, participation policies, and the studies of the submitted article |
| `src/revision/` | further studies prepared during the revision (margin and meter error, per-class and per-tier scores, reduced federations, allocation sweep, small model, transfer replication); they were **not run** for the article, and the article reports none of their output |
| `reproduce.py` | runs every study in order and logs commands, exit codes and output checksums |
| `analysis/` | statistics and figures, computed only from files in `results/` |
| `results/journal/` | outputs of the runs reported in the article (JSON) |
| `tools/` | synthetic data and a smoke test that runs every study end to end in a few minutes |
| `data/` | where the corpora go (see `data/README.md`) |

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
2. Check that the code runs on your machine (synthetic data, a few minutes):
   `bash tools/smoke_test.sh`
3. Run the studies:

```bash
python reproduce.py --list          # what will run and which table or figure it supports
python reproduce.py                 # everything without an output yet
RESFED_WORKERS=8 python reproduce.py --phase revision
python reproduce.py --only policies quantum --force
```

4. Recompute the statistics and draw the figures:

```bash
python analysis/stats_journal.py results/journal results/statistics_journal.json
python analysis/make_article_figures.py     # Figures 3 and 4 (EPS and PDF)
```

Steps 1 to 3 are needed only to train again. Step 4 runs on the saved outputs
alone. After the studies in `src/revision/` have been run, `analysis/stats.py`
and `analysis/make_figures.py` summarise and plot them as well.

When studies are run with `reproduce.py`, `results/RUN_LOG.jsonl` records the
command, the exit code, the wall time and the SHA-256 of each output, and each
new output file carries the package versions, the code hash and the checksums of
the processed data. A full run takes about 35 to 45 hours on four CPU cores.

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
| Backbone | MLP (256, 128, 64), 73,544 parameters; `mlp_tiny` (64, 32, 16), 10,712 parameters |

## Licence

Code: MIT (see `LICENSE`). The corpora remain under the terms of their
publisher, the Canadian Institute for Cybersecurity, University of New Brunswick.
