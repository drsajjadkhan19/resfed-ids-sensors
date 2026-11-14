"""The committed replication: the quantum x server-rule grid on CICIoT2023.

The internal analysis plan listed, among the outcomes to be reported whatever
they showed, a replication of the quantum x server-rule interaction on the
transfer corpus. This study runs the forty cells of the identification grid
(``src.journal.study_quantum_aggregator_grid``) on CICIoT2023 with the same arm
runner, the same energy model and the same budget regime. Clients are Dirichlet
partitions at alpha = 0.5 (72 clients, 1200-row cap), and hardware tiers are
drawn from the tier frequencies of the natural device corpus, so that the stock
distribution is the one under which the interaction was identified.

Run from the repository root:  python -m src.revision.study_transfer_grid
"""
from __future__ import annotations

import numpy as np

from src.journal.federation import FedConfig
from src.journal.partitioning import dirichlet_partition
from src.journal.preprocessing import load_clean
from src.journal.study_cape_factorial import run_arm
from src.journal.study_cape_transfer import assign_hardware
from src.journal.study_quantum_aggregator_grid import grid_arms
from src.revision.common import ROOT, SEEDS, pool_map, regime, save

ALPHA, K, CAP = 0.5, 72, 1200
# A device name that ``device_profiles.classify_device`` maps to each tier, so the
# arm runner assigns the tier drawn for each synthetic client.
TIER_NAME = {"sbc": "raspberry pi", "gateway": "hub", "av": "speaker",
             "camera": "camera", "appliance": "roomba", "mcu": "smart plug"}


class TransferFederation:
    """The interface of ``Federation`` that ``run_arm`` uses, over CICIoT2023."""

    def __init__(self, seed: int):
        d = load_clean(ROOT / "data" / "clean_ciciot2023")
        self.classes = [str(c) for c in d["class_names"]]
        self.n_classes = len(self.classes)
        self.Xtr, self.ytr = d["X_train"], d["y_train"]
        self.Xte, self.yte = d["X_test"], d["y_test"]
        self.n_features = int(self.Xtr.shape[1])
        parts = dirichlet_partition(self.ytr, K, alpha=ALPHA, seed=seed)
        rng = np.random.default_rng(seed)
        parts = [np.asarray(p if len(p) <= CAP else rng.choice(p, CAP, replace=False),
                            dtype=np.int64) for p in parts]
        parts = [p for p in parts if len(p) >= 8]
        hw = assign_hardware(len(parts), "diad", seed)
        self.client_devices = list(range(len(parts)))
        self.device_names = [TIER_NAME[h.name] for h in hw]
        self.client_idx = dict(enumerate(parts))
        self.client_sizes = [len(p) for p in parts]
        self.y = self.ytr

    def xy(self, idx):
        return self.Xtr[idx], self.ytr[idx]

    def eval_xy(self):
        return self.Xte, self.yte


def job(seed):
    cfg = FedConfig(rounds=40, m=12, max_rows_per_client=CAP, local_epochs=2)
    cfg.seed = seed
    fed = TransferFederation(seed)
    y = np.concatenate([fed.ytr[fed.client_idx[d]] for d in fed.client_devices])
    f = np.bincount(y, minlength=fed.n_classes) / len(y)
    cw = 1.0 / np.maximum(f, 1e-9)
    cw = cw / cw.mean()
    rg = regime("R2f-bind")
    out = []
    for arm in grid_arms():
        r = run_arm(fed, rg, cfg, arm, seed, cw)
        r["traj"] = [dict(round=p["round"], macro_f1=p["macro_f1"], energy_j=p["energy_j"],
                          width=p["width"]) for p in r["traj"]]
        out.append(r)
    return out


def main():
    rows = [r for c in pool_map(job, SEEDS) for r in c]
    save("transfer_grid_ciciot2023", dict(alpha=ALPHA, K=K, cap=CAP, seeds=SEEDS,
                                          arms=[a["name"] for a in grid_arms()], rows=rows))


if __name__ == "__main__":
    main()
