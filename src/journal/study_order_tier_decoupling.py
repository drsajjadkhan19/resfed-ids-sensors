"""Binding-regime sweep of the article's contrast with the sbc tier neutralised,
plus the headline cell with the sbc-tier partitions removed from the client pool.

Run from code/:  python -m src.journal.study_order_tier_decoupling
"""
import json
from pathlib import Path
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)
from src.journal.study_order_controls import make_stock_sbc_to_camera, ORIG_MAKE
import src.journal.study_policy_comparison as spc
from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating, GatedDeficit
from src.journal.device_profiles import assign_profiles

TAUS, EPOCHS = [0.2, 0.3, 0.5], [2, 8]
OUT = Path(__file__).resolve().parents[2] / "results" / "journal"


def job(args):
    seed, epochs = args
    cfg = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=epochs)
    cfg.seed = seed
    fed = Federation(cfg)
    cw = spc.class_weights(fed)
    rg = REGIMES["R2f-bind"]
    out = []
    spc.make_stock = make_stock_sbc_to_camera
    for tau in TAUS:
        for order, mk in (("published", lambda: ThresholdGating("p", cfg.m, tau)),
                          ("deficit", lambda: GatedDeficit("g", cfg.m, tau))):
            tr = spc.trajectory(fed, rg, cfg, mk(), seed, cw)
            out.append(dict(variant="sbc2cam", seed=seed, epochs=epochs, tau=tau,
                            order=order, terminal=tr["traj"][-1]["macro_f1"],
                            energy=tr["total_energy_j"], updates=tr["total_updates"]))
    spc.make_stock = ORIG_MAKE
    if epochs == 2:
        # remove the sbc-tier partitions from the client pool (test split unchanged)
        hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
        drop = tuple(int(d) for d, h in zip(fed.client_devices, hw) if h.name == "sbc")
        cfg2 = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
        cfg2.seed = seed
        cfg2.exclude_devices = drop
        fed2 = Federation(cfg2)
        cw2 = spc.class_weights(fed2)
        present = sorted(set(np.concatenate([fed2.y[fed2.client_idx[d]]
                                             for d in fed2.client_devices]).tolist()))
        for order, mk in (("published", lambda: ThresholdGating("p", 12, 0.3)),
                          ("deficit", lambda: GatedDeficit("g", 12, 0.3))):
            tr = spc.trajectory(fed2, rg, cfg2, mk(), seed, cw2)
            out.append(dict(variant="sbc_removed", seed=seed, epochs=2, tau=0.3,
                            order=order, terminal=tr["traj"][-1]["macro_f1"],
                            energy=tr["total_energy_j"], updates=tr["total_updates"],
                            classes_present=present, n_dropped=len(drop)))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(s, e) for e in EPOCHS for s in range(1, 11)]
    with Pool(workers()) as pool:
        rows = [r for chunk in pool.map(job, jobs) for r in chunk]
    path = OUT / "order_tier_decoupling_R2f-bind.json"
    json.dump(rows, open(path, "w"), indent=1)
    print("saved", path)
