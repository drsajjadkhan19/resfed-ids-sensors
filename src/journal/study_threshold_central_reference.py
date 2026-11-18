"""Threshold sweep and centralised device-disjoint reference on the clean federation.

Run from code/:  python -m src.journal.study_threshold_central_reference
"""
import json
from src.journal.study_twolevel_preservation import (build_fed, run, REGIMES, spc,
                                                     ThresholdGating, OUT)
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)
from src.journal.federation import evaluate, _train_local
from src.journal.architectures import build

def central(fed, cfg, seed, cw, epochs=20):
    """Centralised device-disjoint reference: same MLP, pooled clean training clients."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.concatenate([fed.client_idx[d] for d in fed.client_devices])
    X, y = fed.xy(idx)
    model = build("mlp", fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy(); best = []
    for e in range(epochs):
        _train_local(model, X, y, cfg, torch.device("cpu"), cw, epochs=1)
        best.append(evaluate(model, Xe, ye, torch.device("cpu"), fed.n_classes)["macro_f1"])
    return best[-1], max(best)

def job(seed):
    out = []
    fed, cfg = build_fed(seed, clean=True)
    cw = spc.class_weights(fed)
    for reg in ("R2f-bind", "R2f-slack"):
        for tau in (0.2, 0.5):
            out.append(dict(seed=seed, regime=reg, tau=tau,
                            **run(fed, cfg, REGIMES[reg], ThresholdGating("g", cfg.m, tau), seed, cw)))
    last, best = central(fed, cfg, seed, cw)
    out.append(dict(seed=seed, regime="central", tau=None, terminal=last, best=best))
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(workers()) as p:
        rows = [r for c in p.map(job, range(1, 11)) for r in c]
    path = OUT / "threshold_central_reference.json"
    json.dump(rows, open(path, "w"), indent=1)
    print("saved", path, len(rows))
