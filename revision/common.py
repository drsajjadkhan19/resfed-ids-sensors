"""Shared helpers for the studies added in the revision.

Every study in this package trains with the same code path as the original
studies (``src.journal``), seeds each job from its split index alone, and writes
one JSON file with a ``_manifest`` block that records the inputs needed to
reproduce it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import src.journal.runtime  # noqa: F401  (deterministic CPU kernels)
from src.journal.architectures import build
from src.journal.device_profiles import assign_profiles, classify_device
from src.journal.federation import Federation, FedConfig, _train_local, evaluate
from src.journal.operating_regimes import REGIMES
from src.journal.study_policy_comparison import class_weights
from src.journal.study_threshold_characterisation import make_stock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision"
SEEDS = list(range(1, 11))
BASE_CFG = dict(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)


# ---------------------------------------------------------------- manifest --
def _sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest(extra: dict | None = None) -> dict:
    import scipy
    import sklearn
    data = ROOT / "data" / "diad_federated"
    files = {}
    for name in ("X.npy", "y.npy", "device.npy", "metadata.json"):
        p = data / name
        if p.exists():
            files[name] = _sha256(p)
    m = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             command=" ".join(sys.argv), python=sys.version.split()[0],
             platform=platform.platform(), torch=torch.__version__,
             numpy=np.__version__, scipy=scipy.__version__,
             sklearn=sklearn.__version__, code_sha256=code_hash(),
             data_sha256=files, seeds=SEEDS,
             workers=os.environ.get("RESFED_WORKERS", "all cores"))
    if extra:
        m.update(extra)
    return m


def save(name: str, payload: dict, extra: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, _manifest=manifest(extra))
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_json_default))
    print(f"saved {path}", flush=True)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# -------------------------------------------------------------- federation --
def regime(name: str = "R2f-bind", allocation: float | None = None,
           backbone: str | None = None):
    rg = REGIMES[name]
    changes = {}
    if allocation is not None:
        changes["allocation"] = float(allocation)
        changes["name"] = f"{name}@{allocation:g}"
    if backbone is not None:
        changes["backbone"] = backbone
    return dataclasses.replace(rg, **changes) if changes else rg


def federation(seed: int, w: int = 2, drop_tiers: tuple = ()) -> tuple:
    """The federation of a split, optionally without the clients of some tiers.

    Dropped clients are removed from the client pool before the feature
    transforms are fitted; the held-out devices are unchanged.
    """
    cfg = FedConfig(**dict(BASE_CFG, local_epochs=w))
    cfg.seed = seed
    fed = Federation(cfg)
    if drop_tiers:
        hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
        cfg.exclude_devices = tuple(int(d) for d, h in zip(fed.client_devices, hw)
                                    if h.name in drop_tiers)
        fed = Federation(cfg)
    return fed, cfg


def eval_tiers(fed) -> np.ndarray:
    """Hardware tier of the device behind each evaluation row."""
    dev = np.load(ROOT / "data" / "diad_federated" / "device.npy", mmap_mode="r")
    rows_dev = np.asarray(dev[fed.eval_idx])
    tier_of = {d: classify_device(fed.device_names[d]) for d in np.unique(rows_dev)}
    return np.array([tier_of[d] for d in rows_dev])


@torch.no_grad()
def predict(model, X) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 8192):
        out.append(model(torch.tensor(np.asarray(X[i:i + 8192]))).argmax(1).numpy())
    return np.concatenate(out)


def tier_scores(pred, y, tiers, classes) -> dict:
    """Recall on the held-out rows of each hardware tier.

    ``attack_recall`` is the share of non-benign rows predicted as some attack
    class, which is what an operator needs from a detector on that tier;
    ``macro_recall`` averages the recall of the classes present on the tier.
    """
    benign = classes.index("Benign") if "Benign" in classes else -1
    out = {}
    for t in sorted(set(tiers)):
        m = tiers == t
        yt, pt = y[m], pred[m]
        att = yt != benign
        cls = np.unique(yt)
        out[t] = dict(rows=int(m.sum()),
                      attack_rows=int(att.sum()),
                      attack_recall=(float((pt[att] != benign).mean()) if att.any() else None),
                      macro_recall=float(np.mean([(pt[yt == c] == c).mean() for c in cls])))
    return out


def train(fed, cfg, rg, select_round, seed, cw, eval_every: bool = True) -> dict:
    """Train one arm. ``select_round(t)`` returns the clients that train in round t.

    The aggregation is sample-weighted federated averaging, as in every study of
    the article. Every round is evaluated so that budget-matched readouts are
    exact.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")
    model = build(rg.backbone, fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy()
    traj = []
    for t in range(1, cfg.rounds + 1):
        trained, energy = select_round(t)
        ups = []
        for k in trained:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(rg.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            _train_local(loc, Xb, yb, cfg, dev, cw)
            ups.append((loc.state_dict(), len(yb)))
        if ups:
            tot = sum(n for _, n in ups)
            model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                   .to(ups[0][0][q].dtype) for q in ups[0][0]})
        if eval_every or t == cfg.rounds:
            f1 = evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"]
        else:
            f1 = float("nan")
        traj.append(dict(round=t, macro_f1=f1, energy_j=float(energy()), width=len(trained)))
    final = evaluate(model, Xe, ye, dev, fed.n_classes)
    pred = predict(model, Xe)
    f1s = [p["macro_f1"] for p in traj]
    return dict(terminal=final["macro_f1"], accuracy=final["accuracy"],
                tail5=float(np.nanmean(f1s[-5:])), best=float(np.nanmax(f1s)),
                per_class_precision=final["per_class_precision"],
                per_class_recall=final["per_class_recall"],
                per_class_f1=final["per_class_f1"], confusion=final["confusion"],
                tiers=tier_scores(pred, ye, eval_tiers(fed), fed.classes),
                traj=traj)


def single_level(policy, stock):
    """Round function for a server-only policy charged with the exact cost."""
    def step(t, T):
        sel = policy.select(t, T, stock, None)
        for k in sel:
            stock.charge(k)
        return sel
    return step


def pool_map(fn, jobs):
    from multiprocessing import Pool
    from src.journal.runtime import workers
    with Pool(workers()) as pool:
        return pool.map(fn, jobs)


__all__ = ["ROOT", "OUT", "SEEDS", "BASE_CFG", "regime", "federation", "train",
           "single_level", "class_weights", "make_stock", "save", "pool_map",
           "assign_profiles"]
