#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated simulation under a finite energy stock, with device-disjoint evaluation.

Design choices:
  * energy is a finite stock drained by a work-dependent cost model rather than a
    per-round constant;
  * a single execution path, so reported results correspond to the code that
    produced them;
  * clients are natural per-device partitions taken from the capture metadata;
  * evaluation uses devices never seen during training, since a federated
    intrusion detector must generalise to hosts it has not trained on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import src.journal.runtime  # noqa: F401  (deterministic CPU kernels)
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from src.journal.energy_model import EnergyModel, EnergyStock, DeviceProfile
from src.journal.device_profiles import assign_profiles
from src.journal.participation_policies import CoverageUtility

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "diad_federated"


def pick_device(prefer: str = "auto") -> torch.device:
    """Select a compute device, preferring Metal acceleration when available."""
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------- model ----
class MLP(nn.Module):
    def __init__(self, d_in: int, n_cls: int, hidden=(256, 128, 64), p=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------- federation ---
@dataclass
class FedConfig:
    rounds: int = 30
    m: int = 10
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    hold_out_frac: float = 0.25
    hidden: tuple = (256, 128, 64)
    max_rows_per_client: int = 6000
    holdout_pool: int = 5
    exclude_devices: tuple = ()


class Federation:
    """Loads the corpus once and holds the device-disjoint split."""

    def __init__(self, cfg: FedConfig):
        self.cfg = cfg
        meta = json.loads((DATA / "metadata.json").read_text())
        self.classes: List[str] = meta["classes"]
        self.device_names: List[str] = meta["devices"]
        X = np.load(DATA / "X.npy", mmap_mode="r")
        y = np.load(DATA / "y.npy")
        dev = np.load(DATA / "device.npy")

        rng = np.random.default_rng(cfg.seed)
        udev = np.unique(dev)
        n_hold = max(1, int(len(udev) * cfg.hold_out_frac))

        # Stratified device hold-out. Devices in this corpus are strongly
        # class-specialised, so a uniform random device split yields an arbitrary
        # evaluation mixture and conflates genuine device shift with an accidental
        # label shift. The held-out set is grown greedily so its class mixture
        # tracks the corpus, while the device-disjoint property is preserved.
        #
        # The greedy step samples uniformly from the ``holdout_pool`` best
        # candidates rather than taking the arg-min. A pure arg-min is
        # deterministic once the corpus is fixed: the permutation below would
        # then only break exact ties, which do not occur for continuous class
        # profiles, so every seed would return the identical hold-out and the
        # spread across seeds would measure initialisation noise alone rather
        # than sensitivity to which devices are held out.
        prof = np.zeros((len(udev), int(y.max()) + 1))
        for i, d in enumerate(udev):
            v, c = np.unique(y[dev == d], return_counts=True)
            prof[i, v] = c
        target = prof.sum(axis=0) / prof.sum()

        pool = max(1, int(cfg.holdout_pool))
        remaining = list(rng.permutation(len(udev)))
        chosen, acc = [], np.zeros(prof.shape[1])
        for _ in range(n_hold):
            scored = []
            for i in remaining:
                cand = acc + prof[i]
                if cand.sum() == 0:
                    continue
                scored.append((float(np.abs(cand / cand.sum() - target).sum()), i))
            if not scored:
                break
            scored.sort(key=lambda t: t[0])
            take = scored[int(rng.integers(0, min(pool, len(scored))))][1]
            chosen.append(take)
            remaining.remove(take)
            acc = acc + prof[take]

        hold = udev[chosen]
        self.holdout_label_shift = float(np.abs(acc / acc.sum() - target).sum())
        self.eval_devices = set(int(d) for d in hold)
        pool_devices = [int(d) for d in udev if int(d) not in self.eval_devices]

        # Devices named in ``exclude_devices`` are removed from the client pool
        # BEFORE the feature transforms are fitted. Model selection on a
        # device-disjoint validation split is only isolated if those devices
        # influence neither the training representation nor the training data.
        self.excluded_devices = [d for d in pool_devices
                                 if d in set(int(x) for x in cfg.exclude_devices)]
        self.client_devices = [d for d in pool_devices
                               if d not in set(self.excluded_devices)]

        # per-client index sets (capped, class-stratified)
        self.client_idx: Dict[int, np.ndarray] = {}
        for d in self.client_devices + self.excluded_devices:
            idx = np.where(dev == d)[0]
            if len(idx) > cfg.max_rows_per_client:
                idx = rng.choice(idx, cfg.max_rows_per_client, replace=False)
            self.client_idx[d] = idx

        eidx = np.where(np.isin(dev, list(self.eval_devices)))[0]
        if len(eidx) > 200_000:
            eidx = rng.choice(eidx, 200_000, replace=False)

        # Feature transforms are fitted on training devices only: variance
        # filtering and median imputation are estimated after the device split,
        # so no held-out device influences the training representation.
        train_rows = np.concatenate([self.client_idx[d] for d in self.client_devices])
        Xtr_raw = np.asarray(X[train_rows])
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xtr_raw, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        finite = np.isfinite(Xtr_raw)
        col_ok = finite.any(axis=0)
        filled = np.where(finite, Xtr_raw, med)
        nunique = np.array([len(np.unique(filled[:, j])) for j in range(filled.shape[1])])
        keep = col_ok & (nunique > 1)
        self._keep = keep
        self._median = med[keep]

        self.X = X
        self.y = y
        self.eval_idx = eidx
        self.n_features = int(keep.sum())
        self.n_classes = len(self.classes)
        self.transform_note = ("variance filter + median imputation fitted on "
                               f"{len(self.client_devices)} training devices only"
                               + (f"; {len(self.excluded_devices)} devices excluded "
                                  "from the pool before fitting"
                                  if self.excluded_devices else ""))

        # Per-client class-presence bitmap and counts.
        K = len(self.client_devices)
        self.presence = np.zeros((K, self.n_classes), dtype=np.int8)
        self.counts = np.zeros((K, self.n_classes), dtype=np.int64)
        for i, d in enumerate(self.client_devices):
            v, c = np.unique(y[self.client_idx[d]], return_counts=True)
            self.presence[i, v] = 1
            self.counts[i, v] = c
        self.client_sizes = [len(self.client_idx[d]) for d in self.client_devices]

    def profiles(self) -> List[DeviceProfile]:
        hw = assign_profiles([self.device_names[d] for d in self.client_devices])
        return [DeviceProfile(capacity_j=h.capacity_j, eta=1.0, label=h.name) for h in hw], hw

    def xy(self, idx):
        """Return features with the training-fitted transform applied."""
        Xr = np.asarray(self.X[idx])[:, self._keep]
        bad = ~np.isfinite(Xr)
        if bad.any():
            Xr = np.where(bad, np.broadcast_to(self._median, Xr.shape), Xr)
        return Xr, self.y[idx]

    def eval_xy(self):
        return self.xy(self.eval_idx)


def train_local(model, Xb, yb, cfg, device, class_weight=None,
                epochs: int | None = None, mu: float = 0.0,
                global_params=None):
    """One client's local training.

    Inverse-frequency class weighting is applied because the corpus class prior is
    skewed and, under a binding energy budget, training stops early -- so
    early-round convergence matters more than asymptotic accuracy.

    ``epochs`` overrides the configured local-epoch count, allowing a per-client
    work quantum. ``mu`` adds a proximal term anchored at ``global_params``,
    following FedProx (Li et al., MLSys 2020).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    w = (torch.tensor(class_weight, dtype=torch.float32, device=device)
         if class_weight is not None else None)
    lossf = nn.CrossEntropyLoss(weight=w)
    # Keep the whole client shard resident on the device: these shards are small
    # (<= a few thousand rows) and per-batch host->device copies dominate the
    # step time on MPS otherwise.
    Xt = torch.tensor(Xb, device=device)
    yt = torch.tensor(yb, dtype=torch.long, device=device)
    n = len(yt)
    bs = cfg.batch_size
    n_epochs = cfg.local_epochs if epochs is None else int(epochs)
    anchor = ([g.detach().clone() for g in global_params]
              if (mu > 0 and global_params is not None) else None)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            if anchor is not None:
                loss = loss + 0.5 * mu * sum(((q - g) ** 2).sum()
                                             for q, g in zip(model.parameters(), anchor))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, y, device, n_classes, batch: int | None = None):
    """Batched inference.

    Batch size matters far more than it looks for the transformer: with 120
    feature tokens plus [CLS], a batch of 8192 materialises an attention tensor
    of ~1e9 elements and MPS thrashes (measured: 142 s for 5k rows). A small
    batch is ~50x faster there, and costs nothing for the MLP.
    """
    model.eval()
    if batch is None:
        batch = 512 if any(m.__class__.__name__ == "FTBlock"
                           for m in model.modules()) else 8192
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(np.asarray(X[i:i + batch])).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return {
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "accuracy": float((p == y).mean()),
        "per_class_recall": recall_score(y, p, average=None,
                                         labels=list(range(n_classes)),
                                         zero_division=0).tolist(),
        "per_class_precision": precision_score(y, p, average=None,
                                               labels=list(range(n_classes)),
                                               zero_division=0).tolist(),
        "per_class_f1": f1_score(y, p, average=None, labels=list(range(n_classes)),
                                 zero_division=0).tolist(),
        "confusion": confusion_matrix(y, p, labels=list(range(n_classes))).tolist(),
    }


def run(fed: Federation, policy, cfg: FedConfig, cost_scale: float = 1.0,
        use_coverage_util: bool = True, tag: str = "") -> dict:
    """Run one policy to exhaustion or to cfg.rounds, whichever comes first."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    profiles, hw = fed.profiles()
    model = MLP(fed.n_features, fed.n_classes, cfg.hidden)
    P = n_params(model)
    em = EnergyModel(num_params=P, local_epochs=cfg.local_epochs)

    # per-client cost from its own data volume and hardware efficiency
    stock = EnergyStock(profiles, em, fed.client_sizes)
    stock.cost = np.array([
        (6.0 * P * n * cfg.local_epochs * h.j_per_flop + 2 * P * 4 * 6.4e-7) * cost_scale
        for n, h in zip(fed.client_sizes, hw)], dtype=np.float64)

    util = CoverageUtility(fed.presence) if use_coverage_util else None
    Xe, ye = fed.xy(fed.eval_idx)

    hist = []
    for t in range(1, cfg.rounds + 1):
        sel = policy.select(t, cfg.rounds, stock, util)
        updates = []
        for k in sel:
            d = fed.client_devices[k]
            Xb, yb = fed.xy(fed.client_idx[d])
            local = MLP(fed.n_features, fed.n_classes, cfg.hidden)
            local.load_state_dict(model.state_dict())
            _train_local(local, Xb, yb, cfg, device)
            updates.append((local.state_dict(), len(yb)))
            stock.charge(k)
        if util is not None and sel:
            util.commit(sel)

        if updates:
            tot = sum(n for _, n in updates)
            new = {}
            for key in updates[0][0]:
                acc = sum(sd[key].double() * (n / tot) for sd, n in updates)
                new[key] = acc.to(updates[0][0][key].dtype)
            model.load_state_dict(new)

        met = evaluate(model, Xe, ye, device, fed.n_classes) if updates or t == 1 else hist[-1]["metrics"]
        hist.append({"round": t, "selected": len(sel), "selected_ids": list(map(int, sel)),
                     "metrics": met,
                     "energy_spent_j": float(stock.initial.sum() - stock.remaining.sum())})

    summary = stock.summary()
    return {
        "tag": tag, "policy": policy.name, "config": asdict(cfg),
        "n_params": int(P), "cost_scale": cost_scale,
        "history": hist, "energy": summary,
        "final_metrics": hist[-1]["metrics"],
        "best_macro_f1": max(h["metrics"]["macro_f1"] for h in hist),
        "rounds_active": sum(1 for h in hist if h["selected"] > 0),
        "total_participations": int(stock.participations.sum()),
        "stranded_fraction": summary["stranded_fraction"],
        "classes": fed.classes,
        "client_hw": [h.name for h in hw],
    }


# Backwards-compatible alias.
_train_local = train_local
