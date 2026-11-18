#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared harness for federated simulation runs.

Holds the round loop and the per-run bookkeeping used by every study, so that
each study file contains only its own experimental design.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from src.journal.architectures import build
from src.journal.device_profiles import assign_profiles
from src.journal.energy_model import build_client_stock
from src.journal.federation import Federation, FedConfig, train_local, evaluate


def balanced_class_weights(federation: Federation) -> np.ndarray:
    """Inverse-frequency class weights over the training clients' pooled labels."""
    y = np.concatenate([federation.y[federation.client_idx[d]]
                        for d in federation.client_devices])
    freq = np.bincount(y, minlength=federation.n_classes) / len(y)
    w = 1.0 / np.maximum(freq, 1e-9)
    return w / w.mean()


def constant_predictor_floor(federation: Federation) -> float:
    """Macro-F1 of the best constant predictor on the evaluation set.

    Reference point for a task whose class prior is skewed: a policy scoring
    below this is worse than emitting the majority class.
    """
    ye = federation.y[federation.eval_idx]
    p = np.bincount(ye, minlength=federation.n_classes).max() / len(ye)
    return 2 * p / (1 + p) / federation.n_classes


def run_trajectory(federation, regime, cfg, policy, seed, class_weight=None,
                   aggregator=None, work_quantum=None, proximal_mu=None,
                   eval_every: int = 2, unlimited_energy: bool = False,
                   compute_scale: float = 1.0, comms_scale: float = 1.0) -> Dict:
    """Run one policy for the full horizon, logging energy and accuracy.

    Logging the whole trajectory rather than a single end-point allows the
    matched-energy and matched-round comparisons to be read off the same run,
    so those comparisons cannot be selected after seeing an outcome.

    Parameters
    ----------
    aggregator : object with ``step(global_state, updates, round_mass)``, optional
        When omitted, plain sample-weighted federated averaging is used.
    work_quantum : array-like of int, optional
        Per-client local-epoch counts.
    proximal_mu : array-like of float, optional
        Per-client proximal strength; zero disables the term.
    unlimited_energy : bool
        Reference arm that ignores the budget, used to establish a ceiling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    stock, hw, n_par = build_client_stock(federation, regime, cfg, work_quantum,
                                          compute_scale, comms_scale)
    if unlimited_energy:
        stock.initial = stock.initial * 1e9
        stock.remaining = stock.initial.copy()

    K = len(federation.client_devices)
    w_vec = (np.full(K, cfg.local_epochs, dtype=int) if work_quantum is None
             else np.asarray(work_quantum, dtype=int))
    mu_vec = np.zeros(K) if proximal_mu is None else np.asarray(proximal_mu, float)

    model = build(regime.backbone, federation.n_features, federation.n_classes)
    X_eval, y_eval = federation.eval_xy()

    trajectory: List[Dict] = []
    last_active = 0
    for t in range(1, cfg.rounds + 1):
        selected = policy.select(t, cfg.rounds, stock, None)
        updates, round_mass = [], 0.0
        global_params = [p.detach().clone() for p in model.parameters()]
        for k in selected:
            Xb, yb = federation.xy(federation.client_idx[federation.client_devices[k]])
            local = build(regime.backbone, federation.n_features, federation.n_classes)
            local.load_state_dict(model.state_dict())
            train_local(local, Xb, yb, cfg, device, class_weight,
                        epochs=int(w_vec[k]), mu=float(mu_vec[k]),
                        global_params=global_params)
            updates.append((local.state_dict(), len(yb)))
            round_mass += len(yb)
            stock.charge(k)

        if updates:
            if aggregator is not None:
                model.load_state_dict(
                    aggregator.step(model.state_dict(), updates, round_mass))
            else:
                total = sum(n for _, n in updates)
                model.load_state_dict({
                    q: sum(sd[q].double() * (n / total) for sd, n in updates)
                    .to(updates[0][0][q].dtype) for q in updates[0][0]})
            last_active = t

        if t % eval_every == 0 or t == cfg.rounds:
            trajectory.append({
                "round": t,
                "width": len(selected),
                "energy_j": float(stock.initial.sum() - stock.remaining.sum()),
                "updates": int(stock.participations.sum()),
                **evaluate(model, X_eval, y_eval, device, federation.n_classes),
            })

    return {
        "policy": policy.name, "seed": seed, "trajectory": trajectory,
        "last_active_round": last_active,
        "final_macro_f1": trajectory[-1]["macro_f1"],
        "tail_macro_f1": float(np.mean([p["macro_f1"] for p in trajectory[-5:]])),
        "total_energy_j": float(stock.initial.sum() - stock.remaining.sum()),
        "total_updates": int(stock.participations.sum()),
        "stranded_fraction": float(stock.stranded_fraction()),
        "work_quantum_median": int(np.median(w_vec)),
    }


def value_at_energy(trajectory: List[Dict], target_j: float) -> float:
    """Macro-F1 at the first logged point where cumulative energy reaches target."""
    for p in trajectory:
        if p["energy_j"] >= target_j - 1e-9:
            return p["macro_f1"]
    return trajectory[-1]["macro_f1"]


def value_at_round(trajectory: List[Dict], round_index: int) -> float:
    """Macro-F1 at the last logged point on or before ``round_index``."""
    eligible = [p for p in trajectory if p["round"] <= round_index]
    return (eligible[-1] if eligible else trajectory[0])["macro_f1"]

# --------------------------------------------------------------------------- #
#  Device-disjoint validation splits
# --------------------------------------------------------------------------- #
def split_validation(fed, frac=0.2, seed=0):
    """Carve validation devices out of the CLIENT pool (test devices untouched).

    The carve is stratified on the device's dominant class and on its hardware
    class, so the validation pool does not systematically over- or
    under-represent either the label mixture or the cost ratios that the
    selected hyper-parameters have to generalise across.
    """
    rng = np.random.default_rng(seed)
    dv = [int(d) for d in fed.client_devices]
    hw = assign_profiles([fed.device_names[d] for d in dv])
    strata = {}
    for d, h in zip(dv, hw):
        yk = fed.y[fed.client_idx[d]]
        dominant = int(np.bincount(yk, minlength=fed.n_classes).argmax())
        strata.setdefault((h.name, dominant), []).append(d)

    n_val = max(2, int(len(dv) * frac))
    val = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        take = int(round(len(group) * frac))
        val.extend(group[:take])
    # top up or trim to the requested size, drawing from the unselected devices
    rest = [d for d in dv if d not in set(val)]
    rng.shuffle(rest)
    while len(val) < n_val and rest:
        val.append(rest.pop())
    val = val[:n_val]
    train = [d for d in dv if d not in set(val)]
    return val, train


def split_composition(fed, devices):
    """Class mixture, hardware mixture and row count for a set of devices."""
    if not devices:
        return {}
    hw = assign_profiles([fed.device_names[d] for d in devices])
    y = np.concatenate([fed.y[fed.client_idx[d]] for d in devices])
    counts = np.bincount(y, minlength=fed.n_classes).astype(float)
    return {"n_devices": len(devices), "n_rows": int(len(y)),
            "class_mix": (counts / counts.sum()).round(5).tolist(),
            "hw_mix": {k: sum(1 for h in hw if h.name == k)
                       for k in sorted({h.name for h in hw})}}
