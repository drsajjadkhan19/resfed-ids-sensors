#!/usr/bin/env python3
"""Sensitivity of the policy comparison to the analytical cost model.

The cost model is analytical rather than instrumented, so any conclusion drawn
from it must be shown to survive an order of magnitude of error in its
parameters.

The compute term is device-dependent (j_per_flop varies by hardware class) while
the communication term is NOT (the payload is the same model for everyone).
Their ratio therefore controls how much rho_k varies across the federation, and
hence how strongly threshold gating sorts devices by hardware. That makes this
sweep a substantive robustness test rather than a formality.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

from src.journal.federation import Federation, FedConfig
from src.journal.architectures import build, n_params
from src.journal.energy_model import build_client_stock
from src.journal.device_profiles import assign_profiles
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating, LACS, FedAvgFull
from src.journal.simulation import (run_trajectory, balanced_class_weights,
                                    constant_predictor_floor)


OUT = Path(__file__).resolve().parents[2] / "results" / "journal"


def scaled_stock(fed, regime, cfg, compute_scale, comms_scale):
    """Energy stock with the cost-model constants scaled, for the sensitivity grid."""
    stock, hw, _ = build_client_stock(fed, regime, cfg,
                                      compute_scale=compute_scale,
                                      comms_scale=comms_scale)
    return stock, hw


def main():
    cfg = FedConfig(rounds=60, m=15, max_rows_per_client=2000, local_epochs=3)
    fed = Federation(cfg)
    rg = REGIMES["R2f"]
    cw = balanced_class_weights(fed)
    floor = constant_predictor_floor(fed)
    K = len(fed.client_devices)

    # (compute multiplier, comms multiplier, label)
    grid = [
        (1.0,  1.0,  "baseline"),
        (0.1,  1.0,  "compute /10  (comms-dominated)"),
        (10.0, 1.0,  "compute x10  (compute-dominated)"),
        (1.0,  0.1,  "comms /10"),
        (1.0,  10.0, "comms x10"),
    ]
    print(f"cost-model sensitivity | K={K} T={cfg.rounds} m={cfg.m}\n")
    rows = []
    for fm, tm, label in grid:
        s0, hw = scaled_stock(fed, rg, cfg, fm, tm)
        rho = s0.cost / s0.initial
        spread = float(rho.max() / rho.min())
        agg = int(np.floor(s0.initial / s0.cost).sum())
        # fraction of the cost that is communication, for the median client
        comp = np.array([6.0 * n * h.j_per_flop * fm for n, h in zip(fed.client_sizes, hw)])
        comm_frac = float(np.median((s0.cost - comp * n_params(build(rg.backbone, fed.n_features, fed.n_classes))) / s0.cost))
        print(f"### {label}: rho spread={spread:6.1f}x  budget={agg:>5}  "
              f"demand={cfg.m*cfg.rounds}  comms share~{comm_frac:.2f}")

        res = {}
        for name, pol in [("Threshold-0.3", ThresholdGating("Threshold-0.3", cfg.m, 0.3)),
                          ("LACS-pace", LACS("LACS-pace", cfg.m, use_coverage=False))]:
            r = run_trajectory(fed, rg, cfg, pol, seed=42,
                               class_weight=cw,
                               compute_scale=fm, comms_scale=tm)
            res[name] = r
            print(f"    {name:<14} parts={r['total_updates']:>4} "
                  f"strand={r['stranded_fraction']:>6.3f} F1={r['tail_macro_f1']:.4f} "
                  f"stop={str(r['last_active_round']):>5}")
        gap = res["LACS-pace"]["tail_macro_f1"] - res["Threshold-0.3"]["tail_macro_f1"]
        print(f"    -> LACS-pace advantage: {gap:+.4f} macro-F1\n")
        rows.append({"label": label, "flop_mult": fm, "tx_mult": tm,
                     "rho_spread": spread, "aggregate_budget": agg,
                     "lacs_f1": res["LACS-pace"]["final_macro_f1"],
                     "thr_f1": res["Threshold-0.3"]["final_macro_f1"],
                     "advantage": gap,
                     "lacs_stranded": res["LACS-pace"]["stranded_fraction"],
                     "thr_stranded": res["Threshold-0.3"]["stranded_fraction"]})

    print("=== Summary ===")
    adv = [r["advantage"] for r in rows]
    print(f"  LACS-pace advantage over Threshold-0.3 across all cost settings:")
    print(f"    min={min(adv):+.4f}  max={max(adv):+.4f}  "
          f"sign consistent: {'YES' if all(a > 0 for a in adv) else 'NO'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_model_sensitivity.json").write_text(json.dumps(
        {"config": cfg.__dict__, "floor": floor, "rows": rows}, indent=1, default=str))
    print(f"\nSaved -> {OUT/'cost_model_sensitivity.json'}")


if __name__ == "__main__":
    main()
