#!/usr/bin/env python3
"""Task-independence of the stopping time under threshold gating.

The energy model is held fixed while only the learning problem changes. If the
schedule is determined by the energy parameters alone, then the stopping round,
participation count and stranded energy must be identical across arms even
though accuracy differs greatly. The label-shuffled arm is the sharpest case:
the task becomes unlearnable and the schedule does not move.

A cheaper backbone is used with a matched cost-to-capacity profile, since the
schedule depends on the ratio rho = e_k / B_k and not on how that ratio arises.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

from src.journal.federation import Federation, FedConfig, _train_local, evaluate
from src.journal.architectures import build, n_params
from src.journal.study_threshold_characterisation import make_stock
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
EVAL_EVERY = 5
EVAL_ROWS = 39_326


def run_arm(fed, regime, cfg, tau, arm, seed=42):
    """One (task-variant, tau) run. Energy identical across arms by construction."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    stock, hw, P = make_stock(fed, regime, cfg)
    pol = ThresholdGating(f"tau={tau}", cfg.m, tau)

    y_train = fed.y.copy()
    n_cls = fed.n_classes
    lr = cfg.lr
    if arm == "shuffled":                      # unlearnable task
        rng = np.random.default_rng(seed)
        y_train = rng.permutation(fed.y)
    elif arm == "lr_x10":
        lr = cfg.lr * 10
    elif arm == "binary":                      # benign vs attack
        y_train = (fed.y != 0).astype(fed.y.dtype)
        n_cls = 2

    model = build(regime.backbone, fed.n_features, n_cls)
    eidx = fed.eval_idx[:EVAL_ROWS]
    Xe = np.asarray(fed.X[eidx])
    ye = y_train[eidx] if arm in ("shuffled", "binary") else fed.y[eidx]

    cfg_local = FedConfig(**{**cfg.__dict__, "lr": lr})
    stop, hist = None, []
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock)
        ups = []
        for k in sel:
            d = fed.client_devices[k]
            idx = fed.client_idx[d]
            Xb, yb = np.asarray(fed.X[idx]), y_train[idx]
            loc = build(regime.backbone, fed.n_features, n_cls)
            loc.load_state_dict(model.state_dict())
            _train_local(loc, Xb, yb, cfg_local, dev)
            ups.append((loc.state_dict(), len(yb)))
            stock.charge(k)
        if ups:
            tot = sum(n for _, n in ups)
            model.load_state_dict({k_: sum(sd[k_].double() * (n / tot) for sd, n in ups)
                                   .to(ups[0][0][k_].dtype) for k_ in ups[0][0]})
        if not sel and stop is None:
            stop = t
        if t % EVAL_EVERY == 0 or t == cfg.rounds:
            hist.append({"round": t, "selected": len(sel),
                         **evaluate(model, Xe, ye, dev, n_cls)})
    return {
        "arm": arm, "tau": tau,
        "stop_round": stop,
        "participations": int(stock.participations.sum()),
        "rounds_active": int(sum(1 for h in pol.history if h)),
        "stranded_fraction": float(stock.stranded_fraction()),
        "schedule_signature": [len(h) for h in pol.history],
        "final_macro_f1": hist[-1]["macro_f1"], "final_accuracy": hist[-1]["accuracy"],
        "best_macro_f1": max(h["macro_f1"] for h in hist),
        "n_params": int(n_params(build(regime.backbone, fed.n_features, n_cls))),
        "history": hist,
    }


def main():
    cfg = FedConfig(rounds=60, m=15, max_rows_per_client=2000, local_epochs=3)
    fed = Federation(cfg)
    rg = REGIMES["R2f"]
    print(f"task independence | regime={rg.name} K={len(fed.client_devices)} "
          f"T={cfg.rounds} m={cfg.m}\n")

    rows = []
    for tau in (0.0, 0.3):
        for arm in ("normal", "shuffled", "lr_x10", "binary"):
            t0 = time.time()
            r = run_arm(fed, rg, cfg, tau, arm)
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  tau={tau}  {arm:<9} stop={str(r['stop_round']):>5} "
                  f"parts={r['participations']:>4} stranded={r['stranded_fraction']:.4f} "
                  f"macroF1={r['final_macro_f1']:.4f}  ({r['seconds']}s)")

    print("\n=== Schedule invariance check ===")
    ok = True
    for tau in (0.0, 0.3):
        g = [r for r in rows if r["tau"] == tau]
        sig = {tuple(r["schedule_signature"]) for r in g}
        same = len(sig) == 1
        ok &= same
        f1s = [r["final_macro_f1"] for r in g]
        print(f"  tau={tau}: schedules identical across all 4 task arms -> "
              f"{'YES' if same else 'NO'} | macro-F1 spread "
              f"{min(f1s):.3f}..{max(f1s):.3f} (ratio {max(f1s)/max(min(f1s),1e-9):.1f}x)")
    print(f"\n  Stopping time independent of the task: "
          f"{'CONFIRMED' if ok else 'VIOLATED'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stopping_time_independence.json").write_text(
        json.dumps({"regime": rg.name, "config": cfg.__dict__, "rows": rows},
                   indent=1, default=str))
    print(f"\nSaved -> {OUT/'stopping_time_independence.json'}")


if __name__ == "__main__":
    main()
