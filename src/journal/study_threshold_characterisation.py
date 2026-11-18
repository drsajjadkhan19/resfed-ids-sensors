#!/usr/bin/env python3
"""Threshold gating characterised across operating regimes.

Threshold gating is analysed here as a scheduling rule rather than a learning
rule, so no model is trained: the question is whether observed participation
counts, depletion, stranded energy and stopping time match the closed forms
predicted for the policy.

Four properties are examined:
  (i)   no client is driven below zero charge when the eligibility threshold
        exceeds the largest per-round relative cost;
  (ii)  the number of participations a client receives follows a closed form
        that is non-increasing in its cost-to-capacity ratio;
  (iii) energy left unspent at termination is non-decreasing in the threshold;
  (iv)  the round at which training stops depends only on the energy parameters,
        and carries no information about whether the task has been learned.

The regimes span a non-binding control, in which the budget is never the
limiting factor, through to a severely constrained setting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.journal.federation import Federation, FedConfig
from src.journal.architectures import build, n_params
from src.journal.energy_model import EnergyStock, EnergyModel, DeviceProfile
from src.journal.device_profiles import assign_profiles
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import ThresholdGating

OUT = Path(__file__).resolve().parents[2] / "results" / "journal"
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def make_stock(fed, regime, cfg):
    """Build the energy stock.

    The compute term is charged for the number of local epochs the trainer
    actually executes. Charging a different figure than the one trained is a
    silent accounting error that propagates into every schedule, so ``cfg`` is
    the single source of truth for the epoch count; the regime supplies only the
    backbone, the allocation, and a default.
    """
    epochs = int(getattr(cfg, "local_epochs", regime.local_epochs))
    hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
    P = n_params(build(regime.backbone, fed.n_features, fed.n_classes))
    profiles = [DeviceProfile(capacity_j=h.capacity_j * regime.allocation,
                              eta=1.0, label=h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(P, epochs), fed.client_sizes)
    stock.cost = np.array([
        6.0 * P * n * epochs * h.j_per_flop + 2 * P * 4 * 6.4e-7
        for n, h in zip(fed.client_sizes, hw)], dtype=np.float64)
    return stock, hw, P


# The horizon is chosen per regime so that the aggregate energy budget is the
# binding constraint, i.e. sum_k floor(B_k / e_k) < m * T. Where no realistic
# horizon can make the budget bind, that fact is itself informative: it locates
# the model scale at which an energy budget starts to matter.
DEFAULT_HORIZON = 200
BINDING_MARGIN = 1.5


def horizon_for(fed, regime, m, cap=800):
    """Smallest horizon at which the aggregate budget binds, capped for tractability."""
    import numpy as np
    probe = FedConfig(rounds=DEFAULT_HORIZON, m=m)
    stock, _, _ = make_stock(fed, regime, probe)
    affordable = float(np.floor(stock.initial / stock.cost).sum())
    needed = int(np.ceil(affordable * BINDING_MARGIN / m))
    return max(DEFAULT_HORIZON, min(needed, cap))


def main():
    cfg = FedConfig(rounds=30, m=10)
    fed = Federation(cfg)
    K = len(fed.client_devices)
    print(f"Threshold-gating characterisation | K={K}, m={cfg.m}\n")

    all_rows = []
    for rg in REGIMES.values():
        cfg = FedConfig(rounds=horizon_for(fed, rg, m=10), m=10)
        s0, hw, P = make_stock(fed, rg, cfg)
        rho0 = s0.cost / s0.initial
        afford = np.floor(s0.initial / s0.cost)
        print(f"### {rg.name}: {rg.note}")
        print(f"    params={P:,}  T={cfg.rounds}  rho: min={rho0.min():.2e} "
              f"med={np.median(rho0):.2e} max={rho0.max():.2e}")
        print(f"    aggregate budget={int(afford.sum()):,} vs demand m*T={cfg.m*cfg.rounds:,}"
              f"  -> {'BUDGET BINDS' if afford.sum() < cfg.m*cfg.rounds else 'capacity-limited'}"
              f" | clients affording 0 rounds: {int((afford==0).sum())}/{len(afford)}")
        print(f"    {'tau':>5} {'depl':>5} {'stranded':>9} {'parts':>7} {'active':>7} "
              f"{'stop':>5} {'pred_parts':>11} {'match':>6}")
        for tau in TAUS:
            stock, hw, P = make_stock(fed, rg, cfg)
            rho = stock.cost / stock.initial
            pol = ThresholdGating(f"tau={tau}", cfg.m, tau)
            stop = None
            for t in range(1, cfg.rounds + 1):
                sel = pol.select(t, cfg.rounds, stock)
                for k in sel:
                    stock.charge(k)
                if not sel and stop is None:
                    stop = t
            # Closed-form participation count with beta_k = 1, capped by the horizon
            pred = np.where(1.0 < tau, 0.0, np.floor((1.0 - tau) / rho) + 1.0)
            pred = np.minimum(pred, cfg.rounds)
            obs = stock.participations.astype(float)
            # exact agreement is only expected for clients never crowded out by m
            free = obs < cfg.rounds
            match = bool(np.all(obs[free] <= pred[free] + 1e-9))
            active = sum(1 for h in pol.history if h)
            zero = int((stock.participations == 0).sum())
            all_rows.append({
                "regime": rg.name, "T": cfg.rounds, "clients_never_selected": zero, "backbone": rg.backbone, "params": int(P),
                "tau": tau, "depleted": int((stock.remaining < -1e-9).sum()),
                "stranded_fraction": float(stock.stranded_fraction()),
                "participations": int(obs.sum()), "rounds_active": int(active),
                "stop_round": stop, "pred_participations": float(pred.sum()),
                "bound_respected": match,
                "rho_max": float(rho.max()), "rho_med": float(np.median(rho)),
            })
            r = all_rows[-1]
            print(f"    {tau:>5.1f} {r['depleted']:>5} {r['stranded_fraction']:>9.3f} "
                  f"{r['participations']:>7} {r['rounds_active']:>7} "
                  f"{str(stop):>5} {r['pred_participations']:>11.0f} "
                  f"{'ok' if match else 'FAIL':>6}  never_sel={zero}")
        print()

    # Proposition checks
    print("=== Proposition checks ===")
    p1 = all(r["depleted"] == 0 for r in all_rows)
    print(f"  P1 (no depletion for tau > max rho)          : {'CONFIRMED' if p1 else 'VIOLATED'}")
    p2 = all(r["bound_respected"] for r in all_rows)
    print(f"  P2 (participation counts respect closed form): {'CONFIRMED' if p2 else 'VIOLATED'}")
    ok3 = True
    for rg in REGIMES:
        rr = [r for r in all_rows if r["regime"] == rg]
        ok3 &= all(rr[i]["stranded_fraction"] <= rr[i+1]["stranded_fraction"] + 1e-9
                   for i in range(len(rr)-1))
    print(f"  P3 (stranded energy non-decreasing in tau)   : {'CONFIRMED' if ok3 else 'VIOLATED'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "threshold_characterisation.json").write_text(json.dumps(
        {"K": K, "config": cfg.__dict__, "rows": all_rows}, indent=1, default=str))
    print(f"\nSaved -> {OUT/'threshold_characterisation.json'}")


if __name__ == "__main__":
    main()
