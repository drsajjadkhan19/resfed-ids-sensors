"""Controls for the 'service order' contrast (development runs, superseded by
study_order_only.py and study_order_tier_decoupling.py).

Run from code/:  python -m src.journal.study_order_controls

Runs, on the binding regime, w=2, tau=0.3, ten splits:
  published        ThresholdGating (as published)
  gated_deficit    GatedDeficit (the article's arm: gate + DRR admission + DRR order)
  prio_deficit     gate unchanged, priority by accrued deficit (J), nothing withheld
  prio_credit      gate unchanged, priority by accrued deficit / cost, nothing withheld
  random_order     gate unchanged, random order among the eligible, nothing withheld
  gd_T30, gd_T60   GatedDeficit told the horizon is 30 or 60 instead of 40
and repeats published / gated_deficit with
  sbc->camera      every single-board-computer-tier device (Raspberry Pi / VMware)
                   given the camera tier's capacity and J/FLOP
"""
import json
from pathlib import Path
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)

from src.journal.federation import Federation, FedConfig
from src.journal.operating_regimes import REGIMES
from src.journal.participation_policies import Policy, ThresholdGating, GatedDeficit
from src.journal.device_profiles import assign_profiles, HW_CLASSES
from src.journal.energy_model import EnergyStock, EnergyModel, DeviceProfile
from src.journal.architectures import build, n_params
import src.journal.study_policy_comparison as spc

ORIG_MAKE = spc.make_stock
OUT = Path(__file__).resolve().parents[2] / "results" / "journal"


def make_stock_sbc_to_camera(fed, regime, cfg):
    epochs = int(getattr(cfg, "local_epochs", regime.local_epochs))
    hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
    hw = [HW_CLASSES["camera"] if h.name == "sbc" else h for h in hw]
    P = n_params(build(regime.backbone, fed.n_features, fed.n_classes))
    profiles = [DeviceProfile(capacity_j=h.capacity_j * regime.allocation, eta=1.0,
                              label=h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(P, epochs), fed.client_sizes)
    stock.cost = np.array([6.0 * P * n * epochs * h.j_per_flop + 2 * P * 4 * 6.4e-7
                           for n, h in zip(fed.client_sizes, hw)], dtype=np.float64)
    return stock, hw, P


class GatePriority(Policy):
    """Gate unchanged; only the priority among eligible clients changes."""

    def __init__(self, name, m, tau, mode, seed=0):
        super().__init__(name, m)
        self.tau, self.mode = tau, mode
        self.deficit = None
        self.rng = np.random.default_rng(seed)

    def select(self, t, T, stock, util=None):
        K = len(stock.remaining)
        if self.deficit is None:
            self.deficit = np.zeros(K)
        self.deficit += stock.initial / T
        elig = [k for k in range(K)
                if stock.budget_fraction(k) >= self.tau and stock.can_afford(k)]
        if self.mode == "deficit":
            elig.sort(key=lambda k: (-self.deficit[k], stock.participations[k], k))
        elif self.mode == "credit":
            elig.sort(key=lambda k: (-self.deficit[k] / stock.cost[k],
                                     stock.participations[k], k))
        elif self.mode == "random":
            elig = list(self.rng.permutation(elig))
        sel = [int(k) for k in elig[: self.m]]
        for k in sel:
            self.deficit[k] -= stock.cost[k]
        return self._record(sel)


class HorizonLie(Policy):
    def __init__(self, name, m, inner, T_claimed):
        super().__init__(name, m)
        self.inner, self.Tc = inner, T_claimed

    def select(self, t, T, stock, util=None):
        return self._record(self.inner.select(t, self.Tc, stock, util))


def arms(m, tau, seed):
    return {
        "published": lambda: ThresholdGating("p", m, tau),
        "gated_deficit": lambda: GatedDeficit("gd", m, tau),
        "prio_deficit": lambda: GatePriority("pd", m, tau, "deficit"),
        "prio_credit": lambda: GatePriority("pc", m, tau, "credit"),
        "random_order": lambda: GatePriority("ro", m, tau, "random", seed=seed),
        "gd_T30": lambda: HorizonLie("t30", m, GatedDeficit("gd", m, tau), 30),
        "gd_T60": lambda: HorizonLie("t60", m, GatedDeficit("gd", m, tau), 60),
    }


def summarise(tr):
    traj = tr["traj"]
    active = [p for p in traj if p["width"] > 0]
    return dict(terminal=traj[-1]["macro_f1"], energy=tr["total_energy_j"],
                updates=tr["total_updates"], stranded=tr["stranded_fraction"],
                last_active=tr["last_active_round"], n_active=len(active),
                final_width=active[-1]["width"] if active else 0,
                empty_before_last=sum(1 for p in traj
                                      if p["round"] <= tr["last_active_round"]
                                      and p["width"] == 0))


def job(seed):
    cfg = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=2)
    cfg.seed = seed
    fed = Federation(cfg)
    cw = spc.class_weights(fed)
    rg = REGIMES["R2f-bind"]
    out = {}
    spc.make_stock = ORIG_MAKE
    for name, mk in arms(cfg.m, 0.3, seed).items():
        out[name] = summarise(spc.trajectory(fed, rg, cfg, mk(), seed, cw))
    spc.make_stock = make_stock_sbc_to_camera
    for name in ("published", "gated_deficit"):
        mk = arms(cfg.m, 0.3, seed)[name]
        out["sbc2cam_" + name] = summarise(spc.trajectory(fed, rg, cfg, mk(), seed, cw))
    spc.make_stock = ORIG_MAKE
    return seed, out


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(workers()) as pool:
        res = dict(pool.map(job, range(1, 11)))
    path = OUT / "order_controls_R2f-bind.json"
    json.dump(res, open(path, "w"), indent=1)
    print("saved", path)
