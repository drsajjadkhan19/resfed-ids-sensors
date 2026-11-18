"""Evidence for the ResFed-IDS extension.

Run from code/:
    python -m src.journal.study_twolevel_preservation A   # clean vs full federation
    python -m src.journal.study_twolevel_preservation B   # energy-model misspecification

Experiment A: clean federation (attack-source sbc partitions removed from the
client pool, test split unchanged) and full federation, binding and slack, w=2,
ten splits. Arms: FedAvg (capacity m), ResFed-IDS gate, gate + width guard,
gate + paced scheduler, DRR, Random-m. Records accuracy and preservation.

Experiment B: two-level robustness. The server tracks state of charge with the
nominal cost model; the true cost is kappa x nominal x U[0.9, 1.1]. Server-only
gating trusts its estimate; the two-level design adds the conference paper's
client-side self-check on the measured charge. Records depletion events,
reserve violations, refusals and terminal macro-F1.
"""
import json, sys
from pathlib import Path
import numpy as np
from src.journal.runtime import workers
import torch
torch.set_num_threads(1)
from src.journal.federation import Federation, FedConfig, evaluate
from src.journal.federation import _train_local
from src.journal.architectures import build
from src.journal.operating_regimes import REGIMES
from src.journal.study_threshold_characterisation import make_stock
from src.journal.device_profiles import assign_profiles
from src.journal.participation_policies import (FedAvgFull, ThresholdGating, GatedDeficit,
                                                DeficitRoundRobin, RandomM, MinSelected)
import src.journal.study_policy_comparison as spc

TAU = 0.3
OUT = Path(__file__).resolve().parents[2] / "results" / "journal"


def build_fed(seed, clean, w=2):
    cfg = FedConfig(rounds=40, m=12, max_rows_per_client=1200, local_epochs=w)
    cfg.seed = seed
    fed = Federation(cfg)
    if clean:
        hw = assign_profiles([fed.device_names[d] for d in fed.client_devices])
        cfg.exclude_devices = tuple(int(d) for d, h in zip(fed.client_devices, hw)
                                    if h.name == "sbc")
        fed = Federation(cfg)
    return fed, cfg


def run(fed, cfg, regime, pol, seed, cw):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    stock, hw, _ = make_stock(fed, regime, cfg)
    model = build(regime.backbone, fed.n_features, fed.n_classes)
    Xe, ye = fed.eval_xy()
    f1s, widths, last = [], [], 0
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, stock, None)
        ups = []
        for k in sel:
            Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
            loc = build(regime.backbone, fed.n_features, fed.n_classes)
            loc.load_state_dict(model.state_dict())
            _train_local(loc, Xb, yb, cfg, dev, cw)
            ups.append((loc.state_dict(), len(yb)))
            stock.charge(k)
        if ups:
            tot = sum(n for _, n in ups)
            model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                   .to(ups[0][0][q].dtype) for q in ups[0][0]})
            last = t; widths.append(len(sel))
        f1s.append(evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"])
    part = stock.participations > 0
    frac = stock.remaining / stock.initial
    return dict(terminal=f1s[-1], best=max(f1s), tail5=float(np.mean(f1s[-5:])),
                energy=float(stock.initial.sum() - stock.remaining.sum()),
                updates=int(stock.participations.sum()),
                stranded=float(stock.stranded_fraction()),
                n_participating=int(part.sum()),
                exhausted=int(np.sum(part & (stock.remaining < stock.cost))),
                below_10pct=int(np.sum(part & (frac < 0.10))),
                min_frac=float(frac[part].min()) if part.any() else 1.0,
                last_active=last, final_width=widths[-1] if widths else 0)


def arms(m, seed):
    return {
        "FedAvg": lambda: FedAvgFull("f", m),
        "ResFed": lambda: ThresholdGating("g", m, TAU),
        "ResFed+guard": lambda: MinSelected("gq", m, 3, ThresholdGating("g", m, TAU)),
        "ResFed+paced": lambda: GatedDeficit("gd", m, TAU),
        "DRR": lambda: DeficitRoundRobin("d", m),
        "Random-m": lambda: RandomM("r", m, seed=seed),
    }


def job_a(args):
    seed, clean, regime_name = args
    fed, cfg = build_fed(seed, clean)
    cw = spc.class_weights(fed)
    rg = REGIMES[regime_name]
    return [dict(exp="A", seed=seed, clean=clean, regime=regime_name, arm=a,
                 **run(fed, cfg, rg, mk(), seed, cw)) for a, mk in arms(cfg.m, seed).items()]


# ----------------------------------------------------------------- Exp B ----
class View:
    """What the server believes: remaining charge from the nominal cost model."""
    def __init__(self, initial, cost):
        self.initial = initial.copy(); self.remaining = initial.copy()
        self.cost = cost.copy(); self.participations = np.zeros(len(initial), int)
    def can_afford(self, k): return self.remaining[k] >= self.cost[k]
    def budget_fraction(self, k): return float(self.remaining[k] / self.initial[k])


def run_b(fed, cfg, regime, seed, cw, kappa, design, train=True):
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    stock, hw, _ = make_stock(fed, regime, cfg)
    true_rem = stock.initial.copy()
    view = View(stock.initial, stock.cost)
    pol = ThresholdGating("g", cfg.m, TAU)
    model = build(regime.backbone, fed.n_features, fed.n_classes) if train else None
    Xe, ye = fed.eval_xy() if train else (None, None)
    depleted = np.zeros(len(true_rem), bool)
    est = stock.cost.copy()                       # client's own cost estimate: nominal until measured
    events = refusals = 0
    f1 = None
    for t in range(1, cfg.rounds + 1):
        sel = pol.select(t, cfg.rounds, view, None)
        ups = []
        for k in sel:
            if depleted[k]:
                events += 1; continue
            actual = stock.cost[k] * kappa * rng.uniform(0.9, 1.1)
            literal = true_rem[k] / stock.initial[k] < TAU
            measured = literal or true_rem[k] < 1.1 * est[k]
            if (design == "two-level-literal" and literal) or (design == "two-level-measured" and measured):
                refusals += 1
                view.remaining[k] = true_rem[k]          # refusal reports measured charge
                continue
            if true_rem[k] < actual:                     # brown-out mid-training
                true_rem[k] = 0.0; depleted[k] = True; events += 1
                view.remaining[k] -= view.cost[k]; continue
            true_rem[k] -= actual; est[k] = actual       # the client measures what it spent
            view.remaining[k] -= view.cost[k]; view.participations[k] += 1
            if train:
                Xb, yb = fed.xy(fed.client_idx[fed.client_devices[k]])
                loc = build(regime.backbone, fed.n_features, fed.n_classes)
                loc.load_state_dict(model.state_dict())
                _train_local(loc, Xb, yb, cfg, dev, cw)
                ups.append((loc.state_dict(), len(yb)))
        if train and ups:
            tot = sum(n for _, n in ups)
            model.load_state_dict({q: sum(sd[q].double() * (n / tot) for sd, n in ups)
                                   .to(ups[0][0][q].dtype) for q in ups[0][0]})
    if train:
        f1 = evaluate(model, Xe, ye, dev, fed.n_classes)["macro_f1"]
    part = view.participations > 0
    frac = true_rem / stock.initial
    return dict(terminal=f1, depleted=int(depleted.sum()), events=events,
                refusals=refusals, updates=int(view.participations.sum()),
                n_participating=int(part.sum()),
                min_frac=float(frac[part | depleted].min()) if (part | depleted).any() else 1.0,
                energy=float((stock.initial - true_rem).sum()))


def job_b(args):
    seed, regime_name, train = args
    fed, cfg = build_fed(seed, clean=True)
    cw = spc.class_weights(fed)
    rg = REGIMES[regime_name]
    out = []
    kappas = [1.0, 1.5, 2.0] if train else [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for kappa in kappas:
        for design in ("server-only", "two-level-literal", "two-level-measured"):
            out.append(dict(exp="B", seed=seed, regime=regime_name, kappa=kappa,
                            design=design, train=train,
                            **run_b(fed, cfg, rg, seed, cw, kappa, design, train)))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    which = sys.argv[1]
    if which == "A":
        jobs = [(s, c, r) for c in (True, False) for r in ("R2f-bind", "R2f-slack")
                for s in range(1, 11)]
        fn = job_a
    else:
        jobs = ([(s, r, False) for r in ("R2f-bind", "R2f-slack") for s in range(1, 11)]
                + [(s, "R2f-bind", True) for s in range(1, 11)])
        fn = job_b
    with Pool(workers()) as pool:
        rows = [r for chunk in pool.map(fn, jobs) for r in chunk]
    name = {"A": "clean_vs_full_federation", "B": "cost_misspecification"}[which]
    path = OUT / f"twolevel_preservation_{name}.json"
    json.dump(rows, open(path, "w"), indent=1)
    print("saved", path, len(rows))
