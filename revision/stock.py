"""The two-level stock of ``src.journal.two_level``, with the relaxations of Corollary 2.

``RevStock`` adds to ``DualStock``:

* ``kappa_vec``: a misspecification factor per client instead of one factor;
* ``error_on="compute"``: the error applies to the compute term only, so it grows
  with the local work, while the communication term is exact;
* ``gamma`` and ``meter``: the client reads its charge with an additive error of
  at most ``gamma`` times its allocation, drawn uniformly (``"noisy"``) or always
  at the upper end (``"high"``, the worst case for the bound);
* ``refusal_cost``: the energy a refusal costs the client, in joules;
* ``server_margin``: the server's filter requires the same margin as the client.
"""
from __future__ import annotations

import numpy as np

from src.journal.energy_model import BYTES_PER_PARAM, E_TX_J_PER_BYTE
from src.journal.two_level import DualStock


class RevStock(DualStock):
    def __init__(self, base, n_params: int, kappa: float = 1.0, noise: float = 0.0,
                 seed: int = 0, kappa_vec=None, error_on: str = "total",
                 gamma: float = 0.0, meter: str = "noisy", refusal_cost=0.0,
                 server_margin: float = 0.0):
        super().__init__(base, kappa=kappa, noise=noise, seed=seed)
        K = len(self.initial)
        self.kappa_vec = (np.full(K, float(kappa)) if kappa_vec is None
                          else np.asarray(kappa_vec, dtype=float))
        self.comm = np.full(K, 2.0 * n_params * BYTES_PER_PARAM * E_TX_J_PER_BYTE)
        self.comp = np.maximum(self.cost - self.comm, 0.0)
        self.error_on = error_on
        self.gamma = float(gamma)
        self.meter = meter
        self.refusal_cost = np.broadcast_to(np.asarray(refusal_cost, dtype=float), (K,)).copy()
        self.server_margin = float(server_margin)
        self.meter_rng = np.random.default_rng(seed + 7919)
        self.refusal_spent = np.zeros(K)

    # ------------------------------------------------- what the server sees
    def can_afford(self, k: int) -> bool:
        if self.server_drops_refusers and self.has_refused[k]:
            return False
        return self.remaining[k] >= (1.0 + self.server_margin) * self.cost[k]

    # ------------------------------------------------------- the true cost
    def draw(self, k: int) -> float:
        j = 1.0 if self.noise <= 0 else float(self.rng.uniform(1 - self.noise, 1 + self.noise))
        if self.error_on == "compute":
            return float(self.comm[k] + self.kappa_vec[k] * self.comp[k] * j)
        return float(self.cost[k] * self.kappa_vec[k] * j)

    def worst_excess(self, k: int, delta: float) -> float:
        """Delta_k of Equation 6 for this error model."""
        hi = 1.0 + max(self.noise, 0.0)
        if self.error_on == "compute":
            worst = self.comm[k] + self.kappa_vec[k] * self.comp[k] * hi
        else:
            worst = self.cost[k] * self.kappa_vec[k] * hi
        return max(worst - (1.0 + delta) * self.cost[k], 0.0)

    # ---------------------------------------------------- what the client reads
    def reading(self, k: int) -> float:
        G = self.gamma * self.initial[k]
        if G <= 0:
            return float(self.true_remaining[k])
        g = G if self.meter == "high" else float(self.meter_rng.uniform(-G, G))
        return float(self.true_remaining[k] + g)

    def refuse_paid(self, k: int) -> None:
        r = float(self.refusal_cost[k])
        self.true_remaining[k] -= r
        self.refusal_spent[k] += r
        self.refuse(k)

    def bound(self, delta: float) -> np.ndarray:
        """Corollary 2: the lowest charge each client can reach, in joules."""
        K = len(self.initial)
        return np.array([-(self.worst_excess(k, delta) + self.gamma * self.initial[k])
                         - self.refusal_cost[k] for k in range(K)])


def run_round_rev(policy, t, T, stock: RevStock, delta: float, client_check: bool):
    """One round of Definitions 1 and 2 with meter error and paid refusals."""
    selected = policy.select(t, T, stock, None)
    trained = []
    for k in selected:
        if stock.true_remaining[k] <= 0.0:
            stock.refuse(k)            # a flat device cannot answer, and spends nothing
            continue
        if client_check and stock.reading(k) < (1.0 + delta) * stock.cost[k]:
            stock.refuse_paid(k)
            continue
        stock.charge(k, stock.draw(k))
        trained.append(k)
    return trained
