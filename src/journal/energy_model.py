#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-stock energy model for federated IoT clients.

Contrast with prior work
------------------------
The Lyapunov / virtual-queue literature models client energy as a *renewable
rate* (average power under a queue-stability constraint). Battery-powered IoT
nodes instead hold a **finite, irreversibly depleting stock**: a joule spent in
round t is permanently unavailable in every later round.

The published ResFed-IDS model drew battery as ``random.randint(5, 10)``
percentage points per round, independent of dataset size, model size, epochs or
hardware. This module replaces that with an analytical cost model whose terms
are functions of the actual work performed.

    E_k(n_k, P, e) = E_compute + E_comm
    E_compute      = c_train * P * n_k * e * eta_k
    E_comm         = 2 * P * bytes_per_param * eps_tx

where P is the parameter count, n_k the client's local sample count, e the
number of local epochs, and eta_k a per-device inefficiency factor.

This is an analytical model, not a hardware measurement. Its structure follows
common practice for energy-aware federated learning, in which compute energy is
estimated from operation counts and communication energy from payload size, but
its absolute magnitudes are not calibrated against instrumented devices and no
claim is made that they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------- constants --
# Energy per floating-point operation (J/FLOP). Order of magnitude follows
# OptiFLIDS (2.3 pJ/FLOP) for embedded-class arithmetic.
E_FLOP_J = 2.3e-12

# Energy per transmitted byte (J/byte). Low-power radio order of magnitude.
E_TX_J_PER_BYTE = 6.4e-7

# Training FLOPs per parameter per sample: forward ~2, backward ~2x forward.
C_TRAIN_FLOPS_PER_PARAM_PER_SAMPLE = 6.0

BYTES_PER_PARAM = 4  # float32


@dataclass
class DeviceProfile:
    """One IoT client's physical characteristics.

    Attributes
    ----------
    capacity_j : total usable energy stock in joules (finite, non-renewable)
    eta        : compute inefficiency multiplier (1.0 = reference class)
    label      : human-readable device class name
    """

    capacity_j: float
    eta: float = 1.0
    label: str = "generic"


# Reference device classes. Capacities are typical usable stocks for
# battery-powered nodes (e.g. 2000 mAh @ 3.7 V ~ 26.6 kJ), scaled to a
# deployment-realistic spread. eta encodes relative compute inefficiency.
DEVICE_CLASSES: Dict[str, DeviceProfile] = {
    "gateway": DeviceProfile(capacity_j=26_640.0, eta=1.0, label="gateway"),
    "node": DeviceProfile(capacity_j=13_320.0, eta=1.6, label="sensor node"),
    "mote": DeviceProfile(capacity_j=5_328.0, eta=2.5, label="low-power mote"),
}


@dataclass
class EnergyModel:
    """Analytical per-round energy cost for federated local training."""

    num_params: int
    local_epochs: int = 1
    e_flop_j: float = E_FLOP_J
    e_tx_j_per_byte: float = E_TX_J_PER_BYTE

    def compute_energy(self, n_samples: int, eta: float = 1.0) -> float:
        """Energy (J) for one round of local training on ``n_samples``."""
        flops = (
            C_TRAIN_FLOPS_PER_PARAM_PER_SAMPLE
            * self.num_params
            * n_samples
            * self.local_epochs
        )
        return flops * self.e_flop_j * eta

    def comm_energy(self) -> float:
        """Energy (J) for one download + one upload of the full model."""
        payload = self.num_params * BYTES_PER_PARAM
        return 2.0 * payload * self.e_tx_j_per_byte

    def round_cost(self, n_samples: int, eta: float = 1.0) -> float:
        """Total energy (J) drawn by one participation."""
        return self.compute_energy(n_samples, eta) + self.comm_energy()

    def payload_bytes(self) -> int:
        return self.num_params * BYTES_PER_PARAM


class EnergyStock:
    """Tracks the finite, irreversibly depleting energy stock of each client.

    Unlike the published ResFed-IDS ``ResourceManager``, drain is a deterministic
    function of the work actually performed, and there is no recovery: a stock
    only decreases.
    """

    def __init__(
        self,
        profiles: List[DeviceProfile],
        model: EnergyModel,
        client_sizes: List[int],
    ):
        assert len(profiles) == len(client_sizes)
        self.profiles = profiles
        self.model = model
        self.client_sizes = client_sizes
        self.initial = np.array([p.capacity_j for p in profiles], dtype=np.float64)
        self.remaining = self.initial.copy()
        self.participations = np.zeros(len(profiles), dtype=int)
        self.cost = np.array(
            [model.round_cost(n, p.eta) for n, p in zip(client_sizes, profiles)],
            dtype=np.float64,
        )

    # ---------------------------------------------------------------- queries
    def can_afford(self, k: int) -> bool:
        """True if client k can pay for one more full round."""
        return self.remaining[k] >= self.cost[k]

    def affordable_rounds(self, k: int) -> int:
        """How many further participations client k's stock can fund."""
        return int(np.floor(self.remaining[k] / self.cost[k]))

    def budget_fraction(self, k: int) -> float:
        """Remaining stock as a fraction of the client's initial stock."""
        return float(self.remaining[k] / self.initial[k]) if self.initial[k] else 0.0

    # ---------------------------------------------------------------- updates
    def charge(self, k: int) -> float:
        """Deduct one round's cost from client k. Returns the amount drawn."""
        drawn = float(self.cost[k])
        self.remaining[k] = max(0.0, self.remaining[k] - drawn)
        self.participations[k] += 1
        return drawn

    # --------------------------------------------------------------- summary
    def stranded_energy(self) -> float:
        """Total energy left unused across all clients (J)."""
        return float(self.remaining.sum())

    def stranded_fraction(self) -> float:
        """Unused share of the total initial stock."""
        return float(self.remaining.sum() / self.initial.sum())

    def summary(self) -> Dict[str, object]:
        return {
            "initial_total_j": float(self.initial.sum()),
            "remaining_total_j": float(self.remaining.sum()),
            "stranded_fraction": self.stranded_fraction(),
            "participations": self.participations.tolist(),
            "per_client_cost_j": self.cost.tolist(),
            "per_client_remaining_j": self.remaining.tolist(),
            "per_client_initial_j": self.initial.tolist(),
            "depleted_clients": int(np.sum(self.remaining < self.cost)),
            "max_affordable_rounds": [
                int(np.floor(i / c)) for i, c in zip(self.initial, self.cost)
            ],
        }


# --------------------------------------------------------------------------- #
#  Composition helper
# --------------------------------------------------------------------------- #
def build_client_stock(federation, regime, cfg, work_quantum=None,
                       compute_scale: float = 1.0, comms_scale: float = 1.0):
    """Construct the per-client energy stock for a federation and regime.

    The compute term is charged for the number of local epochs the trainer
    actually executes. Charging a different figure than the one trained is a
    silent accounting error that propagates into every schedule, so ``cfg`` is
    the single source of truth for the epoch count unless an explicit
    per-client ``work_quantum`` is supplied.

    Parameters
    ----------
    federation : Federation
    regime : Regime
        Supplies the backbone and the fraction of device capacity budgeted to
        federated training.
    cfg : FedConfig
    work_quantum : array-like of int, optional
        Per-client local-epoch counts. When omitted, every client uses
        ``cfg.local_epochs``.
    compute_scale, comms_scale : float
        Multipliers on the compute and communication terms, used to test how
        sensitive a conclusion is to error in the cost model's constants.

    Returns
    -------
    (EnergyStock, list[HwClass], int)
        The stock, the per-client hardware classes, and the parameter count.
    """
    import numpy as np
    from src.journal.architectures import build, n_params
    from src.journal.device_profiles import assign_profiles

    epochs = int(getattr(cfg, "local_epochs", regime.local_epochs))
    hw = assign_profiles([federation.device_names[d]
                          for d in federation.client_devices])
    n_par = n_params(build(regime.backbone, federation.n_features,
                           federation.n_classes))
    sizes = np.asarray(federation.client_sizes, dtype=float)
    w = (np.full(len(hw), epochs, dtype=float) if work_quantum is None
         else np.asarray(work_quantum, dtype=float))

    profiles = [DeviceProfile(capacity_j=h.capacity_j * regime.allocation,
                              eta=1.0, label=h.name) for h in hw]
    stock = EnergyStock(profiles, EnergyModel(n_par, epochs),
                        federation.client_sizes)
    stock.cost = np.array([
        C_TRAIN_FLOPS_PER_PARAM_PER_SAMPLE * n_par * s * wi * h.j_per_flop * compute_scale
        + 2 * n_par * BYTES_PER_PARAM * E_TX_J_PER_BYTE * comms_scale
        for s, wi, h in zip(sizes, w, hw)], dtype=np.float64)
    return stock, hw, n_par
