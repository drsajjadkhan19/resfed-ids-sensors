#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPE -- Cost-Amortized Participation with Evidence-weighted aggregation.

Three mechanisms, each addressing a distinct failure mode of budget-constrained
federated averaging, and each separately ablatable.

-------------------------------------------------------------------------------
FAILURE 1 -- thin-round overwrite.
Under a finite energy stock the number of clients per round is highly irregular
(measured: 0 to 12 out of 12). Vanilla FedAvg REPLACES the global model with the
average of whoever happened to participate, so a round with one client makes the
global model that client's model. Measured consequence: at work quantum w = 8 the
last active round had 1 client and macro-F1 collapsed to 0.0325 +- 0.0000 across
all seeds and both paced policies.

MECHANISM E -- evidence-weighted aggregation.
    w_global <- (1 - lambda_t) * w_global + lambda_t * FedAvg(S_t)
    lambda_t  = min(1, mass(S_t) / M_ref),  mass(S_t) = sum_{k in S_t} n_k
A round moves the global model in proportion to the evidence it carries, which
is a trust region on the aggregation step sized by round mass. Federated
averaging is the special case lambda_t = 1, corresponding to the assumption that
all rounds carry comparable evidence -- an assumption a finite energy budget
invalidates.

-------------------------------------------------------------------------------
Problem 2 -- communication-dominated cost spent on small local steps.
Per-participation cost is e_k(w) = c_k + a_k * w, where the communication term
c_k does not depend on w. In an embedded setting c_k dominates, so the work
deliverable for a fixed budget, w * floor(B_k / (c_k + a_k w)), rises by an order
of magnitude as w grows. Schedulers in this area fix w and optimise only which
clients participate.

Mechanism C -- per-client cost-amortized work quantum.
    w_k* = clip( c_k / a_k , w_min , w_max )
The rule spends on compute as much as the client is already obliged to spend on
communication. At
w = c/a the two terms are equal and the communication overhead ratio is exactly
2; below it the client is paying mostly for the privilege of talking. The rule is
per-client, so clients with cheap compute do proportionally more local work --
heterogeneity is exploited rather than merely tolerated.

-------------------------------------------------------------------------------
Problem 3 -- client drift grows with the local quantum.
A larger w moves clients further from the global model before averaging, which is
the counterweight to Mechanism C under non-identically distributed data.

Mechanism P -- work-proportional proximal control.
    mu_k = mu0 * (w_k / w_ref)
A proximal term whose strength is derived from the client's own energy-allocated
work, so drift control scales with the amount of local optimisation the budget
affords. The proximal formulation follows FedProx (Li et al., MLSys 2020).

-------------------------------------------------------------------------------
Pacing follows deficit round robin (Shreedhar & Varghese, 1996). Under
heterogeneous stocks, service proportional to capacity outperforms service
equalised by participation count: a client holding ten times the budget should
receive roughly ten times the service, which an equal-count rule prevents.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  MECHANISM C -- cost-amortized work quantum
# --------------------------------------------------------------------------- #
def amortized_quantum(comm_cost: np.ndarray, compute_per_unit: np.ndarray,
                      w_min: int = 1, w_max: int = 16) -> np.ndarray:
    """w_k* = clip(c_k / a_k, w_min, w_max), rounded to whole local epochs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(compute_per_unit > 0, comm_cost / compute_per_unit, w_max)
    return np.clip(np.rint(w), w_min, w_max).astype(int)


# --------------------------------------------------------------------------- #
#  MECHANISM E -- evidence-weighted aggregation
# --------------------------------------------------------------------------- #
class EvidenceAggregator:
    """Global-model update whose step size scales with the round's mass.

    ``m_ref`` is the mass at which a round is trusted fully. It is derived from
    the federation's own scale -- by default the mass of a round at full server
    capacity -- and is not fitted to evaluation data.
    """

    def __init__(self, m_ref: float, floor: float = 0.0):
        self.m_ref = float(max(m_ref, 1e-9))
        self.floor = float(floor)
        self.last_lambda = None

    def step(self, global_sd: Dict[str, torch.Tensor],
             updates: List[tuple], round_mass: float) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_sd
        tot = sum(n for _, n in updates)
        avg = {q: sum(sd[q].double() * (n / tot) for sd, n in updates)
               for q in updates[0][0]}
        lam = min(1.0, max(self.floor, round_mass / self.m_ref))
        self.last_lambda = lam
        return {q: ((1.0 - lam) * global_sd[q].double() + lam * avg[q])
                .to(global_sd[q].dtype) for q in global_sd}


# --------------------------------------------------------------------------- #
#  MECHANISM P -- work-proportional proximal local training
# --------------------------------------------------------------------------- #
def train_local_prox(model: nn.Module, Xb, yb, epochs: int, lr: float,
                     batch_size: int, device, class_weight=None,
                     mu: float = 0.0, global_params: Optional[List] = None):
    """Local training with an optional FedProx term of strength ``mu``.

    The proximal formulation follows FedProx (Li, Sahu, Zaheer, Sanjabi,
    Talwalkar & Smith, MLSys 2020); here ``mu`` is set from the client's
    energy-allocated work quantum.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = (torch.tensor(class_weight, dtype=torch.float32, device=device)
         if class_weight is not None else None)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Xb, device=device)
    yt = torch.tensor(yb, dtype=torch.long, device=device)
    n = len(yt)
    gp = ([p.detach().clone() for p in global_params]
          if (mu > 0 and global_params is not None) else None)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            if gp is not None:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), gp))
                loss = loss + 0.5 * mu * prox
            loss.backward()
            opt.step()
    return model
