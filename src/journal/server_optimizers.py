#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server-side aggregation rules used as identification controls.

Evidence-weighted aggregation damps the global step by the participating set's
sample mass. Damping alone is not evidence of a new aggregation principle: a
constant relaxation coefficient, a decaying schedule, or server momentum all
damp the step too. These rules isolate what the mass signal contributes over
and above ordinary server damping, so they are stated in the same interface as
``EvidenceAggregator`` and are interchangeable with it inside a run.

The interface is a single call

    new_state = rule.step(global_state, updates, round_mass)

where ``updates`` is a list of ``(state_dict, n_samples)`` pairs. All rules
operate on the sample-weighted client average, so they differ only in how far
the global model is moved toward it.

References
----------
Server momentum and adaptive server optimisers follow the federated-optimisation
formulation of Reddi, Charles, Zaheer, Garrett, Rush, Konecny, Kumar and
McMahan, *Adaptive Federated Optimization*, ICLR 2021.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


def weighted_average(updates: List[tuple]) -> Dict[str, torch.Tensor]:
    """Sample-weighted average of client states, in double precision."""
    total = sum(n for _, n in updates)
    keys = updates[0][0]
    return {q: sum(sd[q].double() * (n / total) for sd, n in updates) for q in keys}


class _Relaxation:
    """Base class for rules of the form theta <- theta + lam * (avg - theta)."""

    def __init__(self):
        self.round = 0
        self.last_lambda: Optional[float] = None

    def coefficient(self, round_mass: float) -> float:
        raise NotImplementedError

    def step(self, global_sd, updates, round_mass):
        if not updates:
            return global_sd
        self.round += 1
        avg = weighted_average(updates)
        lam = self.coefficient(round_mass)
        self.last_lambda = lam
        return {q: ((1.0 - lam) * global_sd[q].double() + lam * avg[q])
                .to(global_sd[q].dtype) for q in global_sd}


class FixedRelaxation(_Relaxation):
    """Constant server step size, independent of who participated.

    Separates the benefit of damping the aggregation from the benefit of sizing
    that damping by the round's evidence.
    """

    def __init__(self, lam: float):
        super().__init__()
        self.lam = float(lam)

    def coefficient(self, round_mass: float) -> float:
        return self.lam


class DecayRelaxation(_Relaxation):
    """Server step size decaying as lam0 / (1 + t / tau), mass-agnostic.

    A schedule that shrinks late-round steps reproduces one side effect of
    mass weighting -- small steps once the budget is nearly exhausted -- without
    reading participation at all.
    """

    def __init__(self, lam0: float = 1.0, tau: float = 10.0):
        super().__init__()
        self.lam0, self.tau = float(lam0), float(tau)

    def coefficient(self, round_mass: float) -> float:
        return self.lam0 / (1.0 + self.round / self.tau)


class ServerMomentum:
    """FedAvgM: server momentum on the aggregate pseudo-gradient.

    theta <- theta + eta * v,  v <- beta * v + (avg - theta)
    """

    def __init__(self, beta: float = 0.9, eta: float = 1.0):
        self.beta, self.eta = float(beta), float(eta)
        self.v: Optional[Dict[str, torch.Tensor]] = None
        self.last_lambda = None

    def step(self, global_sd, updates, round_mass):
        if not updates:
            return global_sd
        avg = weighted_average(updates)
        delta = {q: avg[q] - global_sd[q].double() for q in global_sd}
        if self.v is None:
            self.v = {q: torch.zeros_like(delta[q]) for q in delta}
        self.v = {q: self.beta * self.v[q] + delta[q] for q in delta}
        return {q: (global_sd[q].double() + self.eta * self.v[q])
                .to(global_sd[q].dtype) for q in global_sd}


class ServerAdam:
    """Adam-style adaptive server optimiser on the aggregate pseudo-gradient.

    Follows the FedAdam formulation of Reddi et al. (ICLR 2021), with the
    standard bias correction on both moment estimates. It is referred to as an
    Adam-style server optimiser rather than as FedAdam wherever the distinction
    could matter, since the hyper-parameters here are selected on the same
    validation budget as the rules it is compared against, not taken from that
    paper's tuned settings.
    """

    def __init__(self, eta: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.99, eps: float = 1e-3):
        self.eta, self.b1, self.b2, self.eps = eta, beta1, beta2, eps
        self.m: Optional[Dict[str, torch.Tensor]] = None
        self.v: Optional[Dict[str, torch.Tensor]] = None
        self.t = 0
        self.last_lambda = None

    def step(self, global_sd, updates, round_mass):
        if not updates:
            return global_sd
        avg = weighted_average(updates)
        delta = {q: avg[q] - global_sd[q].double() for q in global_sd}
        if self.m is None:
            self.m = {q: torch.zeros_like(delta[q]) for q in delta}
            self.v = {q: torch.zeros_like(delta[q]) for q in delta}
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        out = {}
        for q in global_sd:
            self.m[q] = self.b1 * self.m[q] + (1 - self.b1) * delta[q]
            self.v[q] = self.b2 * self.v[q] + (1 - self.b2) * delta[q] ** 2
            m_hat = self.m[q] / bc1
            v_hat = self.v[q] / bc2
            out[q] = (global_sd[q].double()
                      + self.eta * m_hat / (v_hat.sqrt() + self.eps)
                      ).to(global_sd[q].dtype)
        return out


class MassFloorAggregator:
    """Withhold the global update until a round carries enough evidence.

    Motivated by a measured failure rather than by an analogy: under a finite
    energy stock the number of participants per round is irregular, and a round
    carrying one client replaces the global model with that client's model. The
    earlier response to this was to scale the server step by round mass, which
    was withdrawn when a constant step was shown to beat it. This rule takes the
    other option: a round below the mass floor does not move the global model at
    all, and its client updates are held and merged into the next round that
    qualifies.

    Every parameter below is fixed before any run, because each is a degree of
    freedom that could otherwise be chosen to flatter the result.

    floor_frac      mass floor as a fraction of a full-capacity round's mass
    max_delay       a buffer older than this is flushed regardless of mass, so
                    held work cannot be postponed indefinitely
    stale_decay     a buffered update was trained against an older global model;
                    its weight is multiplied by ``stale_decay ** age`` at flush

    Two properties matter for the honesty of the comparison. Clients selected in
    a withheld round still train and are still charged for it by the caller, so
    the floor buys stability with energy rather than by discarding expensive
    rounds for free. And ``flush`` is called at the horizon, so held work is
    applied rather than silently dropped.
    """

    def __init__(self, m_ref: float, floor_frac: float = 0.5,
                 max_delay: int = 3, stale_decay: float = 0.5):
        self.m_ref = float(max(m_ref, 1e-9))
        self.floor = float(floor_frac) * self.m_ref
        self.max_delay = int(max_delay)
        self.stale_decay = float(stale_decay)
        self.buffer: List[tuple] = []          # (state_dict, n, age_in_flushes)
        self.buffered_mass = 0.0
        self.age = 0
        self.withheld = 0                      # rounds withheld, for reporting
        self.flushes = 0
        self.last_lambda = None

    def _apply(self, global_sd, pool):
        total = sum(w for _, w in pool)
        if total <= 0:
            return global_sd
        keys = pool[0][0]
        avg = {q: sum(sd[q].double() * (w / total) for sd, w in pool) for q in keys}
        self.flushes += 1
        return {q: avg[q].to(global_sd[q].dtype) for q in global_sd}

    def step(self, global_sd, updates, round_mass):
        if not updates:
            return global_sd
        qualifies = (round_mass + self.buffered_mass) >= self.floor
        forced = self.age >= self.max_delay and self.buffer
        if not (qualifies or forced):
            # hold: the global model does not move on under-evidenced rounds
            self.buffer.extend((sd, float(n), 0) for sd, n in updates)
            self.buffered_mass += float(round_mass)
            self.age += 1
            self.withheld += 1
            return global_sd

        pool = [(sd, float(n)) for sd, n in updates]
        pool += [(sd, n * (self.stale_decay ** (a + 1))) for sd, n, a in self.buffer]
        self.buffer, self.buffered_mass, self.age = [], 0.0, 0
        return self._apply(global_sd, pool)

    def flush(self, global_sd):
        """Apply any held updates at the horizon so no paid-for work is dropped."""
        if not self.buffer:
            return global_sd
        pool = [(sd, n * (self.stale_decay ** (a + 1))) for sd, n, a in self.buffer]
        self.buffer, self.buffered_mass, self.age = [], 0.0, 0
        return self._apply(global_sd, pool)


class BufferedAggregator:
    """Buffered aggregation in the manner of FedBuff, as a baseline for the floor.

    Updates accumulate until ``k`` of them are held, then a single global update
    is applied. Unlike the mass floor the trigger counts *updates*, not evidence,
    and there is no staleness weighting -- which is precisely the comparison of
    interest: does sizing the trigger by sample mass, and discounting stale work,
    buy anything over plain buffering?

    Follows Nguyen, Malik, Zhan, Yousefpour, Rabbat, Malek and Huba,
    *Federated Learning with Buffered Asynchronous Aggregation*, AISTATS 2022.
    """

    def __init__(self, k: int = 12):
        self.k = int(max(k, 1))
        self.buffer: List[tuple] = []
        self.last_lambda = None

    def step(self, global_sd, updates, round_mass):
        if not updates:
            return global_sd
        self.buffer.extend((sd, float(n)) for sd, n in updates)
        if len(self.buffer) < self.k:
            return global_sd
        pool, self.buffer = self.buffer, []
        total = sum(w for _, w in pool)
        keys = pool[0][0]
        avg = {q: sum(sd[q].double() * (w / total) for sd, w in pool) for q in keys}
        return {q: avg[q].to(global_sd[q].dtype) for q in global_sd}

    def flush(self, global_sd):
        if not self.buffer:
            return global_sd
        pool, self.buffer = self.buffer, []
        total = sum(w for _, w in pool)
        keys = pool[0][0]
        avg = {q: sum(sd[q].double() * (w / total) for sd, w in pool) for q in keys}
        return {q: avg[q].to(global_sd[q].dtype) for q in global_sd}
