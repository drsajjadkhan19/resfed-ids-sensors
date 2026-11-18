#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The two-level participation control of ResFed-IDS, with the levels separated.

The published policy guards participation twice. The server filters on the
state of charge it believes a client to have, and the client, once selected,
checks its own measured state before training and refuses the task if it cannot
afford it. The journal implementation carried only the first level, which made
every guarantee in this article conditional on the server's energy model being
right.

Separating the two requires separating what the server believes from what is
true. ``DualStock`` holds both: a nominal stock that the server reads and
charges with its own cost model, and a true stock charged with the real cost.
When the two models agree the behaviour is the single-level one. When the
server's model underestimates by a factor kappa, only the client-side check
stands between the deployment and a flat battery.

The experiment this supports is a sweep over kappa. A server-only filter must
deplete clients once kappa exceeds a value the theory predicts from the
participation count, and the shortfall then grows with every further round. A
client that checks its own charge can be overdrawn by at most one
participation's excess over the margin it reserves, whatever kappa is, and by
nothing at all while that margin covers the error.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class DualStock:
    """A nominal stock the server reads, and the true stock that is spent.

    ``kappa`` is the factor by which the true per-participation cost exceeds the
    server's estimate. ``noise`` adds a multiplicative jitter to the true cost of
    each individual participation, so that the misspecification is not a clean
    constant the client could trivially invert.
    """

    def __init__(self, base, kappa: float = 1.0, noise: float = 0.0, seed: int = 0):
        self.initial = base.initial.copy()
        self.cost = base.cost.copy()             # the server's estimate
        self.remaining = base.initial.copy()     # what the server believes
        self.true_remaining = base.initial.copy()
        self.participations = np.zeros(len(self.initial), dtype=int)
        self.refusals = np.zeros(len(self.initial), dtype=int)
        # A client whose charge is non-increasing and whose margin is fixed will
        # refuse for ever once it has refused once, so a server that receives a
        # refusal has no reason to invite that client again. Set False to
        # reproduce the earlier behaviour, in which the refusal was discarded and
        # the published sort key re-selected the same clients every round.
        self.server_drops_refusers = True
        self.has_refused = np.zeros(len(self.initial), dtype=bool)
        self.kappa = float(kappa)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------- what the server sees
    def can_afford(self, k: int) -> bool:
        if self.server_drops_refusers and self.has_refused[k]:
            return False
        return self.remaining[k] >= self.cost[k]

    def budget_fraction(self, k: int) -> float:
        return float(self.remaining[k] / self.initial[k]) if self.initial[k] else 0.0

    # ------------------------------------------------- what the client sees
    def true_fraction(self, k: int) -> float:
        return float(self.true_remaining[k] / self.initial[k]) if self.initial[k] else 0.0

    def draw(self, k: int) -> float:
        """The true cost of one participation by client k."""
        j = 1.0 if self.noise <= 0 else float(self.rng.uniform(1 - self.noise, 1 + self.noise))
        return float(self.cost[k] * self.kappa * j)

    # ------------------------------------------------------------- updates
    def charge(self, k: int, true_cost: Optional[float] = None) -> float:
        """Spend one participation: the server books its estimate, reality books the truth."""
        drawn = self.draw(k) if true_cost is None else float(true_cost)
        self.remaining[k] = max(0.0, self.remaining[k] - self.cost[k])
        self.true_remaining[k] -= drawn
        self.participations[k] += 1
        return drawn

    def refuse(self, k: int) -> None:
        """A refusal costs nothing and leaves both charge views unchanged.

        It is not, however, invisible: the client returns its state, and the
        server records that it has refused. Without that record the published
        sort key, ``(participations, -budget_fraction, k)``, returns the refusing
        client to the head of the queue, and a round in which every selected
        client refuses is a fixed point that ends training for good.
        """
        self.refusals[k] += 1
        self.has_refused[k] = True

    # ------------------------------------------------------------- summary
    def depleted(self, floor: float = 0.0) -> int:
        """Clients whose true remaining energy has fallen to or below ``floor``."""
        return int((self.true_remaining <= floor).sum())

    def drained(self, frac: float = 0.10) -> int:
        """Clients left below ``frac`` of their initial stock, in truth."""
        return int((self.true_remaining < frac * self.initial).sum())


def run_round(policy, t, T, stock: DualStock, delta: float, client_check: bool,
              oracle_refusal: bool = False):
    """One round of two-level control. Returns the clients that actually trained.

    The server selects from what it believes. Each selected client then compares
    its **measured** remaining energy against the cost it *expects* to pay, with
    a margin ``delta``, and refuses if the margin is not covered. A refusal
    spends nothing. This is Definition 2 of the article, and it is what a device
    can actually evaluate: it knows its own charge, and it knows the estimate the
    server sent, but it does not know what this round will really cost until it
    has paid.

    A client whose true charge has already reached zero never trains, in either
    arm: that is physics, not policy.

    ``oracle_refusal`` adds a second test against the *true* draw, so the client
    never overdraws at all. That is a strictly stronger guarantee than
    Definition 2 and it needs information a deployment does not have, so it is
    off by default and reported separately where it is used. An earlier version
    of this module had it on with no flag, which made the measured minimum charge
    non-negative by construction rather than by the theorem.
    """
    selected = policy.select(t, T, stock, None)
    trained = []
    for k in selected:
        if stock.true_remaining[k] <= 0.0:
            # A flat device cannot train, whatever the server believes. Without
            # this, an arm with no client-side check keeps drawing energy from
            # batteries already below zero and keeps contributing updates.
            stock.refuse(k)
            continue
        if client_check:
            expected = stock.cost[k] * (1.0 + delta)
            if stock.true_remaining[k] < expected:
                stock.refuse(k)
                continue
        true_cost = stock.draw(k)
        if client_check and oracle_refusal and stock.true_remaining[k] < true_cost:
            stock.refuse(k)
            continue
        stock.charge(k, true_cost)
        trained.append(k)
    return trained
