#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Participation policies under a finite energy stock.

Policies share a common interface and draw on the same energy ledger, so any two
are compared under identical accounting.

* ``ThresholdGating`` -- eligibility gated on a fixed state-of-charge threshold,
  the design used by resource-aware federated intrusion detection systems.
* ``FedAvgFull``, ``RandomM``, ``UtilityGreedy`` -- reference policies.
* ``TokenBucket``, ``DeficitRoundRobin`` -- standard rate-control and
  fair-queueing primitives, included because any budget-pacing rule must be
  measured against the established mechanisms it resembles.
* ``LACS`` -- budget pacing with an optional coverage term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
#  Coverage utility  f(S) = sum_c w_c * min(1, (h_c + sum_{k in S} n_kc)/theta_c)
# --------------------------------------------------------------------------- #
class CoverageUtility:
    """Monotone submodular per-round coverage (Lemma 1).

    ``presence`` is a (K, C) binary matrix indicating which classes each client
    holds. Exact counts are deliberately not used: the saturating form makes them
    unnecessary, and disclosing presence bits rather than counts limits what each
    client reveals about its local distribution.
    """

    def __init__(self, presence: np.ndarray, weights: Optional[np.ndarray] = None,
                 theta: float = 1.0, decay: float = 0.0):
        self.P = presence.astype(np.float64)
        self.K, self.C = self.P.shape
        self.w = np.ones(self.C) if weights is None else np.asarray(weights, float)
        self.theta = float(theta)
        self.decay = float(decay)          # 0 => history ignored (pure per-round)
        self.h = np.zeros(self.C)          # cumulative exposure

    def value(self, sel: Sequence[int]) -> float:
        a = self.h * self.decay + (self.P[list(sel)].sum(axis=0) if len(sel) else 0.0)
        return float((self.w * np.minimum(1.0, a / self.theta)).sum())

    def marginal(self, sel: Sequence[int], k: int) -> float:
        return self.value(list(sel) + [k]) - self.value(sel)

    def commit(self, sel: Sequence[int]) -> None:
        self.h = self.h * self.decay + (self.P[list(sel)].sum(axis=0) if len(sel) else 0.0)


# --------------------------------------------------------------------------- #
@dataclass
class Policy:
    name: str
    m: int                                  # per-round server capacity
    history: List[List[int]] = field(default_factory=list)

    def select(self, t: int, T: int, stock, util=None) -> List[int]:
        raise NotImplementedError

    def _record(self, sel):
        self.history.append(list(sel))
        return list(sel)


class FedAvgFull(Policy):
    """Full participation subject to server capacity, with no resource policy.

    Server capacity is respected so that this policy stays comparable with the
    others under a common per-round limit. ``uncapped=True`` lifts the limit and
    is used only for an unconstrained reference arm.
    """

    def __init__(self, name, m, uncapped: bool = False):
        super().__init__(name, m)
        self.uncapped = uncapped

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining)) if stock.can_afford(k)]
        if self.uncapped:
            return self._record(elig)
        # deterministic, capacity-respecting: least-served first
        elig.sort(key=lambda k: (stock.participations[k], -stock.budget_fraction(k), k))
        return self._record(elig[: self.m])


class RandomM(Policy):
    """Random m affordable clients — controls for 'fewer clients per round'."""

    def __init__(self, name, m, seed=0):
        super().__init__(name, m)
        self.rng = np.random.default_rng(seed)

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining)) if stock.can_afford(k)]
        if len(elig) > self.m:
            elig = list(self.rng.choice(elig, self.m, replace=False))
        return self._record(elig)


class ThresholdGating(Policy):
    """Eligible iff state of charge >= tau.

    The published gating policy. Its participation count, stranded energy and
    stopping time all follow in closed form from tau and the per-round cost,
    which is what makes it a useful reference: any departure from those
    predictions is a property of the simulation, not of the policy.
    """

    def __init__(self, name, m, tau: float):
        super().__init__(name, m)
        self.tau = float(tau)

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining))
                if stock.budget_fraction(k) >= self.tau and stock.can_afford(k)]
        # published tie-break: least-trained first, then higher charge
        elig.sort(key=lambda k: (stock.participations[k], -stock.budget_fraction(k), k))
        return self._record(elig[: self.m])


class UtilityGreedy(Policy):
    """Oort-style: rank by a supplied per-client utility, ignore lifetime."""

    def __init__(self, name, m, utility: np.ndarray):
        super().__init__(name, m)
        self.u = np.asarray(utility, float)

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining)) if stock.can_afford(k)]
        elig.sort(key=lambda k: -self.u[k])
        return self._record(elig[: self.m])


class LACS(Policy):
    """Budget pacing, optionally combined with coverage-greedy selection.

    Pacing admits a client only while its cumulative expenditure stays within a
    linear entitlement, spent_k + e_k <= B_k (t + Delta) / T, spreading a finite
    stock across the horizon instead of gating on remaining charge. The optional
    coverage term selects, among paced-eligible clients, those offering the
    greatest marginal class coverage per joule.
    """

    def __init__(self, name, m, delta: float = 0.0, use_coverage: bool = True,
                 use_pacing: bool = True):
        super().__init__(name, m)
        self.delta = float(delta)
        self.use_coverage = use_coverage
        self.use_pacing = use_pacing

    def _paced(self, t, T, stock) -> List[int]:
        out = []
        for k in range(len(stock.remaining)):
            if not stock.can_afford(k):
                continue
            if not self.use_pacing:
                out.append(k); continue
            spent = stock.initial[k] - stock.remaining[k]
            # Cold-start floor. Without it, a client that can afford fewer than
            # T rounds is inadmissible until t >= T * rho_k, which stalls the
            # federation in its opening rounds. The floor preserves the budget
            # bound: total expenditure stays at most max(e_k, B_k) = B_k whenever
            # e_k <= B_k, which the affordability check enforces.
            allowance = max(stock.cost[k],
                            stock.initial[k] * min(1.0, (t + self.delta) / T))
            if spent + stock.cost[k] <= allowance + 1e-9:
                out.append(k)
        return out

    def select(self, t, T, stock, util: CoverageUtility = None):
        cand = self._paced(t, T, stock)
        if not cand:
            return self._record([])
        if not self.use_coverage or util is None:
            cand.sort(key=lambda k: (stock.participations[k], -stock.budget_fraction(k), k))
            return self._record(cand[: self.m])

        sel: List[int] = []
        pool = set(cand)
        while len(sel) < self.m and pool:
            best, best_key = None, None
            for k in pool:
                gain = util.marginal(sel, k)
                ratio = gain / max(stock.cost[k], 1e-12)
                # Least-participated tie-break, which is what guarantees every
                # affordable client is eventually served.
                key = (ratio, -stock.participations[k], stock.budget_fraction(k))
                if best_key is None or key > best_key:
                    best, best_key = k, key
            if best is None or best_key[0] <= 0:
                # no coverage gain left: fall back to least-participated
                rest = sorted(pool, key=lambda k: (stock.participations[k], k))
                sel.extend(rest[: self.m - len(sel)])
                break
            sel.append(best); pool.discard(best)
        return self._record(sel)


# --------------------------------------------------------------------------- #
#  Established scheduling primitives
#
#  A cumulative per-client service curve with a startup credit is, in networking
#  terms, a token bucket. Evaluating a budget-pacing rule only against threshold
#  gating would leave the more relevant comparison unasked, so the standard
#  primitives are implemented here and run on the same footing.
# --------------------------------------------------------------------------- #
class TokenBucket(Policy):
    """Token bucket with per-round refill B_k / T and a bounded token ceiling.

    Unused allowance accumulates only up to the ceiling, in contrast to a linear
    cumulative entitlement, which admits the entire unspent budget at any time.
    """

    def __init__(self, name, m, burst_rounds: float | None = None):
        """burst_rounds: token ceiling, expressed in rounds' worth of rate.

        The ceiling must let a client accumulate at least one round's cost: a
        client that can afford N rounds over a horizon T needs T/N rounds of
        accumulation, so any lower ceiling starves it permanently. Passing None
        sets the ceiling per client to exactly one round's cost, the least
        permissive bucket that can still fire.
        """
        super().__init__(name, m)
        self.burst_rounds = burst_rounds
        self.tokens = None
        self.cap = None

    def select(self, t, T, stock, util=None):
        K = len(stock.remaining)
        rate = stock.initial / T
        if self.cap is None:
            self.cap = (rate * self.burst_rounds if self.burst_rounds
                        else stock.cost.astype(float).copy())
            self.cap = np.maximum(self.cap, stock.cost)   # must be able to fire
            self.tokens = np.zeros(K)
        self.tokens = np.minimum(self.tokens + rate, self.cap)
        cand = [k for k in range(K)
                if stock.can_afford(k) and self.tokens[k] >= stock.cost[k]]
        cand.sort(key=lambda k: (stock.participations[k], -stock.budget_fraction(k), k))
        sel = cand[: self.m]
        for k in sel:
            self.tokens[k] -= stock.cost[k]
        return self._record(sel)


class DeficitRoundRobin(Policy):
    """Deficit round robin with per-client quantum B_k / T.

    A standard fair-queueing primitive, included so that any pacing rule is
    measured against the established mechanism it most closely resembles.

    ``scale`` multiplies the per-round quantum. It is the policy's own free
    parameter, and exists so that this family can be given a tuning budget equal
    to that of the selectors it is compared against: a comparison in which only
    the competitors' parameters are held fixed measures the protocol, not the
    algorithms.
    """

    def __init__(self, name, m, scale: float = 1.0):
        super().__init__(name, m)
        self.deficit = None
        self.scale = float(scale)

    def select(self, t, T, stock, util=None):
        K = len(stock.remaining)
        quantum = self.scale * stock.initial / T
        if self.deficit is None:
            self.deficit = np.zeros(K)
        self.deficit += quantum
        cand = [k for k in range(K)
                if stock.can_afford(k) and self.deficit[k] >= stock.cost[k]]
        cand.sort(key=lambda k: (-self.deficit[k], stock.participations[k], k))
        sel = cand[: self.m]
        for k in sel:
            self.deficit[k] -= stock.cost[k]
        return self._record(sel)


class OortEnergyAdapted(Policy):
    """Oort's selection rule, transposed from a time budget to an energy stock.

    Follows Lai, Zhu, Madhyastha & Chowdhury, *Oort: Efficient Federated
    Learning via Guided Participant Selection*, OSDI 2021, with one deliberate
    substitution that must be stated rather than glossed. Oort's system utility
    penalises clients whose participation takes too long, because the constraint
    it was designed for is round duration under stragglers. The constraint here
    is a finite energy stock, so the penalty is applied to per-round energy
    instead. This is a transposition, not a reimplementation: the algorithm is
    referred to as *energy-adapted Oort* throughout, and a comparison against it
    is a comparison against that adaptation, not against the published method on
    the problem it was written for.

    Running Oort without any system-utility term would be the worse option -- it
    would remove the resource awareness that is half of the algorithm, in a study
    whose entire subject is a resource constraint -- but it would at least be
    the published objective. The transposition is the more useful comparator and
    the less literal one, and it is labelled accordingly. A client's
    statistical utility is its data volume times the root mean squared training
    loss observed at its last participation,

        U_k = n_k * sqrt( (1/n_k) * sum_i loss_i^2 ),

    A client's statistical utility is its data volume times the root mean squared
    training loss observed at its last participation, incremented by a staleness
    bonus for clients unseen for many rounds. A fraction ``eps`` of each round is
    filled by exploration among clients with no utility estimate yet, without
    which the ranking never leaves the clients that happened to be sampled first.

    Only clients that can afford a round are eligible, so the policy operates
    under the same energy stock as every other arm.
    """

    def __init__(self, name, m, eps: float = 0.2, staleness: float = 0.5,
                 alpha: float = 0.5, seed: int = 0):
        super().__init__(name, m)
        self.eps = float(eps)
        self.staleness = float(staleness)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.utility: Dict[int, float] = {}
        self.last_seen: Dict[int, int] = {}

    def update(self, k: int, loss: float, n: int, t: int) -> None:
        """Record the statistical utility observed at client k in round t."""
        self.utility[k] = float(n) * float(np.sqrt(max(loss, 0.0)))
        self.last_seen[k] = int(t)

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining)) if stock.can_afford(k)]
        if not elig:
            return self._record([])
        known = [k for k in elig if k in self.utility]
        unknown = [k for k in elig if k not in self.utility]

        n_explore = min(len(unknown), int(round(self.eps * self.m)))
        chosen = list(self.rng.choice(unknown, n_explore, replace=False)) if n_explore else []

        # Oort's system utility penalises clients whose participation is
        # expensive relative to a reference. In the original the cost is round
        # duration; under a finite stock the operative cost is energy, so the
        # per-round charge takes its place. Omitting this term would understate
        # the algorithm, since resource awareness is part of what it is.
        cost = np.asarray(stock.cost, dtype=float)
        e_ref = float(np.median(cost[elig])) if len(elig) else 1.0

        def score(k):
            age = t - self.last_seen.get(k, 0)
            stale = 1.0 + self.staleness * np.sqrt(max(age, 0))
            penalty = ((e_ref / cost[k]) ** self.alpha) if cost[k] > e_ref else 1.0
            return self.utility[k] * stale * penalty

        known.sort(key=lambda k: -score(k))
        for k in known:
            if len(chosen) >= self.m:
                break
            chosen.append(k)
        # if utilities are scarce, fall back to the remaining unexplored clients
        for k in unknown:
            if len(chosen) >= self.m:
                break
            if k not in chosen:
                chosen.append(k)
        return self._record([int(k) for k in chosen[: self.m]])


class PowerOfChoice(Policy):
    """Biased selection from a size-weighted candidate pool, with probe cost.

    Follows Cho, Wang & Joshi, *Towards Understanding Biased Client Selection in
    Federated Learning*, AISTATS 2022. A candidate set of ``d >= m`` clients is
    drawn with probability proportional to local data volume; the ``m``
    candidates with the highest loss **on the current global model** are served.
    The parameter ``d`` interpolates between unbiased sampling (d = m) and greedy
    loss maximisation (d = K).

    The probe is not free and is charged. Every candidate must receive the
    current global model, run a forward pass over its data, and return a scalar,
    so each probed client is billed for the model download, the forward pass, and
    the reply. Leaving this uncharged would let the policy buy information with
    energy that no other policy is billed for, and any comparison drawn from that
    would be about the accounting rather than about the algorithm.

    **The probe cost is itself a model.** The caller supplies it, and the
    conclusion must be shown to survive a range of it rather than resting on one
    accounting choice; the study that uses this policy sweeps a low, central and
    high setting.

    ``probe_fn`` supplies the loss of the current global model on a client's
    data. When it is absent the policy falls back to the loss last observed at
    participation, and ``probe_charged`` records that no probe was billed.
    """

    def __init__(self, name, m, d_mult: float = 3.0, sizes=None, seed: int = 0,
                 probe_fn=None, probe_cost=None):
        super().__init__(name, m)
        self.d = max(int(round(d_mult * m)), m)
        self.rng = np.random.default_rng(seed)
        self.sizes = None if sizes is None else np.asarray(sizes, float)
        self.loss: Dict[int, float] = {}
        self.probe_fn = probe_fn
        self.probe_cost = None if probe_cost is None else np.asarray(probe_cost, float)
        self.probe_charged = 0.0
        self.probes = 0

    def update(self, k: int, loss: float, n: int, t: int) -> None:
        self.loss[k] = float(loss)

    def select(self, t, T, stock, util=None):
        elig = [k for k in range(len(stock.remaining)) if stock.can_afford(k)]
        if not elig:
            return self._record([])
        if len(elig) > self.d:
            if self.sizes is not None:
                p = self.sizes[elig]
                p = p / p.sum() if p.sum() > 0 else None
            else:
                p = None
            cand = [int(k) for k in self.rng.choice(elig, self.d, replace=False, p=p)]
        else:
            cand = list(elig)

        if self.probe_fn is not None:
            for k in cand:
                self.loss[k] = float(self.probe_fn(k))
                self.probes += 1
                if self.probe_cost is not None:
                    draw = min(float(self.probe_cost[k]), float(stock.remaining[k]))
                    stock.remaining[k] -= draw
                    self.probe_charged += draw
            # a probe can exhaust a client, so affordability is re-checked
            cand = [k for k in cand if stock.can_afford(k)]
            if not cand:
                return self._record([])
        else:
            cand.sort(key=lambda k: -self.loss.get(k, float("inf")))
            return self._record([int(k) for k in cand[: self.m]])

        cand.sort(key=lambda k: -self.loss.get(k, float("inf")))
        return self._record([int(k) for k in cand[: self.m]])


class MinParticipants(Policy):
    """Deficit round robin that declines to run an under-filled round.

    The alternative response to thin rounds: rather than aggregating a
    one-client round, or holding it, simply do not run it. No client trains and
    no energy is spent, so the saved budget is available to later rounds. This
    is a genuinely different design from a mass floor, not a variant of it, and
    it is the natural baseline for one.
    """

    def __init__(self, name, m, q_min: int, inner=None):
        super().__init__(name, m)
        self.q_min = int(q_min)
        self.inner = inner or DeficitRoundRobin(name + "-drr", m)

    def select(self, t, T, stock, util=None):
        affordable = sum(1 for k in range(len(stock.remaining)) if stock.can_afford(k))
        if affordable < self.q_min:
            return self._record([])
        return self._record(self.inner.select(t, T, stock, util))


class GatePriority(Policy):
    """The published eligibility test, with only the priority among eligible changed.

    ``GatedDeficit`` also carries deficit round robin's *admission* rule, which
    withholds an eligible client until it has accrued a participation's worth of
    credit. That makes it a different policy rather than a reordering of the
    published one.

    This class changes nothing but the sort key. Every client the published
    policy would serve in a round is still available in that round, so the two
    arms make the same participations in the same quantity and differ only in
    sequence. Consumed energy, update counts and stranded stock are therefore
    equal to machine precision, which is what makes the comparison a contrast
    about order and nothing else.
    """

    def __init__(self, name, m, tau: float, mode: str = "deficit", seed: int = 0):
        super().__init__(name, m)
        self.tau = float(tau)
        self.mode = mode
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
            elig.sort(key=lambda k: (-self.deficit[k], k))
        elif self.mode == "percost":
            elig.sort(key=lambda k: (-self.deficit[k] / max(stock.cost[k], 1e-12), k))
        elif self.mode == "random":
            self.rng.shuffle(elig)
        else:
            raise ValueError(self.mode)
        sel = elig[: self.m]
        for k in sel:
            self.deficit[k] -= stock.cost[k]
        return self._record(sel)


class GatedDeficit(Policy):
    """The published charge gate, with deficit ordering inside it.

    ResFed-IDS decides two things at once: *who is eligible*, by state of charge
    against a threshold, and *who is served first* among the eligible, by least
    trained and then by higher charge. The first decision carries the policy's
    guarantee: with the threshold above the per-round cost fraction no client is
    ever depleted, which Proposition 1 establishes without reference to any run.
    The second carries no guarantee at all.

    This policy keeps the gate exactly as published and replaces only the order,
    serving the eligible set by accumulated deficit as in fair queueing. The
    depletion guarantee is untouched, because eligibility is unchanged; what
    changes is which eligible client trains first when not all of them can.
    """

    def __init__(self, name, m, tau: float, scale: float = 1.0):
        super().__init__(name, m)
        self.tau = float(tau)
        self.scale = float(scale)
        self.deficit = None

    def select(self, t, T, stock, util=None):
        K = len(stock.remaining)
        if self.deficit is None:
            self.deficit = np.zeros(K)
        self.deficit += self.scale * stock.initial / T
        elig = [k for k in range(K)
                if stock.budget_fraction(k) >= self.tau and stock.can_afford(k)
                and self.deficit[k] >= stock.cost[k]]
        elig.sort(key=lambda k: (-self.deficit[k], stock.participations[k], k))
        sel = elig[: self.m]
        for k in sel:
            self.deficit[k] -= stock.cost[k]
        return self._record(sel)


class MinSelected(Policy):
    """Decline a round whose *selected* set is too small, whatever chose it.

    ``MinParticipants`` refuses a round when too few clients can afford one,
    which is the right guard for a pacing rule: there, affordability is what
    thins the round. It is the wrong guard for a gating rule, which can select
    one or two clients while many others remain affordable but ineligible. This
    variant reads the set the inner policy actually returned, so it catches a
    thin round however the thinness arose. It is the behaviour a framework
    default such as ``min_fit_clients`` provides, and which the published
    implementation set to one.
    """

    def __init__(self, name, m, q_min: int, inner):
        super().__init__(name, m)
        self.q_min = int(q_min)
        self.inner = inner

    def select(self, t, T, stock, util=None):
        sel = self.inner.select(t, T, stock, util)
        return self._record([] if len(sel) < self.q_min else sel)


# The published name is kept as an alias so that existing call sites resolve,
# but the adapted name is what appears in results and in the paper.
Oort = OortEnergyAdapted
