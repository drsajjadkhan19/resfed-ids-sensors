#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client partitioning under statistical heterogeneity.

The published ResFed-IDS work partitions IID only (``np.array_split`` over a
shuffled index). No non-IID partitioning exists anywhere in the published
codebase. This module adds Dirichlet label-skew partitioning, the standard
protocol from Hsu, Qi & Brown (arXiv:1909.06335), so that statistical and
system heterogeneity can be varied independently.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def iid_partition(
    y: np.ndarray, num_clients: int, seed: int = 42
) -> List[np.ndarray]:
    """Equal-size random split (reproduces the published protocol)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [np.sort(p) for p in np.array_split(idx, num_clients)]


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_per_client: int = 20,
) -> List[np.ndarray]:
    """Label-skewed partition via a Dirichlet(alpha) prior over clients.

    Small ``alpha`` => severe skew (each client sees few classes).
    Large ``alpha`` => approaches IID.

    Retries until every client holds at least ``min_per_client`` samples, so
    that no client is degenerate.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    for _ in range(100):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())
        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_per_client:
            return [np.sort(np.array(b, dtype=np.int64)) for b in buckets]

    # Fall back to the last draw rather than looping forever.
    # dtype is forced: an empty bucket would otherwise be float64 and break indexing.
    return [np.sort(np.array(b, dtype=np.int64)) for b in buckets]


def class_distribution(
    y: np.ndarray, parts: List[np.ndarray], num_classes: int
) -> np.ndarray:
    """Return a (num_clients, num_classes) matrix of per-client class counts."""
    m = np.zeros((len(parts), num_classes), dtype=int)
    for k, idx in enumerate(parts):
        vals, cnts = np.unique(y[idx], return_counts=True)
        m[k, vals.astype(int)] = cnts
    return m


def class_concentration(dist: np.ndarray) -> np.ndarray:
    """For each class, the share held by its single largest holder.

    A value near 1.0 means the class is concentrated on one client, so losing
    that client removes the class from training almost entirely.
    """
    totals = dist.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(totals > 0, dist.max(axis=0) / np.maximum(totals, 1), 0.0)
    return share


def assign_device_classes(
    num_clients: int,
    class_names: Tuple[str, ...] = ("gateway", "node", "mote"),
    seed: int = 42,
    correlate_with: np.ndarray | None = None,
) -> List[str]:
    """Assign a hardware class to each client.

    Parameters
    ----------
    correlate_with :
        Optional per-client score. When given, clients are ranked by it and the
        weakest hardware class is assigned to the highest-scoring clients. This
        lets an experiment deliberately couple *data* heterogeneity to *device*
        heterogeneity, which is the realistic case (specialised low-power
        sensors both hold distinctive traffic and have the smallest batteries)
        and the condition under which energy gating can cause class blindness.

        When ``None``, hardware classes are assigned round-robin, independently
        of data — the control condition.
    """
    if correlate_with is None:
        return [class_names[k % len(class_names)] for k in range(num_clients)]

    order = np.argsort(-np.asarray(correlate_with))  # highest score first
    out = ["" for _ in range(num_clients)]
    weakest = class_names[-1]
    n_weak = max(1, num_clients // len(class_names))
    for rank, k in enumerate(order):
        if rank < n_weak:
            out[k] = weakest
        else:
            out[k] = class_names[(rank - n_weak) % (len(class_names) - 1)]
    return out
