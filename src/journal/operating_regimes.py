#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operating regimes for the finite-stock study.

Modelling correction
--------------------
``B_k`` is the **energy budget a device allocates to federated learning over the
campaign**, not its total battery. A field device does not spend its whole cell
on model training: it runs a duty cycle in which ML is one of several tasks.
Writing the stock as

    B_k = capacity_k * allocation

is both more realistic and what makes the regime axis meaningful, since the
allocation fraction is a deployment policy rather than a property of the cell.

A small multilayer perceptron trained on a full lithium cell does not exhaust
its budget within any plausible horizon. The regimes below therefore span from
that non-binding case, retained as a control, to a severely constrained
harvested-power setting in which the budget is the limiting factor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    name: str
    backbone: str
    allocation: float      # fraction of device capacity budgeted to FL
    local_epochs: int
    note: str


# Fast-binding regime for experiments whose claim is about the SCHEDULE rather
# than the model. Propositions 1-4 depend on the ratio rho = e_k/B_k only, never
# on how that ratio is achieved, so a cheap backbone with a small allocation
# reproduces R2's cost-ratio profile at a fraction of the compute.
REGIMES = {
    # Two regimes are required because the question at issue is what happens when
    # the energy budget is the binding constraint. The slack setting is retained
    # as an explicit control rather than discarded, since a policy that helps
    # under scarcity may be neutral or harmful without it.
    "R2f-bind":  Regime("R2f-bind",  "mlp", 4.0e-5, 1,
                        "budget BINDS: aggregate ~60% of demand m*T"),
    "R2f-slack": Regime("R2f-slack", "mlp", 8.6e-5, 1,
                        "budget does NOT bind (~139% of demand) -- control condition"),
    "R2f": Regime("R2f", "mlp", 4.0e-5, 1, "alias of R2f-bind"),
    # Control: reproduces the published setting's model scale. The point of
    # including it is to show the constraint does NOT bind there.
    "R0": Regime("R0", "mlp", 0.01, 1,
                 "published analogue: 54K-class MLP, 1% budget -- non-binding control"),
    # Large model, generous budget: constraint present but slack.
    "R1": Regime("R1", "ft", 0.10, 1,
                 "FT-Transformer, 10% budget -- weakly binding"),
    # Primary regime: transformer-scale model on a 1% duty-cycle budget.
    "R2": Regime("R2", "ft", 0.01, 1,
                 "FT-Transformer, 1% budget -- binding"),
    # Severe: larger model, multiple local epochs, same small budget.
    "R3": Regime("R3", "ft_large", 0.01, 5,
                 "FT-Transformer-large x5 epochs, 1% budget -- severely binding"),
}
