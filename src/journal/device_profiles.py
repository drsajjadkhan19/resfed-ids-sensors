#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware profiles for CIC IoT-DIAD 2024 devices (arm A-phys).

Methodological constraint
-------------------------
Profiles are derived from device identity alone -- vendor OUI prefix and
resolved product name -- and are fixed before any federated run. They are never
chosen with reference to which traffic classes a device carries. Whether the
resulting high-cost-ratio devices coincide with the carriers of any particular
class is an empirical question, and this module deliberately leaves it open.

Deployment scenario
-------------------
The source testbed consists of mains-powered smart-home equipment, whereas the
finite-stock premise applies to battery-powered nodes. The experiments therefore
model a counterfactual battery deployment: each device is assumed to run on a
cell sized to its hardware class, as in a remote, industrial or off-grid
installation. This is a modelling assumption and is reported as such.

Energy parameters per class
---------------------------
``capacity_j``   usable stock (J) for a battery sized to that hardware class
``j_per_flop``   energy per training FLOP; more capable silicon is *more*
                 efficient per operation, so SBCs have the lowest value and
                 microcontroller-class nodes the highest
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HwClass:
    name: str
    capacity_j: float
    j_per_flop: float
    note: str


# 18650-cell / power-bank scale stocks; j_per_flop spans ~2 orders of magnitude
# between an SBC running vectorised code and an MCU emulating float math.
HW_CLASSES: Dict[str, HwClass] = {
    "sbc":     HwClass("sbc",     133_200.0, 1.0e-11, "single-board computer (RPi-class), 10 Ah pack"),
    "gateway": HwClass("gateway",  66_600.0, 2.0e-11, "hub / bridge / base station, 5 Ah pack"),
    "av":      HwClass("av",       66_600.0, 2.5e-11, "speaker / TV / display SoC"),
    "camera":  HwClass("camera",   26_640.0, 6.0e-11, "IP camera ARM SoC, 2 Ah cell"),
    "appliance": HwClass("appliance", 39_960.0, 5.0e-11, "robot / purifier / HVAC controller"),
    "mcu":     HwClass("mcu",       5_328.0, 4.0e-10, "ESP8266/ESP32-class plug, bulb, strip"),
}

# Vendor OUI prefixes with an unambiguous hardware class.
OUI_CLASS = {
    "dc:a6:32": "sbc",      # Raspberry Pi Trading
    "e4:5f:01": "sbc",      # Raspberry Pi Trading
    "b8:27:eb": "sbc",      # Raspberry Pi Foundation
    "00:0c:29": "sbc",      # VMware virtual host
}

# Product-name keywords -> hardware class, longest match wins.
NAME_RULES: List[tuple[str, str]] = [
    ("raspberry", "sbc"),
    ("smart board", "av"),
    ("smart tv", "av"), ("echo show", "av"), ("echo spot", "av"),
    ("echo studio", "av"), ("echo dot", "av"), ("echo", "av"),
    ("sonos", "av"), ("nest mini", "av"), ("harman", "av"), ("speaker", "av"),
    ("radio/lamp", "av"),
    ("hub", "gateway"), ("bridge", "gateway"), ("homebase", "gateway"),
    ("base station", "gateway"), ("smartthings", "gateway"),
    ("camera", "camera"), ("cam ", "camera"), ("cam", "camera"),
    ("simcam", "camera"), ("wyze", "camera"), ("arlo q", "camera"),
    ("plug", "mcu"), ("bulb", "mcu"), ("lamp", "mcu"), ("lightbulb", "mcu"),
    ("light strip", "mcu"), ("power strip", "mcu"), ("rgb", "mcu"),
    ("lifx", "mcu"), ("teckin", "mcu"), ("gosund", "mcu"), ("yutron", "mcu"),
    ("roomba", "appliance"), ("coffee", "appliance"), ("purifier", "appliance"),
    ("hvac", "appliance"), ("humidif", "appliance"), ("weather", "appliance"),
]

DEFAULT_CLASS = "camera"   # commonest tier in this testbed


def is_resolved(identity: str) -> bool:
    """True when an explicit OUI or product-name rule matches the identity.

    Devices that match no rule receive the default class. Reporting how many
    fall into that group, and checking that a conclusion survives assigning them
    to a different class, is the honest way to carry an unresolved identifier.
    """
    s = identity.strip().lower()
    if re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", s):
        return s[:8] in OUI_CLASS
    return any(kw in s for kw, _ in NAME_RULES)


def classify_device(identity: str, default: str = DEFAULT_CLASS) -> str:
    """Map a device_mac value (raw MAC or resolved product name) to a class."""
    s = identity.strip().lower()
    if re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", s):
        return OUI_CLASS.get(s[:8], default)
    for kw, cls in sorted(NAME_RULES, key=lambda t: -len(t[0])):
        if kw in s:
            return cls
    return default


def profile_for(identity: str, default: str = DEFAULT_CLASS) -> HwClass:
    return HW_CLASSES[classify_device(identity, default)]


def assign_profiles(identities: List[str],
                    default: str = DEFAULT_CLASS) -> List[HwClass]:
    return [profile_for(d, default) for d in identities]
