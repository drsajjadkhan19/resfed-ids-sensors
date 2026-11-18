#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requirement 5 -- attack<->device concentration in CIC IoT-DIAD 2024.

Blocking check for hypothesis H3: if attack classes are not concentrated on
particular devices, energy-based exclusion cannot cause class-specific harm.

The analysis covers every packet-based capture in the corpus. It is written as
a driver over the two-stage pipeline below rather than as a self-contained
scanner, so that a partial data directory cannot silently yield a subset result
that would appear valid:

    1. build_device_class_matrix.sh  -- single awk pass over every *.csv,
                                        extracting (file, device_mac, count)
    2. analyze_device_class.py       -- maps files to the 8-class CIC taxonomy
                                        and computes concentration + the
                                        exclusion counterfactual
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TSV = ROOT / "results" / "journal" / "device_class_counts.tsv"


def main() -> int:
    rebuild = "--rebuild" in sys.argv or not TSV.exists()
    if rebuild:
        print("Extracting (device_mac, class) counts over the full corpus ...")
        r = subprocess.run(["bash", str(HERE / "build_device_class_matrix.sh")])
        if r.returncode != 0:
            print("extraction failed", file=sys.stderr)
            return r.returncode
    else:
        print(f"Using existing counts: {TSV}   (pass --rebuild to regenerate)")

    from src.journal.device_class_analysis import main as analyze
    analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
