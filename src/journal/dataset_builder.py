#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the federated CIC IoT-DIAD 2024 dataset.

Design points:

* ``device_mac`` is retained only as the federated partition key and never
  reaches the model as a feature.
* Identifier columns are dropped. Roughly a third of the devices in this corpus
  emit traffic of essentially a single class, so any retained identifier would
  let a classifier memorise the device-to-class mapping instead of learning
  traffic behaviour, and every per-class result would then be an artefact.
* Sampling is uniform within each class. This caps a fourteen-fold class
  imbalance while preserving, in expectation, the distribution of each class
  across devices. That distribution is the object of study and must not be
  altered by the sampling.
"""
from __future__ import annotations

import argparse, json, zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cic_iot_diad_2024"
OUT = ROOT / "data" / "diad_federated"
COUNTS = ROOT / "results" / "journal" / "device_class_counts.tsv"

from src.journal.device_class_analysis import FILE_TO_CLASS  # noqa: E402

# Dropped: direct identifiers, vendor OUIs, and remote-endpoint fingerprints.
DROP = {
    "stream", "src_mac", "dst_mac", "src_ip", "dst_ip", "src_port",
    "device_mac", "eth_src_oui", "eth_dst_oui",
    "http_host", "user_agent", "dns_server", "http_uri", "tls_server",
}
# Low-cardinality protocol-level categoricals that are kept and encoded.
CATEGORICAL = ["handshake_version", "http_request_method",
               "highest_layer", "http_content_type"]

CHUNK = 250_000


def class_totals() -> dict[str, int]:
    tot: dict[str, int] = defaultdict(int)
    for line in COUNTS.read_text().splitlines():
        f, _dev, n = line.split("\t")
        c = FILE_TO_CLASS.get(f)
        if c:
            tot[c] += int(n)
    return dict(tot)


def build(cap: int, seed: int) -> None:
    totals = class_totals()
    rates = {c: min(1.0, cap / n) for c, n in totals.items()}
    print("Per-class sampling (uniform within class -> device shares preserved):")
    for c in sorted(totals):
        print(f"  {c:<12} total={totals[c]:>10,}  rate={rates[c]:6.4f}  "
              f"target~{min(cap, totals[c]):>8,}")

    frames = []
    for f in sorted(DATA.glob("*.csv")):
        cls = FILE_TO_CLASS.get(f.stem)
        if cls is None:
            continue
        rate = rates[cls]
        # Python's built-in hash() is salted per process, so it would not
        # reproduce the corpus from a nominal seed across runs. CRC-32 is stable.
        key = zlib.crc32(f"{f.stem}:{seed}".encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(key)
        kept = 0
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False,
                                 on_bad_lines="skip"):
            if rate < 1.0:
                chunk = chunk[rng.random(len(chunk)) < rate]
            if chunk.empty:
                continue
            dev = chunk["device_mac"].astype(str).str.strip()
            feat = chunk.drop(columns=[c for c in DROP if c in chunk.columns],
                              errors="ignore")
            feat = feat.assign(__device=dev.values, __class=cls)
            frames.append(feat)
            kept += len(feat)
        print(f"  {f.stem:<28} -> {cls:<11} kept={kept:>9,}")

    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"\nCombined: {len(df):,} rows x {df.shape[1]} cols")

    devices = df.pop("__device")
    classes = df.pop("__class")

    # Encode the retained low-cardinality categoricals; coerce the rest numeric.
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str).astype("category").cat.codes
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Column filtering and imputation are fitted transforms and must therefore be
    # estimated on training devices only; fitting them over the full corpus would
    # let held-out device distributions influence the training representation.
    # Only columns that are empty everywhere are dropped here, since such a column
    # carries no information under any split. Variance filtering and imputation
    # are deferred to the Federation, which fits them on client devices alone.
    before = df.shape[1]
    df = df.loc[:, df.notna().any()]
    print(f"Features: {before} -> {df.shape[1]} (dropped all-NaN columns only; "
          f"variance filtering and imputation deferred to the training split)")

    dev_codes, dev_uniques = pd.factorize(devices)
    cls_codes, cls_uniques = pd.factorize(classes, sort=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # NaNs are preserved on purpose: imputation happens after the split.
    np.save(OUT / "X.npy", df.to_numpy(dtype=np.float32))
    np.save(OUT / "y.npy", cls_codes.astype(np.int16))
    np.save(OUT / "device.npy", dev_codes.astype(np.int16))

    # Per-device class profile n_{k,c}, used to construct client partitions.
    prof = np.zeros((len(dev_uniques), len(cls_uniques)), dtype=np.int64)
    for d, c in zip(dev_codes, cls_codes):
        prof[d, c] += 1

    meta = {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1]),
        "feature_names": [str(c) for c in df.columns],
        "classes": [str(c) for c in cls_uniques],
        "devices": [str(d) for d in dev_uniques],
        "n_devices": int(len(dev_uniques)),
        "cap_per_class": cap,
        "seed": seed,
        "dropped_identifier_columns": sorted(DROP),
        "imputation": "deferred_to_training_split",
        "seeding": "zlib.crc32 (process-stable)",
        "class_counts": {str(c): int((cls_codes == i).sum())
                          for i, c in enumerate(cls_uniques)},
        "device_class_profile": prof.tolist(),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))

    print(f"\nDevices: {len(dev_uniques)} | Classes: {list(cls_uniques)}")
    print("Class counts:", meta["class_counts"])
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    build(**vars(ap.parse_args()))
