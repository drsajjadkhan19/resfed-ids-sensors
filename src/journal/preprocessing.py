#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-free preprocessing for the journal extension.

The published pipeline (``src/utils/preprocess.py``) fits ``StandardScaler`` on
the full corpus *before* ``train_test_split``, so test-set statistics leak into
the scaler. Verified at ``preprocess.py:514 -> 517`` and in the CICIIoT2025 and
CIC-ToN-IoT variants.

This module reuses the published loading / label-mapping / filtering / balancing
steps unchanged, then applies the correct order:

    split  ->  fit scaler on TRAIN ONLY  ->  transform train and test

Outputs go to a separate directory so the published artifacts are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CODE_ROOT = Path(__file__).resolve().parents[2]   # .../seidy-thesis/code
sys.path.insert(0, str(CODE_ROOT))

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _fit_transform_train_only(X_train, X_test):
    """Fit the scaler on training data only, then transform both splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def build_ciciot2023(out_dir: Path, max_samples: int = 10_000) -> Dict[str, Any]:
    """Rebuild CICIoT2023 without scale-before-split leakage."""
    from src.utils.preprocess import (
        DATA_PATH, load_data, map_labels, filter_classes,
        balance_classes, encode_labels,
    )

    df = load_data(DATA_PATH)
    df = map_labels(df)
    df = filter_classes(df)
    df = balance_classes(df, max_samples=max_samples)
    y, label_encoder = encode_labels(df)

    feature_cols = [c for c in df.columns if c not in ("label", "category")]
    X = df[feature_cols].values.astype(np.float64)

    # ---- correct order: split BEFORE fitting the scaler ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, scaler = _fit_transform_train_only(X_train, X_test)

    return _save(out_dir, X_train, X_test, y_train, y_test, scaler,
                 feature_cols, list(label_encoder.classes_), "CICIoT2023")


def build_ciciiot2025(out_dir: Path, max_samples: int = 10_000) -> Dict[str, Any]:
    """Rebuild CICIIoT2025 without scale-before-split leakage."""
    import src.utils.preprocess_ciciiot2025 as p25

    df = p25.load_data(p25.DATA_DIR) if hasattr(p25, "DATA_DIR") else p25.load_data()
    for step in ("map_labels", "filter_classes"):
        if hasattr(p25, step):
            df = getattr(p25, step)(df)
    if hasattr(p25, "balance_classes"):
        df = p25.balance_classes(df, max_samples=max_samples)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    feature_cols = [c for c in df.columns
                    if c not in ("label", "category", "Label", "Attack")]
    X = df[feature_cols].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, scaler = _fit_transform_train_only(X_train, X_test)

    return _save(out_dir, X_train, X_test, y_train, y_test, scaler,
                 feature_cols, list(le.classes_), "CICIIoT2025")


def _save(out_dir, X_train, X_test, y_train, y_test, scaler,
          feature_names, class_names, dataset_name) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(out_dir / "X_test.npy", X_test.astype(np.float32))
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)

    meta = {
        "dataset": dataset_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(class_names)),
        "class_names": [str(c) for c in class_names],
        "feature_names": [str(f) for f in feature_names],
        "leakage_free": True,
        "scaler_fitted_on": "train_only",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[OK] {dataset_name}: train={len(X_train):,} test={len(X_test):,} "
          f"features={X_train.shape[1]} classes={len(class_names)}")
    print(f"     scaler fitted on TRAIN ONLY -> {out_dir}")
    return meta


def load_clean(processed_dir: Path) -> Dict[str, Any]:
    """Load a leakage-free processed dataset."""
    d = Path(processed_dir)
    meta = json.loads((d / "metadata.json").read_text())
    return {
        "X_train": np.load(d / "X_train.npy"),
        "X_test": np.load(d / "X_test.npy"),
        "y_train": np.load(d / "y_train.npy"),
        "y_test": np.load(d / "y_test.npy"),
        **meta,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ciciot2023", "ciciiot2025"],
                    default="ciciot2023")
    args = ap.parse_args()

    base = CODE_ROOT / "data"
    if args.dataset == "ciciot2023":
        build_ciciot2023(base / "clean_ciciot2023")
    else:
        build_ciciiot2025(base / "clean_ciciiot2025")
