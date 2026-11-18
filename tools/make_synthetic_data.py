#!/usr/bin/env python3
"""Write small synthetic inputs in the shape the pipeline expects.

This exists only to check that the code runs end to end (see tools/smoke_test.sh).
The arrays are random and carry no information about intrusion detection;
nothing produced from them may be reported.

    python tools/make_synthetic_data.py <repo_root>
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rng = np.random.default_rng(0)

# device identities chosen so that every hardware tier is present
names = ([f"raspberry pi {i}" for i in range(3)] + [f"smartthings hub {i}" for i in range(6)]
         + [f"echo dot {i}" for i in range(10)] + [f"wyze camera {i}" for i in range(20)]
         + [f"roomba {i}" for i in range(6)] + [f"teckin plug {i}" for i in range(15)])
classes = ["Benign", "DDoS-TCP_Flood", "DoS-UDP_Flood", "Mirai-greip_flood",
           "Recon-PortScan", "SqlInjection", "MITM-ArpSpoofing", "DNS_Spoofing"]
F, D, C = 40, len(names), len(classes)

rows_X, rows_y, rows_d = [], [], []
centres = rng.normal(0, 2, size=(C, F))
for d in range(D):
    n = int(rng.integers(300, 1500))
    p = rng.dirichlet(np.full(C, 0.4))
    y = rng.choice(C, size=n, p=p)
    X = centres[y] + rng.normal(0, 1.5, size=(n, F)) + rng.normal(0, 0.5, size=F)
    X[rng.random(X.shape) < 0.01] = np.nan
    rows_X.append(X); rows_y.append(y); rows_d.append(np.full(n, d))
X = np.vstack(rows_X).astype(np.float32)
y = np.concatenate(rows_y).astype(np.int16)
dev = np.concatenate(rows_d).astype(np.int16)

out = root / "data" / "diad_federated"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "X.npy", X); np.save(out / "y.npy", y); np.save(out / "device.npy", dev)
prof = np.zeros((D, C), dtype=np.int64)
np.add.at(prof, (dev, y), 1)
(out / "metadata.json").write_text(json.dumps(dict(
    n_rows=int(len(X)), n_features=F, feature_names=[f"f{i}" for i in range(F)],
    classes=classes, devices=names, n_devices=D, synthetic=True,
    class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)},
    device_class_profile=prof.tolist()), indent=1))

# CICIoT2023-shaped clean split for the transfer study
out2 = root / "data" / "clean_ciciot2023"
out2.mkdir(parents=True, exist_ok=True)
n = 20000
yt = rng.integers(0, C, size=n)
Xt = (centres[yt] + rng.normal(0, 1.5, size=(n, F))).astype(np.float32)
cut = int(0.8 * n)
np.save(out2 / "X_train.npy", Xt[:cut]); np.save(out2 / "X_test.npy", Xt[cut:])
np.save(out2 / "y_train.npy", yt[:cut]); np.save(out2 / "y_test.npy", yt[cut:])
(out2 / "metadata.json").write_text(json.dumps(dict(
    dataset="synthetic", n_train=cut, n_test=n - cut, n_features=F, n_classes=C,
    class_names=classes, feature_names=[f"f{i}" for i in range(F)], synthetic=True), indent=2))
print(f"synthetic data written under {root/'data'} ({len(X)} rows, {D} devices)")
