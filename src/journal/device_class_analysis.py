#!/usr/bin/env python3
"""Requirement 5, full corpus: attack<->device concentration in CIC IoT-DIAD 2024.

Consumes device_class_counts.tsv (single-pass awk extraction over all 28
packet-based files) and reports, per attack class, how concentrated its traffic
is across devices — the empirical basis for H3.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TSV = ROOT / "results" / "journal" / "device_class_counts.tsv"
OUT = ROOT / "results" / "journal" / "device_class_concentration.json"

# CIC IoT-DIAD taxonomy: 7 attack categories + Benign (CICIoT2023 convention,
# under which Backdoor_Malware is grouped with the Web-based attacks).
FILE_TO_CLASS = {
    "BenignTraffic1": "Benign",
    "DDoS-ACK_Fragmentation12": "DDoS", "DDoS-HTTP_Flood-": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS", "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    "Mirai-greip_flood1": "Mirai",
    "DictionaryBruteForce": "BruteForce",
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    "BrowserHijacking": "Web", "CommandInjection": "Web",
    "SqlInjection": "Web", "XSS": "Web", "Uploading_Attack": "Web",
    "Backdoor_Malware": "Web",
}


def load():
    counts = defaultdict(lambda: defaultdict(int))   # class -> device -> n
    unmapped = set()
    for line in TSV.read_text().splitlines():
        f, dev, n = line.split("\t")
        cls = FILE_TO_CLASS.get(f)
        if cls is None:
            unmapped.add(f); continue
        counts[cls][dev.strip()] += int(n)
    if unmapped:
        print("!! unmapped files:", unmapped)
    return counts


def main():
    counts = load()
    classes = sorted(counts)
    devices = sorted({d for c in counts.values() for d in c})
    M = np.zeros((len(devices), len(classes)))
    for j, c in enumerate(classes):
        for i, d in enumerate(devices):
            M[i, j] = counts[c].get(d, 0)

    tot = M.sum(axis=0)
    print(f"\n{'='*88}")
    print(f"CIC IoT-DIAD 2024 — FULL CORPUS: {len(devices)} devices x {len(classes)} classes")
    print(f"total records: {int(M.sum()):,}")
    print('='*88)
    print(f"{'class':<12} {'records':>13} {'devs':>6} {'top1':>7} {'d@90%':>7} {'d@99%':>7} {'entropy':>8}")
    print('-'*88)

    summary = []
    for j, c in enumerate(classes):
        col = M[:, j]; s = col / max(tot[j], 1)
        srt = np.sort(s)[::-1]; cs = np.cumsum(srt)
        d90 = int(np.searchsorted(cs, 0.90) + 1)
        d99 = int(np.searchsorted(cs, 0.99) + 1)
        nz = s[s > 0]
        H = float(-(nz*np.log(nz)).sum()/np.log(len(devices))) if len(devices) > 1 else 0.0
        print(f"{c:<12} {int(tot[j]):>13,} {int((col>0).sum()):>6} {srt[0]:>7.3f} "
              f"{d90:>7} {d99:>7} {H:>8.3f}")
        summary.append({"class": c, "records": int(tot[j]),
                        "devices_present": int((col > 0).sum()),
                        "top1_share": float(srt[0]), "devices_90pct": d90,
                        "devices_99pct": d99, "norm_entropy": H})

    # Exclusion counterfactual: drop the devices that carry the most of the
    # MOST-concentrated attack class, and measure collateral loss per class.
    atk = [c for c in classes if c != "Benign"]
    target = min(atk, key=lambda c: summary[classes.index(c)]["devices_90pct"])
    jt = classes.index(target)
    share = M / np.maximum(tot, 1)
    order = np.argsort(-share[:, jt])

    print(f"\nMost concentrated attack class: {target} "
          f"({summary[jt]['devices_90pct']} devices cover 90%)")
    print(f"\nExclusion counterfactual — drop top-N {target}-carrying devices:")
    hdr = f"  {'N':>3} | " + " | ".join(f"{c[:9]:>9}" for c in classes)
    print(hdr); print("  " + "-"*(len(hdr)-2))
    cf = []
    for N in (1, 3, 5, 7, 10, 15):
        ex = order[:N]
        row = [float(share[ex, k].sum()) for k in range(len(classes))]
        print(f"  {N:>3} | " + " | ".join(f"{v:>9.3f}" for v in row))
        cf.append({"n_excluded": N, "loss_by_class": dict(zip(classes, row))})

    OUT.write_text(json.dumps({
        "classes": classes, "devices": devices, "matrix": M.tolist(),
        "summary": summary, "target_class": target, "counterfactual": cf,
        "total_records": int(M.sum()),
    }, indent=1))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
