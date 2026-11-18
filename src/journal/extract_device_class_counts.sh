#!/bin/bash
# Single-pass extraction of (device_mac, class) counts over the full corpus.
# device_mac is column 32 in every packet-based file (verified).
D="$(cd "$(dirname "$0")/../../data/cic_iot_diad_2024" && pwd)"
OUT="$(cd "$(dirname "$0")/../../results" && pwd)/journal/device_class_counts.tsv"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
for f in "$D"/*.csv; do
  b=$(basename "$f" .csv)
  awk -F',' -v cls="$b" 'NR>1 && NF>=32 {c[$32]++} END {for (d in c) printf "%s\t%s\t%d\n", cls, d, c[d]}' "$f" >> "$OUT"
  echo "  done: $b" >&2
done
echo "wrote $OUT" >&2
