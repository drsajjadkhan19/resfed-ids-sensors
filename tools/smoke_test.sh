#!/usr/bin/env bash
# Runs every study on synthetic data with 2 seeds and 4 rounds, in a throw-away
# copy of the repository, to check that the code runs end to end. Nothing it
# produces is a result. Usage: bash tools/smoke_test.sh [study names...]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)/repo"
mkdir -p "$TMP"
cp -r "$SRC/src" "$SRC/reproduce.py" "$SRC/tools" "$TMP/"
mkdir -p "$TMP/results"
cd "$TMP"
grep -rl "range(1, 11)" src | xargs -r sed -i 's/range(1, 11)/range(1, 3)/g'
grep -rl "rounds=40" src | xargs -r sed -i 's/rounds=40/rounds=4/g'
python tools/make_synthetic_data.py "$TMP"
RESFED_SMOKE=1 python reproduce.py ${@:+--only "$@"}
echo "smoke test passed in $TMP"
