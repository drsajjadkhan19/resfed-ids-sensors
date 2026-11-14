#!/usr/bin/env python3
"""Statistics quoted in the article, computed only from saved run outputs.

    python analysis/stats.py     -> results/statistics.json

Paired comparisons over the ten splits use the paired t-interval of the mean
difference and the exact Wilcoxon signed-rank test; collapse counts (terminal
macro-F1 below 0.10) use Fisher's exact test.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "statistics_journal.json"
subprocess.check_call([sys.executable, str(ROOT / "analysis" / "stats_journal.py"),
                       str(ROOT / "results" / "journal"), str(out)])
stats = {"journal": json.loads(out.read_text())}
rev = ROOT / "results" / "revision"
sys.path.insert(0, str(ROOT))
from analysis.stats_revision import summarise  # noqa: E402
stats["revision"] = summarise(rev)
(ROOT / "results" / "statistics.json").write_text(json.dumps(stats, indent=1))
print("wrote results/statistics.json")
