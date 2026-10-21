"""Cross-context comparison of XAI method suitability (EEG workload vs vision engagement), D-V4.

  .venv/bin/python scripts/compare_contexts.py --eeg-run full --vision-run vision_full

Reads both runs' finished analysis; writes results/cross_context/<eeg>__<vision>/comparison.json.
If either context has no results yet it says what is missing and writes nothing.
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from xaicomp.compare.cross_context import compare_contexts, find_context_dir, save_comparison
from xaicomp.paths import RESULTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-run", default="full")
    ap.add_argument("--vision-run", default="vision_full")
    a = ap.parse_args()
    doc = compare_contexts(find_context_dir(RESULTS_DIR / a.eeg_run, "eeg"), find_context_dir(RESULTS_DIR / a.vision_run, "vision"))
    if not doc["available"]:
        print("Not available yet: " + "; ".join(doc["missing"]))
        return 1
    doc, path = save_comparison(a.eeg_run, a.vision_run)
    print(doc["framing"])
    print(f"EEG recommends {doc['eeg']['recommended']}, vision recommends {doc['vision']['recommended']} "
          f"(same: {doc['same_recommendation']}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
