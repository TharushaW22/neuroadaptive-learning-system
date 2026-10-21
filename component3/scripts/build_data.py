"""Step 1: STEW (component3/data/raw/STEW) -> component3/data/processed/ via Component 1's library.

Usage (from component3/):  .venv/bin/python scripts/build_data.py
"""
import json
import sys

import _bootstrap  # noqa: F401

from xaicomp.data.eeg_data import build_dataset, stew_files_present
from xaicomp.paths import RAW_STEW_DIR


def main() -> None:
    ok, missing = stew_files_present(RAW_STEW_DIR)
    if not ok:
        print(f"STEW missing in {RAW_STEW_DIR} ({len(missing)} files, e.g. {missing[:3]}). "
              "Place sub01_lo.txt ... sub48_hi.txt and ratings.txt there. Nothing was downloaded.")
        sys.exit(1)
    summary = build_dataset()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
