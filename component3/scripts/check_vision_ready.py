"""Is Component 2's trained model (and its data) ready for the vision context?  READ-ONLY.

  .venv/bin/python scripts/check_vision_ready.py            # checklist; exit code 0 only when vision can run
  .venv/bin/python scripts/check_vision_ready.py --json     # same, as JSON
  .venv/bin/python scripts/check_vision_ready.py --no-accuracy   # skip scoring the test windows

Reads component2/vision_cbu/{models, data/processed/features, configs/default.yaml}; writes nothing.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401

from xaicomp.config import load_context
from xaicomp.run.vision_readiness import check_vision_ready, format_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="vision_engagement_gru")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    a = ap.parse_args()
    res = check_vision_ready(load_context(a.context), compute_accuracy=not a.no_accuracy)
    print(json.dumps(res, indent=1) if a.json else format_report(res))
    return 0 if res["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
