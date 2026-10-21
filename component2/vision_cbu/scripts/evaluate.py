"""Evaluate the latest real baseline and temporal models on the held-out TEST split.

    python scripts/evaluate.py
    python scripts/evaluate.py --no-ablation
    python scripts/evaluate.py --demo      # synthetic models/data, report under data/demo/evaluation
Prints 'NOT EVALUATED: REAL DATASET NOT AVAILABLE' and exits 2 without real data/models.
"""
import argparse
import sys

from _bootstrap import EXIT_OK, EXIT_REFUSED
from src.data.validate_dataset import RealDataUnavailableError
from src.evaluation.reports import DEMO_BANNER, NOT_EVALUATED, run_evaluation
from src.models.labels import LabelSchemeError
from src.models.registry import ModelCompatibilityError
from src.utils.config import load_config
from src.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config, {"evaluation": {"ablation": not a.no_ablation}})
    try:
        rep = run_evaluation(cfg, demo=a.demo)
    except (RealDataUnavailableError, FileNotFoundError, LabelSchemeError, ModelCompatibilityError) as exc:
        print(NOT_EVALUATED if not a.demo else "DEMO NOT EVALUATED")
        print(f"Reason: {exc}")
        return EXIT_REFUSED
    if a.demo:
        print(DEMO_BANNER)
    for c in rep["comparison"]:
        print(f"{c['model_name']}: accuracy={c['accuracy']:.3f} macro_f1={c['macro_f1']:.3f} "
              f"latency={c['latency_mean_ms']:.2f} ms/window")
    print(f"Report: {rep['output_dir']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
