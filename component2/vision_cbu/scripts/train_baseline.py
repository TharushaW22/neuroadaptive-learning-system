"""Train the scikit-learn baseline on REAL labelled features (refuses otherwise, exit 2).

    python scripts/train_baseline.py
    python scripts/train_baseline.py --model random_forest
    python scripts/train_baseline.py --demo          # synthetic, DEMO ONLY, saved under data/demo/models
"""
import argparse
import sys

from _bootstrap import EXIT_OK, EXIT_REFUSED
from src.data.validate_dataset import RealDataUnavailableError
from src.models.labels import LabelSchemeError
from src.models.train import train_demo, train_real
from src.utils.config import load_config
from src.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["logistic_regression", "random_forest"], default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="train on SYNTHETIC data (DEMO ONLY)")
    a = ap.parse_args()
    setup_logging()
    ov = {}
    if a.model:
        ov["baseline"] = {"type": a.model}
    if a.window or a.stride:
        ov["sequences"] = {k: v for k, v in (("window_size", a.window), ("stride", a.stride)) if v}
    cfg = load_config(a.config, ov)
    try:
        meta = train_demo(cfg, "baseline") if a.demo else train_real(cfg, "baseline")
    except (RealDataUnavailableError, LabelSchemeError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    print(f"{'DEMO ONLY ' if a.demo else ''}baseline saved: {meta['path']}")
    print(f"Validation accuracy (model selection only, not a test result): {meta['history']['val_accuracy']:.3f}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
