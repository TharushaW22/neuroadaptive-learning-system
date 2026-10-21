"""Generate SYNTHETIC feature data in data/demo/ (and optionally DEMO ONLY models + report).

    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --train-demo-models --evaluate
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from src.data.synthetic import generate_synthetic_features, write_synthetic
from src.utils.config import load_config, resolve_path, set_seed
from src.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--participants", type=int, default=12)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--train-demo-models", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    setup_logging()
    overrides = {"temporal": {"epochs": a.epochs}} if a.epochs else None
    cfg = load_config(a.config, overrides)
    set_seed(cfg["project"]["seed"])
    df = generate_synthetic_features(a.participants, a.videos, a.seconds, cfg["extraction"]["target_fps"],
                                     cfg["project"]["seed"])
    path = write_synthetic(df, resolve_path(cfg, "demo_dir"))
    print(f"SYNTHETIC data: {len(df)} rows, {df['video_id'].nunique()} videos -> {path}")
    if a.train_demo_models:
        from src.models.train import train_demo
        for kind in ("baseline", "temporal"):
            meta = train_demo(cfg, kind)
            print(f"DEMO ONLY {kind} model -> {meta['path']}")
    if a.evaluate:
        from src.evaluation.reports import DEMO_BANNER, run_evaluation
        rep = run_evaluation(cfg, demo=True)
        print(f"{DEMO_BANNER}: report -> {rep['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
