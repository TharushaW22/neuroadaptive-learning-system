"""Train the GRU/LSTM model on REAL labelled features (refuses otherwise, exit 2).

    python scripts/train_temporal.py
    python scripts/train_temporal.py --cell lstm --hidden 128 --epochs 40 --lr 0.0005
    python scripts/train_temporal.py --demo        # synthetic, DEMO ONLY
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
    ap.add_argument("--cell", choices=["gru", "lstm"], default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument("--demo", action="store_true", help="train on SYNTHETIC data (DEMO ONLY)")
    a = ap.parse_args()
    setup_logging()
    t = {k: v for k, v in (("cell", a.cell), ("hidden_size", a.hidden), ("epochs", a.epochs),
                           ("batch_size", a.batch_size), ("learning_rate", a.lr), ("device", a.device)) if v}
    s = {k: v for k, v in (("window_size", a.window), ("stride", a.stride)) if v}
    cfg = load_config(a.config, {"temporal": t, "sequences": s})
    try:
        meta = train_demo(cfg, "temporal") if a.demo else train_real(cfg, "temporal")
    except (RealDataUnavailableError, LabelSchemeError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    h = meta["history"]
    print(f"{'DEMO ONLY ' if a.demo else ''}temporal model saved: {meta['path']}")
    print(f"Epochs run: {len(h['epochs'])}, best validation loss {h['best_val_loss']:.4f} (not a test result)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
