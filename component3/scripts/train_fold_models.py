"""Step 2 / D1: retrain per-fold models with Component 1's unchanged run_fold, into component3/artifacts/.

Full retraining (all 5 folds, C1's full budget; run it yourself, ~10-20 min on a laptop CPU):
    .venv/bin/python scripts/train_fold_models.py --context eeg_T1_A8
Smoke test (one fold, tiny epoch budget, written to a SEPARATE test folder so it is never used):
    .venv/bin/python scripts/train_fold_models.py --context eeg_T1_A8 --folds 0 --max-epochs 1 --test
"""
import argparse
import json

import _bootstrap  # noqa: F401

from xaicomp.config import load_context
from xaicomp.data.eeg_data import load_data
from xaicomp.models.fold_training import train_fold_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="eeg_T1_A8")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--max-epochs", type=int, help="override C1's max_epochs (tests only)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--test", action="store_true", help="write to artifacts/fold_models_test/ (never used by experiments)")
    args = ap.parse_args()
    ctx = load_context(args.context)
    out = ctx["fold_models_dir"]
    if args.test or args.max_epochs is not None:
        out = out.replace("artifacts/fold_models", "artifacts/fold_models_test")
    s = train_fold_models(load_data(), ctx["task"], ctx["model_name"], out, args.folds, args.max_epochs, args.threads)
    print(json.dumps({k: v for k, v in s.items() if k != "c1_reference"}, indent=2, default=float))
    if "done_check" in s:
        print("DONE-CHECK:", s["done_check"])


if __name__ == "__main__":
    main()
