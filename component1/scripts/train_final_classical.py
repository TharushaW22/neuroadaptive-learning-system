"""Train deployable classical models on ALL labelled subjects (for the inference engine / API).

The CV results in results/ are what we report; these final models are what we deploy. They are
trained without the rest baseline (a live session may not have one) on the chosen feature set.

Usage:  python scripts/train_final_classical.py [--model xgb] [--feature-set all] [--tasks ...]
Writes: models/<T>_<model>_<feature_set>.joblib + .json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import joblib
import numpy as np

from eegengine.config import PROJECT_ROOT, RESULTS_DIR
from eegengine.data import load_features
from eegengine.features.extract import family_columns
from eegengine.labels import TASKS, task_view
from eegengine.models.classical import fit_pipeline, make_pipeline
from eegengine.training.classical_runner import FEATURE_SETS, FoldModel, experiment_id


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--tasks", nargs="*", default=["workload_binary", "workload_within_task"])
    args = ap.parse_args()
    model = args.model
    if model is None:
        with open(RESULTS_DIR / "stats_classical.json", encoding="utf-8") as f:
            model = json.load(f)["best_model"]

    F, meta, _ = load_features()
    cols = family_columns(F.columns, FEATURE_SETS[args.feature_set])
    for task in args.tasks:
        mask, y = task_view(meta, task)
        X = F[cols].to_numpy(np.float32)[mask]
        pipe = fit_pipeline(make_pipeline(model, 42, n_jobs=8), X, y)
        fm = FoldModel(pipe, np.arange(len(TASKS[task]["classes"])), len(TASKS[task]["classes"]))
        short = TASKS[task]["short"]
        eid = experiment_id(task, model, args.feature_set, "none", "loso")
        cv = {}
        p = RESULTS_DIR / eid / "metrics.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            cv = {"experiment_id": eid, "per_fold_macro_f1": r["per_fold"].get("macro_f1"), "pooled": r["pooled"],
                  "recording_level": r["recording_level"]}
        version = f"{model}-{args.feature_set}-{short}-{dt.date.today():%Y%m%d}"
        info = {"version": version, "task": task, "classes": TASKS[task]["classes"], "model": model,
                "feature_set": args.feature_set, "normalisation": "none", "n_train_epochs": int(len(y)),
                "n_train_subjects": int(meta[mask]["subject_id"].nunique()), "cv_summary": cv}
        out = PROJECT_ROOT / "models" / f"{short}_{model}_{args.feature_set}"
        out.parent.mkdir(exist_ok=True)
        joblib.dump({**info, "estimator": fm, "columns": cols}, out.with_suffix(".joblib"))
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"saved {out.with_suffix('.joblib').name}  ({len(y)} epochs, {len(cols)} features)")


if __name__ == "__main__":
    main()
