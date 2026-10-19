"""Phase 5: deep models over the fixed 5 subject folds, and final deployable checkpoints.

Usage:
  python scripts/run_deep.py --smoke                       # 4 subjects, 2 epochs, every model
  python scripts/run_deep.py [--models A10_hybrid eegnet] [--tasks workload_binary] [--skip-existing]
  python scripts/run_deep.py --final A10_hybrid --tasks workload_binary   # train on all subjects -> models/
Logs go to results/logs/deep_*.log; results to results/deep_<T>_<model>_kfold5/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import numpy as np

from eegengine.config import PROJECT_ROOT, RESULTS_DIR, load_config, resolve
from eegengine.data import load_epochs, load_features
from eegengine.labels import TASKS, task_view
from eegengine.training.cv import inner_validation_split, load_assignment, save_assignment, subject_fold_assignment
from eegengine.training.deep_runner import deep_experiment_id, run_deep, save_checkpoint
from eegengine.training.trainer import DeepInputs, edge_matrix, train_model


def ensure_folds(path, subjects, seed=42):
    if not path.exists():
        save_assignment(subject_fold_assignment(subjects, 5, seed), path)
    return load_assignment(path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/deep.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--final", nargs="*", help="train deployable checkpoints for these models on all subjects")
    args = ap.parse_args()
    cfg = load_config(args.config)

    X, meta = load_epochs(load_config(cfg["features_config"])["preprocess_config"])
    _, meta_f, arrays = load_features(cfg["features_config"])
    assert len(meta_f) == len(meta)
    inputs = DeepInputs(X, arrays["bandpower"], edge_matrix(arrays, cfg["input"]["edge_source"], cfg["input"]["edge_band"]))
    assignment = ensure_folds(resolve(cfg["folds_file"]), meta["subject_id"].unique(), cfg.get("seed", 42))

    models = args.models or list(cfg["models"])
    tasks = args.tasks or cfg["tasks"]
    logs = RESULTS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.final is not None:
        for task in tasks:
            for name in (args.final or models):
                train_final(name, cfg["models"][name], cfg, task, inputs, meta, logs)
        return

    subjects, suffix = None, ""
    if args.smoke:
        subjects, suffix = [1, 2, 3, 4, 6, 7], "_smoke"     # rated subjects so T3 has labels too
        cfg["train"]["max_epochs"] = 2

    for task in tasks:
        for name in models:
            eid = deep_experiment_id(task, name, suffix)
            out = (RESULTS_DIR / "_smoke" / eid) if args.smoke else (RESULTS_DIR / eid)
            if args.skip_existing and (out / "metrics.json").exists():
                print(f"skip {eid}", flush=True)
                continue
            logf = open(logs / f"{eid}.log", "w", encoding="utf-8")

            def log(msg, _f=logf):
                print(msg, flush=True)
                _f.write(msg + "\n")
                _f.flush()

            log(f"=== {eid} ({dt.datetime.now():%H:%M:%S})")
            t = time.perf_counter()
            r = run_deep(name, cfg["models"][name], cfg, task, inputs, meta, assignment, out, log, subjects)
            pf = r["per_fold"]
            log(f"=== {eid}: fold macro-F1 {pf['macro_f1']['mean']:.3f}±{pf['macro_f1']['std']:.3f}  "
                f"pooled F1 {r['pooled']['macro_f1']:.3f}  bal.acc {r['pooled']['balanced_accuracy']:.3f}  "
                f"AUROC {r['pooled']['auroc'] or float('nan'):.3f}  ECE {r['calibration']['ece_before']:.3f}->"
                f"{r['calibration']['ece_after']:.3f}  latency {r['latency']['model_ms_single_window_median']:.1f} ms  "
                f"[{(time.perf_counter() - t) / 60:.1f} min]")
            if (r["pooled"]["balanced_accuracy"] or 0) > 0.97:
                log(f"!! {eid}: cross-subject balanced accuracy > 0.97 — audit for leakage")
            logf.close()


def train_final(name, spec, cfg, task, inputs, meta, logs) -> None:
    """Train on all labelled subjects (inner validation split for early stopping + temperature)."""
    mask, _ = task_view(meta, task)
    y_epoch = meta[TASKS[task]["column"]].to_numpy()
    groups = meta["subject_id"].to_numpy()
    rows = np.where(mask)[0]
    fit_l, val_l = inner_validation_split(groups[rows], cfg["train"].get("val_frac", 0.15), cfg.get("seed", 42))
    log_path = logs / f"final_{TASKS[task]['short']}_{name}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        def log(msg):
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()
        log(f"=== final {name} {task}: fit {len(fit_l)} epochs, val {len(val_l)} epochs")
        model, T, hist, best_ep, *_ = train_model(spec, cfg["model"], cfg["train"], cfg["augment"], inputs, meta,
                                                   y_epoch, rows[fit_l], rows[val_l], len(TASKS[task]["classes"]),
                                                   cfg.get("seed", 42), log)
    cv_path = RESULTS_DIR / deep_experiment_id(task, name) / "metrics.json"
    cv_summary = {}
    if cv_path.exists():
        import json
        with open(cv_path, encoding="utf-8") as f:
            r = json.load(f)
        cv_summary = {"experiment_id": r["experiment_id"], "per_fold_macro_f1": r["per_fold"]["macro_f1"],
                      "pooled": r["pooled"], "ece_after": r["calibration"]["ece_after"]}
    version = f"{name}-{TASKS[task]['short']}-{dt.date.today():%Y%m%d}"
    path = PROJECT_ROOT / "models" / f"{TASKS[task]['short']}_{name}.pt"
    save_checkpoint(path, model, spec, cfg, task, inputs.scalers, T, cv_summary, version)
    print(f"saved {path} (T={T:.2f}, best epoch {best_ep})")


if __name__ == "__main__":
    main()
