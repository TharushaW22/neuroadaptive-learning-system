"""D1: retrain per-fold models with Component 1's UNCHANGED `run_fold` (imported, not copied).

Mirrors the fold loop of component1/src/eegengine/training/deep_runner.py:31-56 (run_deep):
same task mask, same folds (component1/results/folds_5fold.json, read-only), same test sequences
(strided_items, stride = test_stride), same seed (cfg seed + fold), same hyper-parameters
(component1/configs/experiments/deep.yaml). Differences, all on the OUTPUT side only:
  * run_deep only writes predictions; here each fold's trained model (state_dict, scalers,
    temperature) is ALSO saved, to component3/artifacts/fold_models/<model id>/fold<f>.pt
  * nothing is written to component1/ (no save_experiment, no plots, no logs there)

`run_fold` itself has no side effects (component1/src/eegengine/training/trainer.py:219-252);
it sets torch's thread count and seeds, which only affect this process.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xaicomp import _bootstrap  # noqa: F401
from xaicomp.config import load_c1_config, load_c3_config
from xaicomp.data.eeg_data import EEGData
from xaicomp.paths import C1_ROOT, resolve_c3, safe_path

def c1_reference_metrics(task_short: str, model_name: str) -> dict | None:
    """C1's reported CV result (read-only), e.g. pooled macro-F1 0.774 for T1 A8."""
    p = C1_ROOT / "results" / f"deep_{task_short}_{model_name}_kfold5" / "metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    return {"pooled": r["pooled"], "per_fold_macro_f1": r["per_fold"]["macro_f1"],
            "temperatures": r["calibration"]["temperatures"]}


def train_fold_models(data: EEGData, task: str, model_name: str, out_dir: Path | str, folds: list[int] | None = None,
                      max_epochs: int | None = None, threads: int | None = None, log=print) -> dict:
    """Train the requested folds; save fold<f>.pt + fold<f>_predictions.csv; return a summary."""
    from eegengine.evaluation.metrics import classification_metrics
    from eegengine.labels import TASKS, task_view
    from eegengine.training.cv import assert_no_subject_overlap, load_assignment
    from eegengine.training.trainer import DeepInputs, recording_groups, run_fold, strided_items

    dcfg = load_c3_config("data.yaml")
    cfg = copy.deepcopy(load_c1_config(dcfg["c1_deep_config"]))
    spec = cfg["models"][model_name]
    if max_epochs is not None:                     # tiny budget for tests ONLY (results flagged)
        cfg["train"]["max_epochs"] = int(max_epochs)
    if threads is not None:
        cfg["train"]["threads"] = int(threads)
    out = resolve_c3(out_dir)

    meta = data.meta
    inputs = DeepInputs(data.X, data.bandpower, data.adj)
    assignment = load_assignment(C1_ROOT / dcfg["c1_folds_file"])

    # --- identical to deep_runner.run_deep lines 31-56
    mask, _ = task_view(meta, task)
    y_epoch = meta[TASKS[task]["column"]].to_numpy()
    groups = meta["subject_id"].to_numpy()
    valid = np.where(mask)[0]
    n_classes = len(TASKS[task]["classes"])
    L = cfg["train"]["seq_len"]
    seed = cfg.get("seed", 42)
    fold_of = np.array([assignment[int(s)] for s in groups])

    summary = {"task": task, "model": model_name, "folds": {}, "max_epochs": cfg["train"]["max_epochs"],
               "tiny_budget_test": max_epochs is not None}
    for f in (folds if folds is not None else sorted(set(assignment.values()))):
        test_rows = valid[fold_of[valid] == f]
        train_rows = valid[fold_of[valid] != f]
        assert_no_subject_overlap(groups[train_rows], groups[test_rows])
        test_items = strided_items(recording_groups(meta, test_rows), L, cfg["train"]["test_stride"])
        log(f"fold {f}: train subjects {len(np.unique(groups[train_rows]))}, test subjects "
            f"{len(np.unique(groups[test_rows]))}, test sequences {len(test_items)}")
        t0 = time.perf_counter()
        res = run_fold(spec, cfg["model"], cfg["train"], cfg["augment"], inputs, meta, y_epoch, groups,
                       train_rows, test_rows, test_items, n_classes, seed + f, log)
        y_te = y_epoch[test_items[:, -1]]
        fm = classification_metrics(y_te, res["p_cal"], n_classes)

        first, last = meta.iloc[test_items[:, 0]], meta.iloc[test_items[:, -1]]
        pred = pd.DataFrame({"subject_id": first["subject_id"].to_numpy(), "condition": first["condition"].to_numpy(),
                             "t_start": first["t_start"].to_numpy(), "t_end": last["t_end"].to_numpy(),
                             "fold": f, "y": y_te})
        for c in range(n_classes):
            pred[f"p{c}"] = res["p_cal"][:, c]
            pred[f"raw_p{c}"] = res["p_raw"][:, c]
        pred.to_csv(safe_path(out / f"fold{f}_predictions.csv"), index=False, float_format="%.5f")

        torch.save({
            "state_dict": res["model"].state_dict(), "spec": spec, "model_config": cfg["model"],
            "train_config": cfg["train"], "task": task, "classes": TASKS[task]["classes"],
            "scalers": res["scalers"].to_dict(), "temperature": res["temperature"], "fold": f,
            "test_subjects": sorted(int(s) for s in np.unique(groups[test_rows])),
            "best_epoch": res["best_epoch"], "seed": seed + f,
            "version": f"{model_name}-{TASKS[task]['short']}-fold{f}-c3retrain",
            "provenance": "Component 3 retrain with eegengine.training.trainer.run_fold (unchanged)",
        }, safe_path(out / f"fold{f}.pt"))
        summary["folds"][int(f)] = {**fm, "temperature": res["temperature"], "best_epoch": res["best_epoch"],
                                    "epochs_run": len(res["history"]), "train_s": round(time.perf_counter() - t0, 1)}
        log(f"fold {f}: macro-F1 {fm['macro_f1']:.3f} T={res['temperature']:.2f} [{time.perf_counter() - t0:.0f}s]")

    # pooled out-of-fold metrics over every fold available on disk
    files = sorted(out.glob("fold*_predictions.csv"))
    if files:
        allp = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        pooled = classification_metrics(allp["y"].to_numpy(), allp[[f"p{c}" for c in range(n_classes)]].to_numpy(),
                                         n_classes)
        summary["pooled_available_folds"] = {"folds": sorted(allp["fold"].unique().tolist()), **pooled}
    ref = c1_reference_metrics(TASKS[task]["short"], model_name)
    summary["c1_reference"] = ref
    if ref and summary.get("pooled_available_folds") and len(summary["pooled_available_folds"]["folds"]) == 5:
        diff = summary["pooled_available_folds"]["macro_f1"] - ref["pooled"]["macro_f1"]
        summary["done_check"] = {"pooled_macro_f1_diff_vs_c1": diff, "tolerance": 0.03, "pass": abs(diff) <= 0.03}
    with open(safe_path(out / "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary
