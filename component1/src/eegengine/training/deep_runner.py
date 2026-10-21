"""Run one deep experiment (model spec x task) over the fixed 5 subject folds and save results."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eegengine.evaluation.metrics import (classification_metrics, confusion, expected_calibration_error, mean_std,
                                          per_subject_scores, recording_level)
from eegengine.evaluation.reports import save_experiment
from eegengine.labels import TASKS, task_view
from eegengine.training.cv import assert_no_subject_overlap
from eegengine.training.trainer import measure_latency, recording_groups, run_fold, strided_items


def deep_experiment_id(task: str, name: str, suffix: str = "") -> str:
    return f"deep_{TASKS[task]['short']}_{name}_kfold5{suffix}"


def run_deep(name: str, spec: dict, cfg: dict, task: str, inputs, meta: pd.DataFrame, assignment: dict,
             out_dir: Path, log=print, subjects: list[int] | None = None) -> dict:
    mask, _ = task_view(meta, task)
    col = TASKS[task]["column"]
    y_epoch = meta[col].to_numpy()
    groups = meta["subject_id"].to_numpy()
    valid = np.where(mask)[0]
    if subjects is not None:
        valid = valid[np.isin(groups[valid], subjects)]
    n_classes = len(TASKS[task]["classes"])
    L = cfg["train"]["seq_len"]
    seed = cfg.get("seed", 42)

    fold_of = np.array([assignment[int(s)] for s in groups])
    pred_rows, fold_rows, histories, temps, ece_rows = [], [], [], [], []
    t_start = time.perf_counter()
    last = None
    for f in sorted(set(assignment.values())):
        test_rows = valid[fold_of[valid] == f]
        train_rows = valid[fold_of[valid] != f]
        if len(test_rows) == 0 or len(train_rows) == 0:
            continue
        assert_no_subject_overlap(groups[train_rows], groups[test_rows])
        test_items = strided_items(recording_groups(meta, test_rows), L, cfg["train"]["test_stride"])
        log(f"  fold {f}: train subj {len(np.unique(groups[train_rows]))}, test subj "
            f"{len(np.unique(groups[test_rows]))}, test sequences {len(test_items)}")
        res = run_fold(spec, cfg["model"], cfg["train"], cfg["augment"], inputs, meta, y_epoch, groups,
                       train_rows, test_rows, test_items, n_classes, seed + f, log)
        y_te = y_epoch[test_items[:, -1]]
        fm = classification_metrics(y_te, res["p_cal"], n_classes)
        ece_b, _ = expected_calibration_error(y_te, res["p_raw"])
        ece_a, _ = expected_calibration_error(y_te, res["p_cal"])
        fold_rows.append({"fold": f, "n_test_sequences": len(test_items), "best_epoch": res["best_epoch"],
                          "epochs_run": len(res["history"]), "train_s": round(res["train_s"], 1),
                          "temperature": res["temperature"], "ece_before": ece_b, "ece_after": ece_a, **fm})
        for h in res["history"]:
            histories.append({"fold": f, **h})
        temps.append(res["temperature"])
        first = meta.iloc[test_items[:, 0]]
        lastm = meta.iloc[test_items[:, -1]]
        df = pd.DataFrame({"subject_id": first["subject_id"].to_numpy(), "condition": first["condition"].to_numpy(),
                           "t_start": first["t_start"].to_numpy(), "t_end": lastm["t_end"].to_numpy(),
                           "fold": f, "y": y_te})
        for c in range(n_classes):
            df[f"p{c}"] = res["p_cal"][:, c]
            df[f"raw_p{c}"] = res["p_raw"][:, c]
        pred_rows.append(df)
        last = res
        log(f"  fold {f}: macro-F1 {fm['macro_f1']:.3f} bal.acc {fm['balanced_accuracy']:.3f} "
            f"AUROC {fm['auroc'] if fm['auroc'] == fm['auroc'] else float('nan'):.3f} T={res['temperature']:.2f} "
            f"ECE {ece_b:.3f}->{ece_a:.3f}")

    pred = pd.concat(pred_rows, ignore_index=True)
    pcols = [f"p{c}" for c in range(n_classes)]
    rcols = [f"raw_p{c}" for c in range(n_classes)]
    y = pred["y"].to_numpy()
    p_cal, p_raw = pred[pcols].to_numpy(), pred[rcols].to_numpy()
    ece_before, bins_before = expected_calibration_error(y, p_raw)
    ece_after, bins_after = expected_calibration_error(y, p_cal)
    rec_df, ry, rp = recording_level(pred, n_classes)
    single = (pred.groupby("subject_id")["y"].nunique() == 1).all()
    subj_metric = "accuracy" if single else "macro_f1"
    lat = measure_latency(last["model"], spec, inputs, L, threads=1)
    n_params = int(sum(p.numel() for p in last["model"].parameters()))

    result = {
        "experiment_id": out_dir.name,
        "kind": "deep",
        "task": task, "task_short": TASKS[task]["short"], "classes": TASKS[task]["classes"],
        "model": name, "model_type": spec["type"], "branches": spec.get("branches"),
        "ablation_id": spec.get("ablation_id"),
        "feature_set": "raw_epochs+graph" if spec.get("branches") and "gnn" in spec["branches"] else "raw_epochs",
        "normalisation": "train_fold_channel_scale",
        "cv": "kfold5", "n_folds": len(fold_rows), "seed": seed,
        "unit": f"sequence of {L} epochs ({(L - 1) * 2 + 4} s), stride {cfg['train']['test_stride']}",
        "n_sequences": int(len(pred)), "n_subjects": int(pred["subject_id"].nunique()),
        "n_params": n_params,
        "per_fold": mean_std(fold_rows),
        "pooled": classification_metrics(y, p_cal, n_classes),
        "recording_level": classification_metrics(ry, rp, n_classes),
        "confusion_pooled": confusion(y, p_cal, n_classes),
        "confusion_recording": confusion(ry, rp, n_classes),
        "per_subject_metric": subj_metric,
        "per_subject": {int(k): v for k, v in per_subject_scores(pred, n_classes, subj_metric).items()},
        "calibration": {"ece_before": ece_before, "ece_after": ece_after, "temperatures": temps,
                        "reliability_before": bins_before, "reliability_after": bins_after,
                        "ece_per_fold_before": mean_std(fold_rows, ("ece_before",)).get("ece_before"),
                        "ece_per_fold_after": mean_std(fold_rows, ("ece_after",)).get("ece_after")},
        "latency": {"model_ms_single_window_median": lat["median_ms"], "model_ms_p95": lat["p95_ms"],
                    "threads": lat["threads"], "window_epochs": lat["window_epochs"]},
        "train_time_s_total": round(time.perf_counter() - t_start, 1),
        "model_config": {"spec": spec, "model": cfg["model"]},
        "train_config": {"train": cfg["train"], "augment": cfg["augment"], "input": cfg.get("input")},
    }
    save_experiment(out_dir, result, pd.DataFrame(fold_rows), pred)
    pd.DataFrame(histories).to_csv(out_dir / "history.csv", index=False)
    plot_reliability(out_dir / "reliability.png", bins_before, bins_after, ece_before, ece_after, out_dir.name)
    return result


def plot_reliability(path, before, after, ece_b, ece_a, title) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color="#a3a29d", lw=1, ls="--", label="perfect calibration")
    ax.plot(before["confidence"], before["accuracy"], "o-", color="#2a78d6", ms=5, label=f"before (ECE {ece_b:.3f})")
    ax.plot(after["confidence"], after["accuracy"], "o-", color="#eb6834", ms=5, label=f"after T-scaling (ECE {ece_a:.3f})")
    ax.set_xlabel("confidence (max probability)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_checkpoint(path: Path, model, spec, cfg, task, scalers, temperature, cv_summary: dict, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "spec": spec, "model_config": cfg["model"],
        "train_config": cfg["train"], "input_config": cfg.get("input"), "task": task,
        "classes": TASKS[task]["classes"], "scalers": scalers.to_dict(), "temperature": temperature,
        "version": version, "cv_summary": cv_summary,
    }, path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"version": version, "task": task, "classes": TASKS[task]["classes"], "spec": spec,
                   "temperature": temperature, "cv_summary": cv_summary}, f, indent=2)
