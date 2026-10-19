"""Run one classical experiment: (task, model, feature set, normalisation, CV) -> results folder.

Everything that learns from data — scaler, feature selection/PCA, classifier — is fit on the
training subjects of each fold only. The only per-subject statistic computed outside the folds
is the optional REST BASELINE, which uses each subject's own rest epochs (no labels of other
subjects) — see preprocess/normalize.py for why it is reported separately.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eegengine.evaluation.metrics import classification_metrics, confusion, mean_std, per_subject_scores, recording_level
from eegengine.evaluation.reports import save_experiment
from eegengine.features.extract import family_columns
from eegengine.labels import TASKS, task_view
from eegengine.models.classical import PARALLEL_FOLDS, fit_pipeline, make_pipeline, predict_scores
from eegengine.preprocess.normalize import rest_baseline_features
from eegengine.training.cv import assert_no_subject_overlap, kfold_splits, loso_splits
from eegengine.training.sequences import recording_groups, strided_items

FEATURE_SETS = {
    "A1_spectral": ["spectral"],
    "A2_spectral_entropy": ["spectral", "entropy"],
    "A3_spectral_wavelet": ["spectral", "wavelet"],
    "A4_spectral_connectivity": ["spectral", "plv", "coh"],
    "A5_spectral_connectivity_graph": ["spectral", "plv", "coh", "graph"],
    "all": ["spectral", "entropy", "wavelet", "plv", "coh", "graph"],
}


def experiment_id(task: str, model: str, feature_set: str, norm: str, cv: str, suffix: str = "") -> str:
    """Result folder name, e.g. cls_T1_xgb_all_none_loso."""
    return f"cls_{TASKS[task]['short']}_{model}_{feature_set}_{norm}_{cv}{suffix}"


class FoldModel:
    """A fitted pipeline that may have seen only some classes; scores are mapped back to all classes."""

    def __init__(self, pipe, present: np.ndarray, n_classes: int):
        self.pipe, self.present, self.n_classes = pipe, present, n_classes

    def predict_scores(self, X):
        if self.pipe is None:                      # only one class seen in training -> constant
            p = np.ones((len(X), 1))
        else:
            p = predict_scores(self.pipe, X, len(self.present))
        full = np.zeros((len(X), self.n_classes))
        full[:, self.present] = p
        return full


def _fit_predict_fold(model, seed, selection, Xtr, ytr, Xte, n_classes, n_jobs):
    present = np.unique(ytr)                       # a fold can miss a class (e.g. T2 "high")
    y_enc = np.searchsorted(present, ytr)          # contiguous 0..k-1 (XGBoost requires it)
    pipe = make_pipeline(model, seed, selection, n_jobs=n_jobs) if len(present) > 1 else None
    t = time.perf_counter()
    if pipe is not None:
        fit_pipeline(pipe, Xtr, y_enc)
    fit_s = time.perf_counter() - t
    fm = FoldModel(pipe, present, n_classes)
    t = time.perf_counter()
    proba = fm.predict_scores(Xte)
    pred_ms = (time.perf_counter() - t) * 1000 / max(1, len(Xte))
    return proba, fit_s, pred_ms, fm


def run_classical(F: pd.DataFrame, meta: pd.DataFrame, *, task: str, model: str, feature_set: str,
                  normalisation: str, cv: str, out_dir: Path, seed: int = 42, selection: dict | None = None,
                  exclude_bad_windows: bool = False, fold_assignment: dict | None = None,
                  feature_latency_ms: float | None = None, n_jobs: int = 8, extra_config: dict | None = None,
                  seq_len: int = 8, seq_stride: int = 2) -> dict:
    families = FEATURE_SETS[feature_set]
    cols = family_columns(F.columns, families)
    Xall = F[cols]
    if normalisation == "rest_baseline":
        Xall = rest_baseline_features(Xall, meta)       # uses ALL of a subject's rest epochs (D9)
    elif normalisation != "none":
        raise ValueError(normalisation)

    mask, y = task_view(meta, task)
    X = Xall.to_numpy(dtype=np.float32)[mask]
    m = meta[mask].reset_index(drop=True)
    groups = m["subject_id"].to_numpy()
    n_classes = len(TASKS[task]["classes"])
    usable = (m["quality_flag"] != "bad").to_numpy()

    if cv == "loso":
        splits = list(loso_splits(groups))
    elif cv == "kfold5":
        splits = list(kfold_splits(groups, fold_assignment))
    else:
        raise ValueError(cv)

    def train_idx_of(tr):
        return tr[usable[tr]] if exclude_bad_windows else tr

    parallel = PARALLEL_FOLDS[model]
    t0 = time.perf_counter()
    if parallel:
        outs = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict_fold)(model, seed, selection, X[train_idx_of(tr)], y[train_idx_of(tr)], X[te], n_classes, 1)
            for _, tr, te in splits)
    else:
        outs = [_fit_predict_fold(model, seed, selection, X[train_idx_of(tr)], y[train_idx_of(tr)], X[te], n_classes, n_jobs)
                for _, tr, te in splits]
    total_s = time.perf_counter() - t0

    proba = np.zeros((len(y), n_classes))
    fold_rows = []
    fold_id = np.zeros(len(y), dtype=int)
    for (f, tr, te), (p, fit_s, pred_ms, _) in zip(splits, outs):
        assert_no_subject_overlap(groups[tr], groups[te])
        proba[te] = p
        fold_id[te] = f
        fm = classification_metrics(y[te], p, n_classes)
        fold_rows.append({"fold": f, "n_train": int(len(train_idx_of(tr))), "n_test": int(len(te)),
                          "fit_s": round(fit_s, 2), "predict_ms_per_epoch": pred_ms, **fm})

    pred_df = m[["subject_id", "condition", "epoch_index", "quality_flag"]].copy()
    pred_df["fold"] = fold_id
    pred_df["y"] = y
    for c in range(n_classes):
        pred_df[f"p{c}"] = proba[:, c]
    pred_df["pred"] = proba.argmax(axis=1)

    rec_df, ry, rp = recording_level(pred_df, n_classes)

    # Sequence view (5-fold runs): average epoch scores over the SAME 8-epoch / stride-2 windows the
    # deep models are scored on, so classical and deep results are directly comparable.
    sequence_level = None
    if cv == "kfold5":
        items = strided_items(recording_groups(pred_df, np.arange(len(pred_df))), seq_len, seq_stride)
        sp, sy, sf = proba[items].mean(axis=1), y[items[:, -1]], fold_id[items[:, 0]]
        srows = [{"fold": int(f), **classification_metrics(sy[sf == f], sp[sf == f], n_classes)} for f in np.unique(sf)]
        sdf = pd.DataFrame({"subject_id": pred_df["subject_id"].to_numpy()[items[:, 0]], "y": sy,
                            **{f"p{c}": sp[:, c] for c in range(n_classes)}})
        single = (sdf.groupby("subject_id")["y"].nunique() == 1).all()
        smetric = "accuracy" if single else "macro_f1"
        sequence_level = {"unit": f"sequence of {seq_len} epochs, stride {seq_stride}", "n_sequences": int(len(items)),
                          "per_fold": mean_std(srows), "pooled": classification_metrics(sy, sp, n_classes),
                          "confusion": confusion(sy, sp, n_classes), "per_subject_metric": smetric,
                          "per_subject": {int(k): v for k, v in per_subject_scores(sdf, n_classes, smetric).items()}}
    single_class_subjects = (pred_df.groupby("subject_id")["y"].nunique() == 1).all()
    subj_metric = "accuracy" if single_class_subjects else "macro_f1"
    subj_scores = per_subject_scores(pred_df, n_classes, subj_metric)

    # single-window latency: one epoch through the fitted pipeline of the last fold
    last_pipe = outs[-1][3]
    lat = []
    for i in range(30):
        t = time.perf_counter()
        last_pipe.predict_scores(X[i:i + 1])
        lat.append((time.perf_counter() - t) * 1000)

    fold_ok = [r for r in fold_rows if not single_class_subjects or cv != "loso"]
    result = {
        "experiment_id": out_dir.name,
        "kind": "classical",
        "task": task,
        "task_short": TASKS[task]["short"],
        "classes": TASKS[task]["classes"],
        "model": model,
        "feature_set": feature_set,
        "families": families,
        "n_features": len(cols),
        "normalisation": normalisation,
        "cv": cv,
        "n_folds": len(splits),
        "n_epochs": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "exclude_bad_windows": exclude_bad_windows,
        "selection": selection,
        "seed": seed,
        "per_fold": mean_std(fold_ok) if fold_ok else {},
        "per_fold_note": ("per-fold metrics undefined: every LOSO test fold of this task contains one class"
                          if not fold_ok else "mean ± std over folds where the metric is defined"),
        "pooled": classification_metrics(y, proba, n_classes),
        "recording_level": classification_metrics(ry, rp, n_classes),
        "confusion_pooled": confusion(y, proba, n_classes),
        "confusion_recording": confusion(ry, rp, n_classes),
        "per_subject_metric": subj_metric,
        "per_subject": {int(k): v for k, v in subj_scores.items()},
        "latency": {
            "predict_ms_per_epoch_batch": float(np.mean([r["predict_ms_per_epoch"] for r in fold_rows])),
            "predict_ms_single_window_median": float(np.median(lat)),
            "feature_extraction_ms_single_window": feature_latency_ms,
            "end_to_end_ms_single_window": (float(np.median(lat)) + feature_latency_ms) if feature_latency_ms else None,
        },
        "sequence_level": sequence_level,
        "train_time_s_total": round(total_s, 1),
        **(extra_config or {}),
    }
    save_experiment(out_dir, result, pd.DataFrame(fold_rows), pred_df)
    return result
