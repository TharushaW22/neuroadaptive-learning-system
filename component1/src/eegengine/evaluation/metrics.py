"""Evaluation metrics: macro-F1, balanced accuracy, precision, recall, AUROC, confusion, ECE.

Three views of the same out-of-fold predictions are reported:
  * per fold   : metrics inside each test fold, then mean ± std across folds (only folds where
                 the metric is defined — e.g. AUROC needs both classes in the fold)
  * pooled     : metrics over all out-of-fold predictions at once
  * recording  : the epoch probabilities of each recording are averaged into one prediction per
                 recording (for T3 this is one prediction per subject)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """All scalar metrics for one set of predictions. proba: (N, C)."""
    y = np.asarray(y)
    pred = proba.argmax(axis=1)
    labels = list(range(n_classes))
    out = {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float((pred == y).mean()),
        "precision_macro": float(precision_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "auroc": auroc(y, proba, n_classes),
    }
    return out


def auroc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Binary AUROC, or macro one-vs-rest AUROC over the classes present (NaN if undefined)."""
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    if n_classes == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    if len(present) < n_classes:
        # OvR over the present classes only
        aucs = [roc_auc_score((y == c).astype(int), proba[:, c]) for c in present]
        return float(np.mean(aucs))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes))))


def confusion(y, proba, n_classes: int) -> list[list[int]]:
    return confusion_matrix(y, proba.argmax(axis=1), labels=list(range(n_classes))).tolist()


def mean_std(rows: list[dict], keys=("macro_f1", "balanced_accuracy", "precision_macro", "recall_macro", "auroc", "accuracy")) -> dict:
    df = pd.DataFrame(rows)
    out = {}
    for k in keys:
        if k in df:
            v = df[k].dropna()
            out[k] = {"mean": float(v.mean()) if len(v) else float("nan"),
                      "std": float(v.std(ddof=0)) if len(v) else float("nan"), "n_folds": int(len(v))}
    return out


def recording_level(pred_df: pd.DataFrame, n_classes: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Average epoch probabilities per recording -> (frame, y, proba)."""
    pcols = [f"p{c}" for c in range(n_classes)]
    g = pred_df.groupby(["subject_id", "condition"])
    rec = g[pcols].mean()
    rec["y"] = g["y"].first()
    return rec.reset_index(), rec["y"].to_numpy(), rec[pcols].to_numpy()


def per_subject_scores(pred_df: pd.DataFrame, n_classes: int, metric: str = "macro_f1") -> pd.Series:
    """Per-subject score for paired tests (Wilcoxon). macro-F1 needs both classes per subject;
    when a subject has a single class (T3) use metric='accuracy'."""
    pcols = [f"p{c}" for c in range(n_classes)]
    out = {}
    for s, g in pred_df.groupby("subject_id"):
        proba = g[pcols].to_numpy()
        pred = proba.argmax(axis=1)
        if metric == "accuracy":
            out[s] = float((pred == g["y"].to_numpy()).mean())
        else:
            out[s] = float(f1_score(g["y"], pred, average="macro", zero_division=0))
    return pd.Series(out, name=metric)


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> tuple[float, dict]:
    """ECE with equal-width confidence bins; also returns the reliability-diagram data."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, {"bin_lower": [], "bin_upper": [], "confidence": [], "accuracy": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        for k, v in zip(bins, (lo, hi, cf, acc, int(m.sum()))):
            bins[k].append(float(v))
    return float(ece), bins
