"""Classification metrics and latency measurement on held-out samples."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    if len(y_true) == 0:
        raise ValueError("No held-out samples to evaluate")
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(wf),
        "per_class": [{"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                       "support": int(s[i])} for i, c in enumerate(class_names)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
        "majority_class_accuracy": float(np.bincount(y_true, minlength=len(labels)).max() / len(y_true)),
    }


def measure_latency(predict_fn, single_input, repeats: int = 50, warmup: int = 3) -> dict:
    """Wall-clock latency of predict_fn(single_input) in milliseconds (one window)."""
    for _ in range(warmup):
        predict_fn(single_input)
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        predict_fn(single_input)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"repeats": int(len(t)), "mean_ms": float(t.mean()), "p50_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95))}


def video_level(y_true, proba, video_ids) -> tuple[np.ndarray, np.ndarray]:
    """Average window probabilities per video -> video-level (y_true, y_pred)."""
    vids = np.asarray(video_ids)
    yt, yp = [], []
    for v in np.unique(vids):
        idx = vids == v
        yt.append(int(np.asarray(y_true)[idx][0]))
        yp.append(int(np.asarray(proba)[idx].mean(0).argmax()))
    return np.array(yt), np.array(yp)
