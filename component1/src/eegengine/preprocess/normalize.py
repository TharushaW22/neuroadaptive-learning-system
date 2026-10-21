"""Normalisation options: none | epoch_zscore | rest_baseline.

LEAKAGE NOTE (important for the viva)
`rest_baseline` scales every epoch of a subject using statistics of that subject's REST (lo)
recording. For T1 (rest vs task) this means the test subject's rest data — i.e. data from one of
the two classes — shapes the test features. It is a realistic calibration step (a user records a
short rest baseline before a session), but it can inflate T1 scores. Every T1 result is therefore
reported both with and without it. The statistics never use labels of other subjects and never
use the hi recording, so it is not cross-subject leakage, only a within-subject calibration.

Two levels are provided:
  * signal level  (epochs, channels x samples): per-channel mean/std
  * feature level (feature table)             : per-column mean/std
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("none", "epoch_zscore", "rest_baseline")


def epoch_zscore(X: np.ndarray) -> np.ndarray:
    """Z-score each epoch and channel over time (uses only the epoch itself -> no leakage)."""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mu) / sd


def rest_baseline_signal(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per subject, per channel: (x - mean_rest) / std_rest using the subject's lo epochs."""
    out = np.empty_like(X)
    for s, idx in meta.groupby("subject_id").indices.items():
        rest = idx[(meta["condition"].to_numpy()[idx] == "lo")]
        if len(rest) == 0:
            raise ValueError(f"Subject {s} has no rest epochs for baseline normalisation")
        mu = X[rest].mean(axis=(0, 2), keepdims=True)[0]
        sd = X[rest].std(axis=(0, 2), keepdims=True)[0] + 1e-8
        out[idx] = (X[idx] - mu) / sd
    return out


def normalize_epochs(X: np.ndarray, meta: pd.DataFrame, method: str) -> np.ndarray:
    if method == "none":
        return X
    if method == "epoch_zscore":
        return epoch_zscore(X)
    if method == "rest_baseline":
        return rest_baseline_signal(X, meta)
    raise ValueError(f"Unknown normalisation {method!r}; choose from {METHODS}")


def rest_baseline_features(F: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Feature-level rest baseline: z-score each feature with the subject's own rest epochs."""
    vals = F.to_numpy(dtype=np.float64)
    out = np.empty_like(vals)
    cond = meta["condition"].to_numpy()
    for s, idx in meta.groupby("subject_id").indices.items():
        rest = idx[cond[idx] == "lo"]
        mu = vals[rest].mean(axis=0)
        sd = vals[rest].std(axis=0) + 1e-8
        out[idx] = (vals[idx] - mu) / sd
    return pd.DataFrame(out, columns=F.columns, index=F.index)
