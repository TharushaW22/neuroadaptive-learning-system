"""Scikit-learn baseline: window statistics (mean/std/min/max per feature) + classifier."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AGG_STATS = ("mean", "std", "min", "max")


def aggregate_windows(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,T,F) + (N,T) -> (N, 4F). Padded / NaN timesteps are ignored."""
    if X.ndim != 3:
        raise ValueError(f"Expected (N,T,F), got {X.shape}")
    Xn = X.astype(np.float64).copy()
    Xn[mask < 0.5] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = [np.nanmean(Xn, 1), np.nanstd(Xn, 1), np.nanmin(Xn, 1), np.nanmax(Xn, 1)]
    return np.concatenate(feats, axis=1).astype(np.float32)


def aggregated_feature_names(feature_columns: list[str]) -> list[str]:
    return [f"{c}_{s}" for s in AGG_STATS for c in feature_columns]


def build_baseline(cfg: dict, seed: int = 42) -> Pipeline:
    b = cfg["baseline"]
    cw = b.get("class_weight") or None
    if b["type"] == "logistic_regression":
        clf = LogisticRegression(max_iter=int(b.get("max_iter", 2000)), class_weight=cw, random_state=seed)
    elif b["type"] == "random_forest":
        clf = RandomForestClassifier(n_estimators=int(b.get("n_estimators", 300)), class_weight=cw,
                                     random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown baseline type {b['type']}")
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("clf", clf)])


def baseline_predict_proba(model: Pipeline, X: np.ndarray, mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities over ALL n_classes (classes unseen in training get 0)."""
    proba = model.predict_proba(aggregate_windows(X, mask))
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    out[:, model.classes_.astype(int)] = proba
    return out
