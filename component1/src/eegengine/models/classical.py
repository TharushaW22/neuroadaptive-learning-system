"""Classical baselines as scikit-learn Pipelines: StandardScaler -> optional SelectKBest/PCA -> model.

The whole pipeline is fit inside each CV fold, so the scaler mean/std, the selected features and
the PCA axes are learned from training subjects only (no leakage into the test subject).

Hyper-parameters are fixed, sensible defaults (no nested tuning — see docs/decisions.md D12):
  lr   Logistic Regression, L2, C=0.1, class_weight=balanced
  svm  SVM with RBF kernel, C=1, gamma='scale', class_weight=balanced
  rf   Random Forest, 200 trees, sqrt features, min_samples_leaf=2, bootstrap 50 %, balanced_subsample
  xgb  XGBoost, 200 trees, depth 4, lr 0.1, subsample 0.8, colsample 0.3, 64 bins, balanced weights
The smaller ensembles were checked against 300-tree versions on 6 held-out subjects (T1 AUROC
0.932 vs 0.935 for RF, 0.951 vs 0.949 for XGB) and are 3-4x faster, which makes 48-fold LOSO
affordable on a CPU (D12).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

MODELS = ("lr", "svm", "rf", "xgb")
PARALLEL_FOLDS = {"lr": True, "svm": True, "rf": True, "xgb": True}  # one process per fold, 1 thread each


def make_pipeline(name: str, seed: int = 42, selection: dict | None = None, n_jobs: int = 8) -> Pipeline:
    steps = [("scale", StandardScaler())]
    sel = selection or {}
    if sel.get("method") == "kbest":
        steps.append(("select", SelectKBest(f_classif, k=sel.get("k", 200))))
    elif sel.get("method") == "pca":
        steps.append(("select", PCA(n_components=sel.get("n_components", 0.95), random_state=seed)))

    if name == "lr":
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    elif name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=2, max_samples=0.5,
                                     class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
    elif name == "xgb":
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.3, max_bin=64, tree_method="hist", n_jobs=n_jobs, random_state=seed,
                            eval_metric="logloss", verbosity=0)
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")
    steps.append(("clf", clf))
    return Pipeline(steps)


def fit_pipeline(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit; XGBoost has no class_weight, so balanced sample weights are passed instead."""
    if isinstance(pipe.named_steps["clf"], XGBClassifier):
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def predict_scores(pipe: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Class scores (N, C) that sum to 1.

    LR/RF/XGB return predict_proba. The SVM is trained without Platt scaling (5x cheaper), so its
    decision function is squashed (sigmoid / softmax). These SVM scores rank correctly (AUROC is
    valid) but are not calibrated probabilities.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba") and not isinstance(clf, SVC):
        return pipe.predict_proba(X)
    d = pipe.decision_function(X)
    if n_classes == 2:
        p1 = 1 / (1 + np.exp(-d))
        return np.column_stack([1 - p1, p1])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
