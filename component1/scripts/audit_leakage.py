"""Leakage audit: checks that can be re-run at any time. Writes results/leakage_audit.json.

1. Fold integrity — from every saved predictions.csv: in LOSO the fold id equals the test subject;
   in 5-fold runs each subject's fold matches results/folds_5fold.json. A subject therefore never
   appears in the test fold of a run that also trained on it.
2. Label-permutation test — the full LOSO pipeline (scaler + LR, all features) on T1 with the
   lo/hi labels randomly swapped *within each subject* (so the class balance per subject is kept).
   If the pipeline leaked information it could still score above chance; it must score ~0.5.
3. Subject-identity probe — can the same features identify the SUBJECT in a subject-wise split?
   (They cannot, by construction: the test subject is unseen. Reported to show why random splits
   would be misleading: with a random epoch split, subject ID is easy to predict.)
4. Feature locality — features are computed per epoch: recomputing one epoch alone gives the same
   values as the batch extraction (no cross-epoch statistics).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline as sk_pipeline
from sklearn.preprocessing import StandardScaler

from eegengine.config import RESULTS_DIR, load_config
from eegengine.data import load_epochs, load_features
from eegengine.features.extract import extract_features
from eegengine.io.stew import CHANNELS, FS
from eegengine.training.classical_runner import run_classical
from eegengine.training.cv import load_assignment


def fold_integrity() -> dict:
    assign = load_assignment(RESULTS_DIR / "folds_5fold.json") if (RESULTS_DIR / "folds_5fold.json").exists() else {}
    checked, problems = 0, []
    for p in sorted(RESULTS_DIR.glob("*/predictions.csv")):
        exp = p.parent.name
        m = json.load(open(p.parent / "metrics.json", encoding="utf-8"))
        df = pd.read_csv(p, usecols=["subject_id", "fold"])
        if m["cv"] == "loso":
            bad = df[df["subject_id"] != df["fold"]]
        else:
            bad = df[df["subject_id"].map(assign) != df["fold"]]
        checked += 1
        if len(bad):
            problems.append({"experiment": exp, "n_rows": int(len(bad))})
    return {"experiments_checked": checked, "problems": problems, "ok": not problems}


def permutation_test(F, meta, n_perm: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    scores = []
    for k in range(n_perm):
        m = meta.copy()
        flip = {s: bool(rng.integers(0, 2)) for s in m["subject_id"].unique()}   # swap lo/hi labels per subject
        m["y_binary"] = [(1 - y) if flip[s] else y for s, y in zip(m["subject_id"], m["y_binary"])]
        # an additional within-subject shuffle of epochs breaks any remaining label/time alignment
        for s, idx in m.groupby("subject_id").indices.items():
            m.loc[m.index[idx], "y_binary"] = rng.permutation(m["y_binary"].to_numpy()[idx])
        r = run_classical(F, m, task="workload_binary", model="lr", feature_set="all", normalisation="none",
                          cv="loso", out_dir=RESULTS_DIR / "_audit" / f"perm_{k}", seed=seed + k, n_jobs=4)
        scores.append({"pooled_balanced_accuracy": r["pooled"]["balanced_accuracy"],
                       "pooled_auroc": r["pooled"]["auroc"],
                       "fold_macro_f1_mean": r["per_fold"]["macro_f1"]["mean"]})
    return {"n_permutations": n_perm, "runs": scores,
            "mean_balanced_accuracy": float(np.mean([s["pooled_balanced_accuracy"] for s in scores])),
            "expected": "≈ 0.5 if there is no leakage"}


def subject_identity_probe(F, meta) -> dict:
    """Random (non-subject-wise) 5-fold split: how well can LR tell WHICH subject an epoch is from?"""
    X = F.to_numpy(np.float32)
    y = meta["subject_id"].to_numpy()
    pipe = sk_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    acc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=4)
    return {"random_split_subject_id_accuracy": float(acc.mean()), "chance": 1 / 48,
            "meaning": "epochs carry a strong subject fingerprint, so random epoch splits would leak identity; "
                       "all reported results use subject-wise splits instead"}


def feature_locality(n: int = 5) -> dict:
    fcfg = load_config("features.yaml")
    X, _ = load_epochs(fcfg["preprocess_config"])
    F, _, _ = load_features()
    idx = [0, 1000, 3000, 5000, 7000][:n]
    single = pd.concat([extract_features(X[i:i + 1], FS, CHANNELS, fcfg)[0] for i in idx], ignore_index=True)
    diff = np.abs(single.to_numpy() - F.iloc[idx].to_numpy()).max()
    return {"epochs_checked": idx, "max_abs_difference": float(diff), "ok": bool(diff < 1e-4)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    F, meta, _ = load_features()
    out = {"fold_integrity": fold_integrity()}
    print("fold integrity:", out["fold_integrity"])
    out["feature_locality"] = feature_locality()
    print("feature locality:", out["feature_locality"])
    out["permutation_test"] = permutation_test(F, meta)
    print("permutation:", out["permutation_test"]["mean_balanced_accuracy"])
    out["subject_identity_probe"] = subject_identity_probe(F, meta)
    print("subject id probe:", out["subject_identity_probe"]["random_split_subject_id_accuracy"])
    high = []
    for p in sorted(RESULTS_DIR.glob("*/metrics.json")):
        r = json.load(open(p, encoding="utf-8"))
        views = {"pooled": r.get("pooled"), "recording_level": r.get("recording_level"),
                 "sequence_level": (r.get("sequence_level") or {}).get("pooled")}
        for view, m in views.items():
            ba = (m or {}).get("balanced_accuracy")
            if ba and ba > 0.97 and not p.parent.name.startswith("_"):
                high.append({"experiment": r["experiment_id"], "view": view, "balanced_accuracy": ba,
                             "normalisation": r.get("normalisation")})
    out["results_above_97pct"] = high
    with open(RESULTS_DIR / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("results > 0.97:", high)


if __name__ == "__main__":
    main()
