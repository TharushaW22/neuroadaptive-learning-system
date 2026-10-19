"""Subject-wise cross-validation splitters with a built-in leakage guard.

* LOSO (leave-one-subject-out): one fold per subject — the primary protocol for classical models.
* Fixed 5-fold by subject: subjects are shuffled once with a fixed seed and dealt round-robin into
  5 folds. The SAME assignment is reused by every deep and classical model (and by every task:
  a task simply skips the subjects it has no labels for), so model comparisons are paired.

Every split passes through `assert_no_subject_overlap`, so a subject can never be in train and
test at the same time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def assert_no_subject_overlap(train_groups, test_groups) -> None:
    overlap = set(np.unique(train_groups)) & set(np.unique(test_groups))
    if overlap:
        raise AssertionError(f"LEAKAGE: subjects in both train and test: {sorted(overlap)}")


def loso_splits(groups: np.ndarray):
    """Yield (fold_id, train_idx, test_idx) — one fold per subject."""
    for s in np.unique(groups):
        test = np.where(groups == s)[0]
        train = np.where(groups != s)[0]
        assert_no_subject_overlap(groups[train], groups[test])
        yield int(s), train, test


def subject_fold_assignment(subjects, n_splits: int = 5, seed: int = 42) -> dict[int, int]:
    """Deterministic subject -> fold map (shuffle once, deal round-robin)."""
    subs = np.array(sorted(set(int(s) for s in subjects)))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    return {int(s): i % n_splits for i, s in enumerate(subs)}


def kfold_splits(groups: np.ndarray, assignment: dict[int, int]):
    """Yield (fold_id, train_idx, test_idx) from a fixed subject->fold assignment."""
    fold_of = np.array([assignment[int(g)] for g in groups])
    for f in sorted(set(assignment.values())):
        test = np.where(fold_of == f)[0]
        train = np.where(fold_of != f)[0]
        if len(test) == 0:
            continue
        assert_no_subject_overlap(groups[train], groups[test])
        yield int(f), train, test


def inner_validation_split(groups: np.ndarray, frac: float = 0.15, seed: int = 42):
    """Split TRAINING indices into (fit_idx, val_idx) by subject (for early stopping/calibration)."""
    subs = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(subs))))
    val_subs = rng.choice(subs, size=n_val, replace=False)
    val = np.isin(groups, val_subs)
    assert_no_subject_overlap(groups[~val], groups[val])
    return np.where(~val)[0], np.where(val)[0]


def save_assignment(assignment: dict[int, int], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(assignment.items())}, f, indent=1)


def load_assignment(path: Path | str) -> dict[int, int]:
    with open(path, encoding="utf-8") as f:
        return {int(k): int(v) for k, v in json.load(f).items()}
