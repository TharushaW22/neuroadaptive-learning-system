"""Task label builder. STEW only provides WORKLOAD information.

T1 workload_binary      : lo (rest) = 0, hi (SIMKAP task) = 1.  All 48 subjects.
T2 workload_level       : own rating 1-3 low (0), 4-6 medium (1), 7-9 high (2).  Unrated -> masked.
                          CAUTION: nearly identical to the rest/task split (rest 42/3/0, task 0/20/25),
                          so T2 mostly re-measures T1.
T3 workload_within_task : hi recordings only; rating 4-6 moderate (0), 7-9 high (1). lo recordings
                          and unrated subjects are masked. Both classes come from the SAME condition,
                          so rest-vs-task differences cannot drive it: the confound-free workload task.

A masked label is stored as -1. Never drop a subject: masks decide which rows a task uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegengine.io.stew import rating_to_level

TASKS: dict[str, dict] = {
    "workload_binary": {"column": "y_binary", "classes": ["low", "high"], "short": "T1"},
    "workload_level": {"column": "y_level", "classes": ["low", "medium", "high"], "short": "T2"},
    "workload_within_task": {"column": "y_within", "classes": ["moderate", "high"], "short": "T3"},
}


def within_task_label(condition: str, rating: float) -> int:
    """T3 label: only for hi recordings with a rating. 4-6 -> 0, 7-9 -> 1, otherwise -1."""
    if condition != "hi" or rating is None or np.isnan(rating):
        return -1
    if rating <= 3:   # never happens in STEW (task ratings are 4-9) — kept masked to be safe
        return -1
    return 0 if rating <= 6 else 1


def add_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Add y_binary / y_level / y_within columns (-1 = masked) to a frame with condition + rating."""
    out = meta.copy()
    out["y_binary"] = (out["condition"] == "hi").astype(int)
    out["y_level"] = [rating_to_level(r) for r in out["rating"]]
    out["y_within"] = [within_task_label(c, r) for c, r in zip(out["condition"], out["rating"])]
    return out


def task_view(meta: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_mask, y) for a task: rows whose label is not masked, and their labels."""
    col = TASKS[task]["column"]
    y = meta[col].to_numpy()
    mask = y >= 0
    return mask, y[mask]


def label_counts(meta: pd.DataFrame, unit: str = "rows") -> dict:
    """Class counts per task (and per condition) for documentation."""
    out = {}
    for task, spec in TASKS.items():
        mask, y = task_view(meta, task)
        sub = meta[mask]
        out[task] = {
            "n": int(mask.sum()),
            "n_subjects": int(sub["subject_id"].nunique()),
            "classes": {name: int((y == i).sum()) for i, name in enumerate(spec["classes"])},
            "by_condition": {
                cond: {name: int(((sub["condition"] == cond).to_numpy() & (y == i)).sum())
                       for i, name in enumerate(spec["classes"])}
                for cond in ("lo", "hi")
            },
            "unit": unit,
        }
    return out
