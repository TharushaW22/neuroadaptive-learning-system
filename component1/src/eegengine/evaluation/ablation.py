"""Paired statistical comparison of two experiments (Wilcoxon signed-rank on per-subject scores).

Each experiment stores one score per test subject (`per_subject` in metrics.json): macro-F1 when
every subject has both classes (T1, T2), otherwise accuracy (T3, where each subject has a single
class). Two configurations are compared on the subjects they share, so the test is paired.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(res_a: dict, res_b: dict) -> dict:
    a, b = res_a["per_subject"], res_b["per_subject"]
    common = sorted(set(a) & set(b), key=int)
    xa = np.array([a[s] for s in common], dtype=float)
    xb = np.array([b[s] for s in common], dtype=float)
    diff = xa - xb
    out = {
        "a": res_a["experiment_id"], "b": res_b["experiment_id"],
        "metric": res_a.get("per_subject_metric"),
        "n_subjects": len(common),
        "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
        "mean_diff": float(diff.mean()),
        "n_a_better": int((diff > 0).sum()), "n_b_better": int((diff < 0).sum()), "n_ties": int((diff == 0).sum()),
    }
    if np.all(diff == 0):
        out.update(statistic=None, p_value=1.0)
    else:
        r = wilcoxon(xa, xb, zero_method="wilcox")
        out.update(statistic=float(r.statistic), p_value=float(r.pvalue))
    return out


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error over a set of tests)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
