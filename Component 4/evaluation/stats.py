"""Paired t-test + Wilcoxon + Cohen's d + 95% CI."""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    t, p = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = float("nan"), float("nan")
    diff = a - b
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    ci = stats.t.interval(0.95, len(diff) - 1,
                          loc=diff.mean(), scale=stats.sem(diff))
    return {"t": float(t), "p": float(p),
            "wilcoxon_stat": float(w), "wilcoxon_p": float(pw),
            "mean_diff": float(diff.mean()), "cohens_d": float(d),
            "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            "n": int(len(diff))}


def check_significance(result, alpha=0.05) -> bool:
    return result["p"] < alpha