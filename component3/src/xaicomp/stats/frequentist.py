"""Frequentist comparison of methods per criterion (subjects = blocks).

  Friedman test        do the methods differ at all? (+ Kendall's W effect size, 0..1)
  Wilcoxon signed-rank every pair of methods, Holm-corrected p-values
  rank-biserial r      matched-pairs effect size for each pair, -1..1 (sign follows 'A better than B')
  cluster bootstrap    95 % CIs of each method's mean and of each pairwise difference, resampling SUBJECTS
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


def kendalls_w(W: np.ndarray) -> float:
    n, k = W.shape
    R = np.apply_along_axis(rankdata, 1, W).sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (n ** 2 * (k ** 3 - k))) if n > 0 and k > 1 else float("nan")


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def friedman(W: pd.DataFrame) -> dict:
    if W.shape[1] < 3 or W.shape[0] < 2:
        return {"statistic": np.nan, "p_value": np.nan, "kendalls_w": np.nan, "n_subjects": len(W)}
    X = W.to_numpy()
    if np.allclose(X, X[:, :1]):
        return {"statistic": 0.0, "p_value": 1.0, "kendalls_w": 0.0, "n_subjects": len(W)}
    st, p = friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    return {"statistic": float(st), "p_value": float(p), "kendalls_w": kendalls_w(X), "n_subjects": len(W)}


def pairwise_wilcoxon(W: pd.DataFrame, direction: str, alpha: float = 0.05, correction: str = "holm") -> pd.DataFrame:
    sign = 1.0 if direction == "+" else -1.0
    rows = []
    for a, b in combinations(W.columns, 2):
        d = (W[a] - W[b]).to_numpy() * sign                    # > 0 means a is better
        if np.all(d == 0) or len(d) < 2:
            p = 1.0
        else:
            p = float(wilcoxon(d, zero_method="zsplit").pvalue)
        rows.append({"method_a": a, "method_b": b, "mean_diff_a_minus_b": float((W[a] - W[b]).mean()),
                     "p_value": p, "rank_biserial_a_better": rank_biserial(d), "n_subjects": len(d)})
    df = pd.DataFrame(rows)
    if len(df):
        rej, padj, _, _ = multipletests(df["p_value"], alpha=alpha, method=correction)
        df["p_holm"], df["significant"] = padj, rej
    return df


def cluster_bootstrap(W: pd.DataFrame, n_boot: int, ci: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap over subjects: CIs of method means and of pairwise mean differences."""
    rng = np.random.default_rng(seed)
    X = W.to_numpy()
    n = len(X)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = X[idx].mean(axis=1)                                  # (n_boot, methods)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    m_rows = [{"method": m, "mean": float(X[:, j].mean()), "ci_low": float(np.percentile(means[:, j], lo)),
               "ci_high": float(np.percentile(means[:, j], hi))} for j, m in enumerate(W.columns)]
    d_rows = []
    for (i, a), (j, b) in combinations(enumerate(W.columns), 2):
        diff = means[:, i] - means[:, j]
        d_rows.append({"method_a": a, "method_b": b, "diff": float(X[:, i].mean() - X[:, j].mean()),
                       "ci_low": float(np.percentile(diff, lo)), "ci_high": float(np.percentile(diff, hi))})
    return pd.DataFrame(m_rows), pd.DataFrame(d_rows)


def bootstrap_matrix(subject_df: pd.DataFrame, criteria: list[str], methods: list[str], n_boot: int,
                     seed: int) -> np.ndarray:
    """(n_boot, methods, criteria) of bootstrap means over subjects (for SMAA)."""
    subs = sorted(subject_df["subject_id"].unique())
    cube = np.full((len(subs), len(methods), len(criteria)), np.nan)
    for si, s in enumerate(subs):
        g = subject_df[subject_df["subject_id"] == s].set_index("method")
        for mi, m in enumerate(methods):
            if m in g.index:
                cube[si, mi] = g.loc[m, criteria].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n_boot, len(subs)))
    with np.errstate(all="ignore"):
        return np.nanmean(cube[idx], axis=1)
