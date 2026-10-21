"""Test-retest reliability of each metric: ICC(2,1) (Shrout & Fleiss, 1979; absolute agreement).

A metric is reliable when measuring it again (another explainer seed / noise draw) gives the same
value for the same (window, method). Targets = (window, method) pairs, "raters" = repeats:
  fidelity_on / fidelity_off / sparsity / cost : the explainer seeds
  stability                                    : the seed PAIRS (needs >= 3 seeds -> 3 pairs)
  robustness                                   : the noise draws (needs n_noise >= 2)
ICC(2,1) = (MSR - MSE) / (MSR + (k-1) MSE + k (MSC - MSE) / n)
Interpretation (Koo & Li, 2016): < 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from xaicomp.metrics.stability import spearman


def icc_2_1(Y: np.ndarray) -> float:
    """Y: (n targets, k raters) without missing values."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 3 or k < 2:
        return float("nan")
    gm = Y.mean()
    ssr = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(axis=0) - gm) ** 2).sum()
    sst = ((Y - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def icc_label(v: float) -> str:
    if v != v:
        return "not assessable"
    return "poor" if v < 0.5 else "moderate" if v < 0.75 else "good" if v < 0.9 else "excellent"


def stability_pairs(explanations: dict[tuple[str, str, int], np.ndarray], seeds: list[int]) -> pd.DataFrame:
    """Per (instance, method): stability of each seed pair -> wide table (pairs as raters)."""
    rows = []
    keys = sorted({(i, m) for i, m, _ in explanations})
    for i, m in keys:
        rows.append({"instance_id": i, "method": m,
                     **{f"{a}-{b}": spearman(explanations[(i, m, a)], explanations[(i, m, b)])
                        for a, b in combinations(seeds, 2)}})
    return pd.DataFrame(rows)


def test_retest(tables: dict[str, pd.DataFrame], explanations: dict | None, seeds: list[int],
                methods: list[str]) -> pd.DataFrame:
    rows = []
    bs = tables["by_seed"]
    bs = bs[bs["method"].isin(methods)]
    for crit, g in bs.groupby("criterion"):
        W = g.pivot_table(index=["instance_id", "method"], columns="seed", values="value")
        rows.append({"criterion": crit, "icc": icc_2_1(W.to_numpy()), "repeats": "seeds", "k": W.shape[1],
                     "n_targets": len(W)})
    # stability: seed pairs as repeats
    if explanations is not None and len(seeds) >= 3:
        sp = stability_pairs({k: v for k, v in explanations.items() if k[1] in methods}, seeds)
        W = sp.drop(columns=["instance_id", "method"]).to_numpy()
        rows.append({"criterion": "stability", "icc": icc_2_1(W), "repeats": "seed pairs", "k": W.shape[1],
                     "n_targets": len(W)})
    else:
        rows.append({"criterion": "stability", "icc": np.nan, "repeats": "needs >= 3 seeds", "k": len(seeds),
                     "n_targets": 0})
    rob = tables.get("robustness")
    if rob is not None and not rob.empty and rob["noise"].nunique() >= 2:
        W = rob[rob["method"].isin(methods)].pivot_table(index=["instance_id", "method"], columns="noise",
                                                          values="robustness").to_numpy()
        rows.append({"criterion": "robustness", "icc": icc_2_1(W), "repeats": "noise draws", "k": W.shape[1],
                     "n_targets": len(W)})
    else:
        rows.append({"criterion": "robustness", "icc": np.nan, "repeats": "needs >= 2 noise draws",
                     "k": 1, "n_targets": 0})
    df = pd.DataFrame(rows)
    df["label"] = df["icc"].map(icc_label)
    return df
