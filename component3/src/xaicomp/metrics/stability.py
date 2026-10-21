"""Stability: does the method give the same explanation when it is simply run again (another seed)?

stability = mean pairwise Spearman rank correlation between the attribution vectors of the
different seeds for the same instance (1 = identical ranking). Also reported: mean pairwise
top-k Jaccard overlap of the k most important features (by |attribution|).
A deterministic method (IG with zero baseline) scores 1 trivially; it is flagged, never rewarded.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def topk_set(a: np.ndarray, k: int) -> set:
    return set(np.argsort(-np.abs(a), kind="stable")[:k].tolist())


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    A, B = topk_set(a, k), topk_set(b, k)
    return len(A & B) / len(A | B)


def stability(values_by_seed: list[np.ndarray], k: int = 10) -> dict:
    pairs = list(combinations(range(len(values_by_seed)), 2))
    if not pairs:
        return {"stability": float("nan"), "stability_topk_jaccard": float("nan"), "n_seeds": len(values_by_seed)}
    rho = [spearman(values_by_seed[i], values_by_seed[j]) for i, j in pairs]
    jac = [topk_jaccard(values_by_seed[i], values_by_seed[j], k) for i, j in pairs]
    return {"stability": float(np.nanmean(rho)) if not np.all(np.isnan(rho)) else float("nan"),
            "stability_topk_jaccard": float(np.mean(jac)), "n_seeds": len(values_by_seed)}
