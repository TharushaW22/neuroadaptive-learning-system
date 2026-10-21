"""Sparsity: is the explanation short and focused (a few features carry most of the weight)?

gini = Gini index of |attribution| (0 = all features equal, -> 1 = one feature carries everything)
n_mass = how many features are needed to reach `mass_fraction` (e.g. 80 %) of the total |attribution|
Higher gini / lower n_mass = easier to read. Sparsity says nothing about correctness on its own.
"""

from __future__ import annotations

import numpy as np


def gini(a: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(a, dtype=float)))
    n = len(x)
    if x.sum() <= 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float((2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n)


def n_mass(a: np.ndarray, mass_fraction: float = 0.8) -> int:
    x = np.sort(np.abs(np.asarray(a, dtype=float)))[::-1]
    if x.sum() <= 0:
        return len(x)
    return int(np.searchsorted(np.cumsum(x) / x.sum(), mass_fraction) + 1)


def sparsity(a: np.ndarray, mass_fraction: float = 0.8) -> dict:
    return {"sparsity": gini(a), "n_features_80": n_mass(a, mass_fraction)}
