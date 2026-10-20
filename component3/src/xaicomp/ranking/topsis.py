"""TOPSIS (Hwang & Yoon, 1981), vectorised over many weight / value draws.

For every draw: each criterion column is vector-normalised over the methods, multiplied by its
weight; the 'ideal' method has the best value on every criterion and the 'anti-ideal' the worst;
closeness = d(anti-ideal) / (d(ideal) + d(anti-ideal)) in [0, 1], higher = better.
Inputs are on fixed, positive scales (see criteria.py), so vector normalisation is well defined.
"""

from __future__ import annotations

import numpy as np


def topsis(V: np.ndarray, w: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """V: (..., M methods, C criteria) > 0; w: (..., C); benefit: (C,) bool -> closeness (..., M)."""
    V = np.asarray(V, dtype=float)
    norm = np.sqrt((V ** 2).sum(axis=-2, keepdims=True))
    R = np.divide(V, norm, out=np.zeros_like(V), where=norm > 0)
    Wt = R * np.asarray(w)[..., None, :]
    ideal = np.where(benefit, Wt.max(axis=-2), Wt.min(axis=-2))[..., None, :]
    anti = np.where(benefit, Wt.min(axis=-2), Wt.max(axis=-2))[..., None, :]
    dp = np.sqrt(((Wt - ideal) ** 2).sum(axis=-1))
    dm = np.sqrt(((Wt - anti) ** 2).sum(axis=-1))
    den = dp + dm
    return np.divide(dm, den, out=np.full_like(dm, 0.5), where=den > 0)
