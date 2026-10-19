"""Complexity / time-domain features (family prefix `entropy_`).

Per epoch and channel:
  * spectral entropy   Shannon entropy of the normalised 1-45 Hz PSD, divided by log(n_bins) -> 0..1
                       (1 = flat/white spectrum, low = power concentrated in few frequencies)
  * Hjorth mobility    sqrt(var(x') / var(x))              ~ mean frequency
  * Hjorth complexity  mobility(x') / mobility(x)          ~ bandwidth / irregularity
  * variance (log10)   = Hjorth activity, stored once
  * skewness, kurtosis (excess)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, fmin: float = 1.0, fmax: float = 45.0) -> np.ndarray:
    sel = (freqs >= fmin) & (freqs <= fmax)
    p = psd[..., sel]
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-20)
    h = -(p * np.log(p + 1e-20)).sum(axis=-1)
    return h / np.log(sel.sum())


def hjorth(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (activity, mobility, complexity), each (n_epochs, channels)."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(axis=-1), d1.var(axis=-1), d2.var(axis=-1)
    mobility = np.sqrt(v1 / (v0 + 1e-20))
    complexity = np.sqrt(v2 / (v1 + 1e-20)) / (mobility + 1e-20)
    return v0, mobility, complexity


def entropy_features(X: np.ndarray, freqs, psd, ch_names: list[str]) -> dict[str, np.ndarray]:
    se = spectral_entropy(freqs, psd)
    act, mob, comp = hjorth(X)
    sk = skew(X, axis=-1)
    ku = kurtosis(X, axis=-1)
    cols = {}
    for name, arr in (("spec", se), ("hjorth_mobility", mob), ("hjorth_complexity", comp),
                      ("logvar", np.log10(act + 1e-12)), ("skew", sk), ("kurt", ku)):
        for k, ch in enumerate(ch_names):
            cols[f"entropy_{name}_{ch}"] = arr[:, k]
    return cols
