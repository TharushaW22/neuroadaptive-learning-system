"""Discrete wavelet transform features (family prefix `wavelet_`).

db4 DWT with 5 levels at 128 Hz gives these sub-bands (approximately):
  D1 32-64 Hz, D2 16-32 Hz, D3 8-16 Hz, D4 4-8 Hz, D5 2-4 Hz, A5 0-2 Hz.
Per sub-band and channel: energy (log10 sum of squared coefficients), relative energy and std.
Per channel: wavelet entropy = Shannon entropy of the relative energies across the 6 sub-bands.
"""

from __future__ import annotations

import numpy as np
import pywt


def wavelet_features(X: np.ndarray, ch_names: list[str], wavelet: str = "db4", level: int = 5) -> dict[str, np.ndarray]:
    coeffs = pywt.wavedec(X, wavelet, level=level, axis=-1)        # [A5, D5, D4, D3, D2, D1]
    names = [f"A{level}"] + [f"D{level - i}" for i in range(level)]
    energy = np.stack([(c ** 2).sum(axis=-1) for c in coeffs], axis=-1)   # (N, ch, 6)
    std = np.stack([c.std(axis=-1) for c in coeffs], axis=-1)
    rel = energy / (energy.sum(axis=-1, keepdims=True) + 1e-20)
    went = -(rel * np.log(rel + 1e-20)).sum(axis=-1)
    cols = {}
    for b, nm in enumerate(names):
        for k, ch in enumerate(ch_names):
            cols[f"wavelet_energy_{nm}_{ch}"] = np.log10(energy[:, k, b] + 1e-12)
            cols[f"wavelet_relenergy_{nm}_{ch}"] = rel[:, k, b]
            cols[f"wavelet_std_{nm}_{ch}"] = std[:, k, b]
    for k, ch in enumerate(ch_names):
        cols[f"wavelet_entropy_{ch}"] = went[:, k]
    return cols
