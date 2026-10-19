"""Spectral features from the Welch power spectral density (family prefix `spectral_`).

For every epoch and channel:
  * absolute band power  (log10 uV^2)   delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
  * relative band power  (fraction of 1-45 Hz power; the five bands tile 1-45 Hz, so they sum to 1)
  * band ratios          (log10) theta/alpha, theta/beta, alpha/beta
Plus frontal alpha asymmetry (ln alpha right - ln alpha left) for F4-F3, AF4-AF3, F8-F7.

Logs are used because band powers are strongly right-skewed; log values suit linear models.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
RATIOS = [("theta", "alpha"), ("theta", "beta"), ("alpha", "beta")]
ASYMMETRY_PAIRS = [("F4", "F3"), ("AF4", "AF3"), ("F8", "F7")]  # (right, left)


def welch_psd(X: np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None):
    """PSD for epochs X (n_epochs, channels, samples) -> freqs, psd (n_epochs, channels, n_freqs)."""
    nperseg = min(nperseg, X.shape[-1])
    return welch(X, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)


def band_power(freqs: np.ndarray, psd: np.ndarray, bands: dict = BANDS) -> np.ndarray:
    """Absolute band power (uV^2) by integrating the PSD -> (n_epochs, channels, n_bands).

    Bins are assigned half-open [lo, hi) so adjacent bands never double count; the last band
    includes its upper edge.
    """
    df = freqs[1] - freqs[0]
    names = list(bands)
    out = np.empty(psd.shape[:-1] + (len(names),))
    for b, name in enumerate(names):
        lo, hi = bands[name]
        sel = (freqs >= lo) & ((freqs < hi) if b < len(names) - 1 else (freqs <= hi))
        out[..., b] = psd[..., sel].sum(axis=-1) * df
    return out


def spectral_features(freqs, psd, ch_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return ({column: values}, band_power array) for one batch of epochs."""
    bp = band_power(freqs, psd)                        # (N, ch, 5)
    rel = bp / (bp.sum(axis=-1, keepdims=True) + 1e-20)
    log_bp = np.log10(bp + 1e-12)
    names = list(BANDS)
    cols: dict[str, np.ndarray] = {}
    for b, band in enumerate(names):
        for k, ch in enumerate(ch_names):
            cols[f"spectral_abs_{band}_{ch}"] = log_bp[:, k, b]
    for b, band in enumerate(names):
        for k, ch in enumerate(ch_names):
            cols[f"spectral_rel_{band}_{ch}"] = rel[:, k, b]
    for num, den in RATIOS:
        i, j = names.index(num), names.index(den)
        for k, ch in enumerate(ch_names):
            cols[f"spectral_ratio_{num}_{den}_{ch}"] = log_bp[:, k, i] - log_bp[:, k, j]
    a = names.index("alpha")
    for right, left in ASYMMETRY_PAIRS:
        if right in ch_names and left in ch_names:
            r, l = ch_names.index(right), ch_names.index(left)
            cols[f"spectral_asym_alpha_{right}-{left}"] = np.log(bp[:, r, a] + 1e-12) - np.log(bp[:, l, a] + 1e-12)
    return cols, bp
