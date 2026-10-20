"""Split an EEG signal into band-limited components (slow, delta ... gamma) plus a residual.

Interpretable features of Component 3 = 14 channels x 6 bands = 84 COMPONENTS of a window
(14 x 5 = 70 when the 'slow' band is switched off in the context config).
"Removing" a feature means replacing that one component; the model input is rebuilt as

    signal = sum over (channel, band) of component  +  residual

Bands: C1's five (delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz) plus, by default,
a sixth "slow" band (< 1 Hz). Measured on STEW after C1's 0.5-45 Hz FIR: about two thirds of the
window power lies below 1 Hz (wide FIR transition band; drifts, eye movements, skin potentials),
and removing it flips ~21 % of A8's predicted labels. Leaving it in the never-explained residual
would hide what the model uses, so it is an explained feature (config: representation.bands).

Filter bank: zero-phase, FFT-domain, with raised-cosine (sin^2) crossovers of width `crossover_hz`
centred on each band edge. Adjacent bands use complementary weights (sin^2 + cos^2 = 1), so the
weights sum to exactly 1 from 0 Hz (with the slow band; 1.5 Hz without it) up to 44.5 Hz.
Everything the bands do not cover (> ~45 Hz; C1 already low-passed at 45 Hz) stays in the
residual, which is never explained and never perturbed. Because residual = signal - sum(bands), the reconstruction is
exact up to float rounding; `residual_fraction` reports how much signal power lies outside the bands.

The decomposition is applied to the WHOLE continuous recording (150 s) once, then windows are
sliced out, so the 50 %-overlapping epochs of a window stay mutually consistent.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}


def _ramp(f: np.ndarray, edge: float, width: float) -> np.ndarray:
    """0 below edge-width/2, 1 above edge+width/2, sin^2 in between."""
    if width <= 0:
        return (f >= edge).astype(float)
    t = np.clip((f - (edge - width / 2)) / width, 0.0, 1.0)
    return np.sin(np.pi / 2 * t) ** 2


def band_weights(n_samples: int, fs: float, bands: dict = DEFAULT_BANDS, crossover_hz: float = 1.0) -> np.ndarray:
    """Frequency weights (n_bands, n_rfft_bins) of the filter bank. A band whose lower edge is
    <= 0 Hz (the 'slow' band) starts at DC with full weight."""
    f = np.fft.rfftfreq(n_samples, 1 / fs)
    rows = []
    for lo, hi in bands.values():
        rise = np.ones_like(f) if lo <= 0 else _ramp(f, lo, crossover_hz)
        rows.append(rise * (1 - _ramp(f, hi, crossover_hz)))
    return np.stack(rows)


def decompose(x: np.ndarray, fs: float, bands: dict = DEFAULT_BANDS, crossover_hz: float = 1.0,
              pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """x (..., T) -> (components (..., n_bands, T), residual (..., T)), float64.

    `pad` > 0 reflect-pads both ends before the FFT (used when decomposing a short window,
    to reduce wrap-around at the window edges).
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[-1]
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect") if pad else x
    W = band_weights(xp.shape[-1], fs, bands, crossover_hz)               # (B, F)
    X = np.fft.rfft(xp, axis=-1)[..., None, :]                             # (..., 1, F)
    comps = np.fft.irfft(X * W, n=xp.shape[-1], axis=-1)                   # (..., B, T')
    if pad:
        comps = comps[..., pad:pad + T]
    residual = x - comps.sum(axis=-2)
    return comps, residual


def reconstruction_error(x: np.ndarray, comps: np.ndarray, residual: np.ndarray) -> float:
    """Max |x - (sum components + residual)| (float rounding only, by construction)."""
    return float(np.max(np.abs(x - (comps.sum(axis=-2) + residual))))


def residual_fraction(x: np.ndarray, residual: np.ndarray) -> float:
    """Share of signal power (mean square) that lies outside the 5 bands (never explained)."""
    return float(np.mean(residual ** 2) / (np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20))


def log_power(comps: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """log10 mean power of each component over time: (..., T) -> (...)."""
    return np.log10(np.mean(np.asarray(comps, dtype=np.float64) ** 2, axis=-1) + eps)
