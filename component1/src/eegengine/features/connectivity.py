"""Functional connectivity per frequency band: PLV and magnitude-squared coherence.

PLV (phase-locking value), per pair of channels i, j:
    1. band-pass the epoch (zero-phase Butterworth), 2. phase phi(t) from the Hilbert transform,
    3. PLV_ij = | mean_t exp(i (phi_i(t) - phi_j(t))) |      0 = random phase, 1 = locked phase.
Magnitude-squared coherence: |S_ij(f)|^2 / (S_ii(f) S_jj(f)) from Welch cross-spectra
(Hann window, 1 s segments, 50 % overlap), averaged over the frequency bins of the band.

Both produce symmetric 14x14 matrices with ones on the diagonal. They feed the GNN and the
dashboard as matrices, and the classical models as the 91 upper-triangle values per band.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, get_window, hilbert, sosfiltfilt

from eegengine.features.spectral import BANDS


def band_filter(X: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def plv_matrix(Xb: np.ndarray) -> np.ndarray:
    """PLV for band-filtered epochs Xb (N, ch, T) -> (N, ch, ch)."""
    phase = np.angle(hilbert(Xb, axis=-1))
    z = np.exp(1j * phase)                                      # unit phasors
    return np.abs(np.einsum("nit,njt->nij", z, z.conj())) / Xb.shape[-1]


def coherence_matrix(X: np.ndarray, fs: float, bands: dict[str, tuple[float, float]],
                     nperseg: int = 128, chunk: int = 256) -> dict[str, np.ndarray]:
    """Magnitude-squared coherence per band -> {band: (N, ch, ch)}."""
    n, ch, T = X.shape
    nperseg = min(nperseg, T)
    step = nperseg // 2
    starts = np.arange(0, T - nperseg + 1, step)
    win = get_window("hann", nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    out = {b: np.empty((n, ch, ch), dtype=np.float32) for b in bands}
    for c0 in range(0, n, chunk):                               # chunks keep memory small
        seg = np.stack([X[c0:c0 + chunk, :, s:s + nperseg] for s in starts], axis=2)  # (n, ch, seg, nperseg)
        seg = seg - seg.mean(axis=-1, keepdims=True)
        F = np.fft.rfft(seg * win, axis=-1)                    # (n, ch, seg, f)
        S = np.einsum("nisf,njsf->nijf", F, F.conj()) / len(starts)   # cross-spectral density
        auto = np.real(np.einsum("niif->nif", S))
        coh = np.abs(S) ** 2 / (auto[:, :, None, :] * auto[:, None, :, :] + 1e-30)
        for b, (lo, hi) in bands.items():
            sel = (freqs >= lo) & (freqs <= hi)
            out[b][c0:c0 + chunk] = coh[..., sel].mean(axis=-1)
    return out


def connectivity_matrices(X: np.ndarray, fs: float, band_names: list[str], coh_nperseg: int = 128,
                          filter_order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (plv, coh), each (N, n_bands, ch, ch) float32, bands in `band_names` order."""
    bands = {b: BANDS[b] for b in band_names}
    plv = np.stack([plv_matrix(band_filter(X, fs, bands[b], filter_order)) for b in band_names], axis=1)
    coh_d = coherence_matrix(X, fs, bands, coh_nperseg)
    coh = np.stack([coh_d[b] for b in band_names], axis=1)
    return plv.astype(np.float32), coh.astype(np.float32)


def upper_triangle_columns(M: np.ndarray, prefix: str, band_names: list[str], ch_names: list[str]) -> dict:
    """Flatten (N, bands, ch, ch) matrices to named upper-triangle columns."""
    iu = np.triu_indices(len(ch_names), k=1)
    cols = {}
    for b, band in enumerate(band_names):
        vals = M[:, b][:, iu[0], iu[1]]
        for p, (i, j) in enumerate(zip(*iu)):
            cols[f"{prefix}_{band}_{ch_names[i]}-{ch_names[j]}"] = vals[:, p]
    return cols
