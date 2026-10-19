"""Zero-phase FIR band-pass and notch filtering (MNE implementations, applied to plain arrays).

`mne.filter.filter_data` designs a windowed-sinc (firwin, Hamming) FIR filter and applies it
forwards-backwards-equivalent with phase="zero", so peaks are not shifted in time. Arrays stay
in microvolts — the filters are linear, so units do not matter.
"""

from __future__ import annotations

import mne
import numpy as np


def bandpass(data: np.ndarray, fs: float, l_freq: float = 0.5, h_freq: float = 45.0) -> np.ndarray:
    """Zero-phase FIR band-pass (channels x samples)."""
    return mne.filter.filter_data(
        data.astype(np.float64), sfreq=fs, l_freq=l_freq, h_freq=h_freq,
        method="fir", phase="zero", fir_design="firwin", pad="reflect_limited", verbose="ERROR",
    )


def notch(data: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    """Zero-phase FIR notch at `freq` (skipped automatically if above Nyquist)."""
    if freq >= fs / 2:
        return data
    return mne.filter.notch_filter(
        data.astype(np.float64), Fs=fs, freqs=[freq], method="fir", phase="zero",
        fir_design="firwin", pad="reflect_limited", verbose="ERROR",
    )


def filter_recording(data: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Mean removal -> band-pass -> optional notch, as configured in preprocess.yaml `filter`."""
    out = data - data.mean(axis=1, keepdims=True)  # DC offset (~4000 uV on Emotiv)
    out = bandpass(out, fs, cfg.get("l_freq", 0.5), cfg.get("h_freq", 45.0))
    if cfg.get("notch_enabled", True):
        out = notch(out, fs, cfg.get("notch_freq", 50.0))
    return out
