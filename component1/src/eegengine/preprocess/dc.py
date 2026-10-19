"""DC-offset removal (the first step for every STEW signal).

Emotiv EPOC values sit on a ~4000 uV offset. We remove it in two steps:
  1. subtract each channel's mean (kills the constant offset), then
  2. apply a zero-phase high-pass filter (removes slow drift that the mean cannot).
A Butterworth IIR filter run forwards and backwards (sosfiltfilt) gives zero phase shift.
The full 0.5-45 Hz FIR band-pass of Phase 2 is applied later in the preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def remove_dc(data: np.ndarray, fs: float, highpass_hz: float | None = 0.5, order: int = 4) -> np.ndarray:
    """Return a DC-free copy of `data` (channels x samples).

    highpass_hz=None only subtracts the mean.
    """
    out = data - data.mean(axis=-1, keepdims=True)
    if highpass_hz:
        sos = butter(order, highpass_hz, btype="highpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out, axis=-1)
    return out
