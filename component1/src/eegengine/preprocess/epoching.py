"""Cut a continuous (channels x samples) recording into fixed-length, overlapping epochs."""

from __future__ import annotations

import numpy as np

from eegengine.quality.checks import window_starts


def make_epochs(data: np.ndarray, fs: float, window_s: float = 4.0, step_s: float = 2.0):
    """Return (epochs, starts): epochs has shape (n_epochs, channels, window_samples).

    4 s windows with a 2 s step = 50% overlap -> 74 epochs per 150 s STEW recording.
    Uses a strided view + copy, so it is fast and the result owns its memory.
    """
    win = int(round(window_s * fs))
    starts = window_starts(data.shape[1], fs, window_s, step_s)
    view = np.lib.stride_tricks.sliding_window_view(data, win, axis=1)  # (ch, n_pos, win)
    epochs = view[:, starts, :].transpose(1, 0, 2).copy()
    return epochs, starts
