"""Heuristic EEG indices for attention, fatigue and readiness (NOT trained predictions).

STEW has no labels for these states, so the engine reports established band-power indices and
marks them source = "heuristic_index". Band powers are absolute (uV^2), summed over the listed
channels before the ratio is taken.

  attention  = engagement index  beta / (alpha + theta)          over all 14 channels
               (Pope, Bogart & Bartolome, 1995)
  fatigue    = (theta + alpha) / beta                             over parietal-occipital P7, P8, O1, O2
               (Jap, Lal, Fischer & Bekiaris, 2009). Computed over the SAME channels it would be
               exactly 1 / engagement, so it uses the posterior sites where alpha/theta increase
               most with fatigue; that makes it carry information of its own.
  readiness  = 0.5 * E/(1+E) + 0.5 * 1/(1+F)                      E = attention, F = fatigue
               Each ratio is squashed to 0-1 (x/(1+x) rises with x; 1/(1+x) falls with x), then the
               two are averaged: high engagement and low fatigue -> readiness near 1. This is a
               transparent combination chosen for the dashboard, not a validated measure.
"""

from __future__ import annotations

import numpy as np

BAND_INDEX = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}
FATIGUE_CHANNELS = ("P7", "P8", "O1", "O2")

FORMULAS = {
    "attention": "engagement = beta / (alpha + theta), all channels (Pope et al., 1995)",
    "fatigue": "(theta + alpha) / beta, channels P7 P8 O1 O2 (Jap et al., 2009)",
    "readiness": "0.5*E/(1+E) + 0.5*1/(1+F), E = engagement, F = fatigue index",
}


def heuristic_indices(bandpower: np.ndarray, ch_names: list[str]) -> dict[str, float]:
    """bandpower: (channels, 5) absolute power of one window -> {attention, fatigue, readiness}."""
    bp = np.asarray(bandpower, dtype=float)
    th, al, be = (bp[:, BAND_INDEX[b]] for b in ("theta", "alpha", "beta"))
    eng = be.sum() / (al.sum() + th.sum() + 1e-12)
    post = [ch_names.index(c) for c in FATIGUE_CHANNELS if c in ch_names]
    fat = (th[post].sum() + al[post].sum()) / (be[post].sum() + 1e-12)
    readiness = 0.5 * eng / (1 + eng) + 0.5 * 1 / (1 + fat)
    return {"attention": float(eng), "fatigue": float(fat), "readiness": float(readiness)}
