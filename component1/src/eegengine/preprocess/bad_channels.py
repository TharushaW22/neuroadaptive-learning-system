"""Detect persistently bad channels in a recording and repair them by spherical-spline interpolation.

A channel is "persistently bad" in a (filtered) recording when ANY of these holds:
  * flat        : it is flat in >= `min_bad_fraction` of the windows
  * noisy       : it fails a hard quality check (amplitude / high variance / EMG / line noise,
                  blink-like frontal deflections excluded) in >= `min_bad_fraction` of the windows
  * huge        : its MEDIAN per-window std is > `max_std_ratio` x that of the median channel
                  (the median ignores a few short spikes; those are handled by window flags)
Low variance alone is NOT a reason (a quiet channel still carries signal).

At most `max_interpolated` channels are repaired per recording (the worst first): with only 14
electrodes, interpolating more would invent too much of the scalp field. Every decision is
returned so it can be logged.
"""

from __future__ import annotations

import numpy as np

from eegengine.quality.checks import QualityConfig, assess_window, window_starts


def detect_bad_channels(data: np.ndarray, fs: float, ch_names: list[str], qcfg: QualityConfig, cfg: dict) -> list[dict]:
    """Return one dict per bad channel: {channel, reasons, bad_fraction, std_ratio}, worst first."""
    win = int(round(qcfg.window_s * fs))
    starts = window_starts(data.shape[1], fs, qcfg.window_s, qcfg.step_s)
    flat = np.zeros((len(starts), len(ch_names)), dtype=bool)
    hard = np.zeros_like(flat)
    win_std = np.zeros(flat.shape)
    for i, s in enumerate(starts):
        res = assess_window(data[:, s:s + win], fs, qcfg, ch_names)
        b = res["bad"]
        high_var = b["variance"] & (res["metrics"]["var_z"] > 0)  # only abnormally HIGH variance
        flat[i] = b["flat"]
        win_std[i] = res["metrics"]["std"]
        hard[i] = (b["amplitude"] | high_var | b["hf"] | b["line"]) & ~res["blink"]

    med_std = np.median(win_std, axis=0)               # typical amplitude of each channel
    std_ratio = med_std / (np.median(med_std) + 1e-12)
    min_frac = cfg.get("min_bad_fraction", 0.6)
    found = []
    for k, ch in enumerate(ch_names):
        reasons = []
        if flat[:, k].mean() >= min_frac:
            reasons.append("flat")
        if hard[:, k].mean() >= min_frac:
            reasons.append("noisy")
        if std_ratio[k] > cfg.get("max_std_ratio", 5.0):
            reasons.append("huge")
        if reasons:
            found.append({
                "channel": ch,
                "reasons": reasons,
                "bad_fraction": round(float(max(flat[:, k].mean(), hard[:, k].mean())), 3),
                "std_ratio": round(float(std_ratio[k]), 2),
            })
    # worst first: by std ratio, then by bad fraction
    found.sort(key=lambda d: (d["std_ratio"], d["bad_fraction"]), reverse=True)
    return found


def interpolate_channels(data: np.ndarray, fs: float, ch_names: list[str], bads: list[str]) -> np.ndarray:
    """Replace `bads` by MNE spherical-spline interpolation from the remaining channels."""
    if not bads:
        return data
    import mne

    from eegengine.io.stew import _montage_name

    info = mne.create_info(list(ch_names), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # MNE works in volts
    raw.set_montage(mne.channels.make_standard_montage(_montage_name()))
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    return raw.get_data() * 1e6


def repair_bad_channels(data, fs, ch_names, qcfg: QualityConfig, cfg: dict):
    """Detect + interpolate. Returns (repaired_data, log_dict)."""
    found = detect_bad_channels(data, fs, ch_names, qcfg, cfg)
    max_n = cfg.get("max_interpolated", 3)
    chosen = [d["channel"] for d in found[:max_n]]
    log = {
        "detected": found,
        "interpolated": chosen,
        "not_interpolated": [d["channel"] for d in found[max_n:]],  # too many -> left, windows stay flagged
    }
    if not cfg.get("interpolate", True):
        log["interpolated"], log["not_interpolated"] = [], [d["channel"] for d in found]
        return data, log
    return interpolate_channels(data, fs, ch_names, chosen), log
