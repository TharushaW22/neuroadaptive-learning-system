"""Per-channel, per-window signal quality checks.

Input signals must already be DC-free (see `eegengine.preprocess.dc.remove_dc`), in microvolts.

For every window and channel we compute five checks:
  flat      : channel std below `flat_std_uv`           (disconnected / saturated electrode)
  amplitude : max |x| (window linearly detrended) above `max_abs_uv`   (blinks, movement, pops)
  variance  : robust z-score of log-variance ACROSS the 14 channels above `var_z_thresh`
              (one channel far noisier/quieter than the others in the same window)
  hf        : power 30-45 Hz / power 1-45 Hz above `hf_ratio_thresh`   (muscle/EMG noise).
              The Emotiv EPOC low-passes on board at ~43 Hz, so the band must sit inside the
              passband; above 45 Hz there is almost no power even in noisy recordings.
  line      : PSD at the line frequency vs. its neighbours above `line_ratio_thresh`

A channel is "bad" in a window if any check fails. Windows are only MARKED, never deleted.

Blink tolerance: eye blinks are large but harmless frontal deflections. A failure on a frontal
channel (`blink_channels`, default AF3/AF4/F7/F8) that is ONLY amplitude and/or variance, with
max |x| below `blink_max_uv`, is counted as a "blink" — it lowers the score and makes the window
"warn", but never "bad" on its own.
  score = 1 - n_bad_channels / n_channels            (blink channels included)
  n_hard = bad channels that are not blink-like
  flag  = "good" (0 bad), "bad" (n_hard > max_bad_channels_warn), otherwise "warn"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.signal import detrend, welch

CHECKS = ("flat", "amplitude", "variance", "hf", "line")


@dataclass
class QualityConfig:
    window_s: float = 4.0
    step_s: float = 2.0
    flat_std_uv: float = 0.5
    max_abs_uv: float = 150.0
    var_z_thresh: float = 3.5
    hf_band_hz: tuple[float, float] = (30.0, 45.0)
    hf_ratio_thresh: float = 0.25
    line_freq_hz: float = 50.0
    line_ratio_thresh: float = 10.0
    max_bad_channels_warn: int = 2
    blink_channels: tuple[str, ...] = ("AF3", "AF4", "F7", "F8")
    blink_max_uv: float = 300.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "QualityConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown quality config keys: {sorted(unknown)}")
        for key in ("hf_band_hz", "blink_channels"):
            if key in d:
                d = {**d, key: tuple(d[key])}
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def window_starts(n_samples: int, fs: float, window_s: float, step_s: float) -> np.ndarray:
    """Start indices of all complete windows."""
    win, step = int(round(window_s * fs)), int(round(step_s * fs))
    if win > n_samples:
        return np.array([], dtype=int)
    return np.arange(0, n_samples - win + 1, step)


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); 1.4826 makes MAD a consistent std estimate."""
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - med) / mad


def channel_metrics(
    x: np.ndarray, fs: float, line_freq_hz: float = 50.0, hf_band_hz: tuple[float, float] = (30.0, 45.0)
) -> dict[str, np.ndarray]:
    """Raw quality metrics for one window `x` (channels x samples). Each value: shape (channels,)."""
    # Linear detrend per window first: a slow drift/offset inside the window is not an artifact,
    # and without this the amplitude check depends on how much <0.5 Hz drift the filter passes.
    xd = detrend(x, axis=1, type="linear")
    std = xd.std(axis=1)
    max_abs = np.abs(xd).max(axis=1)
    logvar = np.log(std**2 + 1e-12)

    # Welch PSD with 1-s segments -> 1 Hz resolution, enough for band ratios.
    freqs, psd = welch(x, fs=fs, nperseg=min(int(fs), x.shape[1]), axis=1)
    lo, hi = hf_band_hz
    total = psd[:, (freqs >= 1) & (freqs <= hi)].sum(axis=1) + 1e-20
    hf = psd[:, (freqs >= lo) & (freqs <= hi)].sum(axis=1)
    hf_ratio = hf / total

    # Line-noise peak vs. flanking bins (only meaningful if the line frequency is below Nyquist)
    if line_freq_hz < fs / 2 - 3:
        peak = psd[:, np.abs(freqs - line_freq_hz) <= 1].mean(axis=1)
        flank_mask = (np.abs(freqs - line_freq_hz) >= 2) & (np.abs(freqs - line_freq_hz) <= 5)
        flank = psd[:, flank_mask].mean(axis=1) + 1e-20
        line_ratio = peak / flank
    else:
        line_ratio = np.ones(x.shape[0])

    return {
        "std": std,
        "max_abs": max_abs,
        "var_z": robust_z(logvar),
        "hf_ratio": hf_ratio,
        "line_ratio": line_ratio,
    }


def assess_window(x: np.ndarray, fs: float, cfg: QualityConfig, ch_names: list[str] | None = None) -> dict:
    """Run all checks on one window. Returns metrics, per-check bad masks and the window verdict.

    `ch_names` defaults to the 14 STEW/Emotiv channels (needed for the blink tolerance).
    """
    if ch_names is None:
        from eegengine.io.stew import CHANNELS
        ch_names = CHANNELS if x.shape[0] == len(CHANNELS) else [f"ch{i}" for i in range(x.shape[0])]
    m = channel_metrics(x, fs, cfg.line_freq_hz, cfg.hf_band_hz)
    bad = {
        "flat": m["std"] < cfg.flat_std_uv,
        "amplitude": m["max_abs"] > cfg.max_abs_uv,
        "variance": np.abs(m["var_z"]) > cfg.var_z_thresh,
        "hf": m["hf_ratio"] > cfg.hf_ratio_thresh,
        "line": m["line_ratio"] > cfg.line_ratio_thresh,
    }
    any_bad = np.logical_or.reduce([bad[c] for c in CHECKS])
    is_frontal = np.isin(ch_names, cfg.blink_channels)
    only_amp_var = ~(bad["flat"] | bad["hf"] | bad["line"])
    blink = any_bad & is_frontal & only_amp_var & (m["max_abs"] < cfg.blink_max_uv)
    n_bad = int(any_bad.sum())
    n_hard = int((any_bad & ~blink).sum())
    n_ch = x.shape[0]
    if n_bad == 0:
        flag = "good"
    elif n_hard > cfg.max_bad_channels_warn:
        flag = "bad"
    else:
        flag = "warn"
    return {
        "metrics": m,
        "bad": bad,
        "channel_bad": any_bad,
        "blink": blink,
        "n_bad": n_bad,
        "n_hard": n_hard,
        "score": 1.0 - n_bad / n_ch,
        "flag": flag,
    }


def assess_recording(data: np.ndarray, fs: float, ch_names: list[str], cfg: QualityConfig) -> tuple[list[dict], np.ndarray]:
    """Assess every window of a DC-free recording.

    Returns
      rows       : one dict per window (ready for a DataFrame)
      chan_flags : bool array (n_windows, n_checks, n_channels) for per-channel summaries
    """
    win = int(round(cfg.window_s * fs))
    starts = window_starts(data.shape[1], fs, cfg.window_s, cfg.step_s)
    rows, chan_flags = [], np.zeros((len(starts), len(CHECKS), data.shape[0]), dtype=bool)
    for i, s in enumerate(starts):
        res = assess_window(data[:, s:s + win], fs, cfg, ch_names)
        for j, c in enumerate(CHECKS):
            chan_flags[i, j] = res["bad"][c]
        rows.append({
            "window_index": i,
            "t_start": s / fs,
            "t_end": (s + win) / fs,
            "n_bad_channels": res["n_bad"],
            "n_hard_bad_channels": res["n_hard"],
            "n_blink_channels": int(res["blink"].sum()),
            "bad_channels": ";".join(np.asarray(ch_names)[res["channel_bad"]]),
            **{f"n_{c}": int(res["bad"][c].sum()) for c in CHECKS},
            "max_abs_uv": float(res["metrics"]["max_abs"].max()),
            "score": res["score"],
            "flag": res["flag"],
        })
    return rows, chan_flags
