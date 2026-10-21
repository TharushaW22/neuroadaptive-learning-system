"""Build the dataset-level quality report (CSV + JSON) from per-recording checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eegengine.io.stew import Recording
from eegengine.preprocess.dc import remove_dc
from eegengine.quality.checks import CHECKS, QualityConfig, assess_recording


def build_quality_report(recs: list[Recording], cfg: QualityConfig, highpass_hz: float = 0.5):
    """Return (windows_df, channels_df, summary_dict) for a list of recordings."""
    win_frames, chan_rows, rec_summaries = [], [], []
    for rec in recs:
        clean = remove_dc(rec.data, rec.fs, highpass_hz=highpass_hz)
        rows, flags = assess_recording(clean, rec.fs, list(rec.ch_names), cfg)
        df = pd.DataFrame(rows)
        df.insert(0, "condition", rec.condition)
        df.insert(0, "subject_id", rec.subject_id)
        win_frames.append(df)

        # Fraction of windows in which each channel failed each check
        for k, ch in enumerate(rec.ch_names):
            row = {"subject_id": rec.subject_id, "condition": rec.condition, "channel": ch}
            for j, c in enumerate(CHECKS):
                row[f"frac_{c}"] = float(flags[:, j, k].mean())
            row["frac_any"] = float(flags[:, :, k].any(axis=1).mean())
            chan_rows.append(row)

        counts = df["flag"].value_counts().to_dict()
        rec_summaries.append({
            "subject_id": rec.subject_id,
            "condition": rec.condition,
            "n_windows": len(df),
            "n_good": int(counts.get("good", 0)),
            "n_warn": int(counts.get("warn", 0)),
            "n_bad": int(counts.get("bad", 0)),
            "mean_score": float(df["score"].mean()),
        })

    windows = pd.concat(win_frames, ignore_index=True)
    channels = pd.DataFrame(chan_rows)
    flag_counts = windows["flag"].value_counts().to_dict()
    worst_channels = (channels.groupby("channel")["frac_any"].mean()
                      .sort_values(ascending=False).round(4).to_dict())
    summary = {
        "config": cfg.to_dict(),
        "dc_removal_highpass_hz": highpass_hz,
        "n_recordings": len(recs),
        "n_windows": int(len(windows)),
        "flag_counts": {k: int(flag_counts.get(k, 0)) for k in ("good", "warn", "bad")},
        "flag_fractions": {k: round(flag_counts.get(k, 0) / len(windows), 4) for k in ("good", "warn", "bad")},
        "windows_failing_check": {c: int((windows[f"n_{c}"] > 0).sum()) for c in CHECKS},
        "mean_channel_bad_fraction": worst_channels,
        "flag_fractions_by_condition": {
            cond: {k: round(float((g["flag"] == k).mean()), 4) for k in ("good", "warn", "bad")}
            for cond, g in windows.groupby("condition")
        },
        "recordings": rec_summaries,
    }
    return windows, channels, summary


def save_quality_report(windows: pd.DataFrame, channels: pd.DataFrame, summary: dict, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out / "quality_windows.csv", index=False)
    channels.to_csv(out / "quality_channels.csv", index=False)
    with open(out / "quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))
