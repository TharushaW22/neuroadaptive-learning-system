"""Temporal window construction.

Windows never span two videos, so windows from one source video always stay in that
video's split. Video-level labels are broadcast to every window of the video
("broadcast_video_label" policy) - they are NOT independently verified frame/window labels.

Outputs per window: X (T, F) features, mask (T,) of valid timesteps (1 = observed,
0 = padding), and identifiers for leakage checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SequenceSet:
    X: np.ndarray                 # (N, T, F) float32 (may contain NaN before normalisation)
    mask: np.ndarray              # (N, T) float32
    y: np.ndarray                 # (N,) int64, -1 if unlabelled
    video_ids: np.ndarray
    participant_ids: np.ndarray
    splits: np.ndarray
    start_ts: np.ndarray
    feature_columns: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, idx) -> "SequenceSet":
        return SequenceSet(self.X[idx], self.mask[idx], self.y[idx], self.video_ids[idx],
                           self.participant_ids[idx], self.splits[idx], self.start_ts[idx],
                           self.feature_columns, dict(self.meta))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, mask=self.mask, y=self.y, video_ids=self.video_ids,
                            participant_ids=self.participant_ids, splits=self.splits, start_ts=self.start_ts)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump({"feature_columns": self.feature_columns, **self.meta}, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "SequenceSet":
        d = np.load(path, allow_pickle=False)
        with open(path.with_suffix(".json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        cols = meta.pop("feature_columns")
        return cls(d["X"], d["mask"], d["y"], d["video_ids"], d["participant_ids"], d["splits"],
                   d["start_ts"], cols, meta)


def clean_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN/inf/negative or duplicate timestamps; sort ascending."""
    ts = pd.to_numeric(df["timestamp_s"], errors="coerce")
    ok = ts.notna() & np.isfinite(ts) & (ts >= 0)
    out = df.loc[ok].assign(timestamp_s=ts[ok]).sort_values("timestamp_s")
    out = out.drop_duplicates("timestamp_s", keep="first")
    return out, int(len(df) - len(out))


def windows_for_video(values: np.ndarray, quality_ok: np.ndarray, window: int, stride: int,
                      min_valid_ratio: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return [(start, X(T,F), mask(T,))] for one video.

    Short videos yield one zero-padded window (mask marks padding) if they contain
    enough quality-OK samples. Windows with too few quality-OK samples are dropped.
    """
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    n, f = values.shape
    out = []
    if n == 0:
        return out
    starts = list(range(0, max(n - window, 0) + 1, stride)) if n >= window else [0]
    for s in starts:
        seg = values[s:s + window]
        q = quality_ok[s:s + window].astype(bool)
        X = np.zeros((window, f), dtype=np.float32)
        m = np.zeros(window, dtype=np.float32)
        X[:len(seg)] = seg
        m[:len(seg)] = 1.0
        if q.sum() / window < min_valid_ratio:
            continue
        out.append((s, X, m))
    return out


def build_sequences(df: pd.DataFrame, feature_columns: list[str], window: int, stride: int,
                    min_valid_ratio: float, label_column: str | None = None,
                    label_transform=None) -> tuple[SequenceSet, dict]:
    """Build windows for every video in df. Returns (SequenceSet, stats)."""
    Xs, Ms, ys, vids, pids, splits, starts = [], [], [], [], [], [], []
    stats = {"videos": 0, "videos_without_windows": 0, "invalid_timestamps_dropped": 0, "windows": 0}
    for vid, g in df.groupby("video_id", sort=True):
        g, dropped = clean_timestamps(g)
        stats["invalid_timestamps_dropped"] += dropped
        stats["videos"] += 1
        if label_column:
            labs = g[label_column].dropna().unique()
            if len(labs) != 1:
                stats["videos_without_windows"] += 1
                continue
            lab = label_transform(labs[0]) if label_transform else int(labs[0])
        else:
            lab = -1
        wins = windows_for_video(g[feature_columns].to_numpy(dtype=np.float32),
                                 g["quality_ok"].to_numpy(), window, stride, min_valid_ratio)
        if not wins:
            stats["videos_without_windows"] += 1
            continue
        ts = g["timestamp_s"].to_numpy()
        for s, X, m in wins:
            Xs.append(X); Ms.append(m); ys.append(lab)
            vids.append(str(vid)); pids.append(str(g["participant_id"].iloc[0]))
            splits.append(str(g["split"].iloc[0])); starts.append(float(ts[s]))
    stats["windows"] = len(Xs)
    F = len(feature_columns)
    seq = SequenceSet(
        X=np.stack(Xs) if Xs else np.zeros((0, window, F), np.float32),
        mask=np.stack(Ms) if Ms else np.zeros((0, window), np.float32),
        y=np.array(ys, dtype=np.int64), video_ids=np.array(vids, dtype=str),
        participant_ids=np.array(pids, dtype=str), splits=np.array(splits, dtype=str),
        start_ts=np.array(starts, dtype=np.float64), feature_columns=list(feature_columns),
        meta={"window": window, "stride": stride, "min_valid_ratio": min_valid_ratio,
              "label_column": label_column, "label_policy": "broadcast_video_label", **stats})
    return seq, stats


def validate_sequence_lengths(X: np.ndarray, expected_T: int, expected_F: int) -> None:
    if X.ndim != 3:
        raise ValueError(f"Expected 3-D sequences (N,T,F), got shape {X.shape}")
    if X.shape[1] != expected_T or X.shape[2] != expected_F:
        raise ValueError(f"Sequence shape {X.shape[1:]} != expected ({expected_T}, {expected_F})")
