"""Missing-value handling and normalisation.

Rules that prevent leakage:
* Imputation within a video uses only that video's past values (bounded forward-fill).
* Remaining gaps are filled with statistics computed on TRAINING data only.
* The Normalizer is fit on training data only, then applied to val/test/live data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def forward_fill_within_video(df: pd.DataFrame, feature_cols: list[str], max_gap: int,
                              group_col: str = "video_id") -> pd.DataFrame:
    """Forward-fill gaps of at most max_gap samples, never across videos.
    Rows that failed quality control are treated as missing before filling."""
    out = df.sort_values([group_col, "timestamp_s"]).copy()
    if "quality_ok" in out.columns:
        bad = ~out["quality_ok"].astype(bool)
        out.loc[bad, feature_cols] = np.nan
    if max_gap and max_gap > 0:
        out[feature_cols] = out.groupby(group_col, sort=False)[feature_cols].ffill(limit=max_gap)
    return out


class Normalizer:
    """Feature-wise normaliser with NaN-aware fitting. method: standard | minmax | none."""

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "minmax", "none"):
            raise ValueError(f"Unknown normalisation: {method}")
        self.method = method
        self.fill_values: np.ndarray | None = None
        self.a: np.ndarray | None = None   # shift
        self.b: np.ndarray | None = None   # scale
        self.fitted = False

    def fit(self, X: np.ndarray) -> "Normalizer":
        """X: (n, F) or (n, T, F); NaNs ignored. Fit on TRAINING data only."""
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns are handled below
            med = np.nanmedian(flat, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            if self.method == "standard":
                a, b = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
            elif self.method == "minmax":
                a, b = np.nanmin(flat, axis=0), np.nanmax(flat, axis=0) - np.nanmin(flat, axis=0)
            else:
                a, b = np.zeros(flat.shape[1]), np.ones(flat.shape[1])
        a = np.where(np.isnan(a), 0.0, a)
        b = np.where(np.isnan(b) | (b < 1e-8), 1.0, b)
        self.fill_values, self.a, self.b, self.fitted = med, a, b, True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN with training medians, then scale. Output contains no NaN."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fit on training data first")
        X = X.astype(np.float64).copy()
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.broadcast_to(self.fill_values, X.shape)[nan]
        return ((X - self.a) / self.b).astype(np.float32)

    def to_dict(self) -> dict:
        return {"method": self.method, "fill_values": self.fill_values.tolist(),
                "a": self.a.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls(d["method"])
        n.fill_values, n.a, n.b = (np.array(d[k], dtype=np.float64) for k in ("fill_values", "a", "b"))
        n.fitted = True
        return n
