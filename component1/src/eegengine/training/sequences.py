"""Sequences of consecutive epochs from the SAME recording (input unit of the temporal models).

With 4 s epochs and a 2 s step, a sequence of 8 epochs covers 18 s of EEG. Sequences never
cross recordings, so a sequence has exactly one condition and one label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(meta: pd.DataFrame, seq_len: int = 8, stride: int = 2) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (idx, seq_meta). idx: (n_seq, seq_len) row indices into `meta` (and the epoch array).

    seq_meta has one row per sequence: subject_id, condition, rating, the label columns of the
    first epoch (identical for all epochs of a recording), and t_start / t_end of the sequence.
    """
    idx_list, rows = [], []
    order = meta.sort_values(["subject_id", "condition", "epoch_index"]).index.to_numpy()
    sorted_meta = meta.loc[order]
    label_cols = [c for c in meta.columns if c.startswith("y_")]
    for (s, c), g in sorted_meta.groupby(["subject_id", "condition"], sort=False):
        rows_idx = g.index.to_numpy()
        for start in range(0, len(rows_idx) - seq_len + 1, stride):
            seq = rows_idx[start:start + seq_len]
            idx_list.append(seq)
            first, last = meta.loc[seq[0]], meta.loc[seq[-1]]
            rows.append({"subject_id": s, "condition": c, "rating": first["rating"],
                         **{k: first[k] for k in label_cols},
                         "t_start": first["t_start"], "t_end": last["t_end"],
                         "frac_bad": float((meta.loc[seq, "quality_flag"] == "bad").mean())
                         if "quality_flag" in meta else 0.0})
    return np.array(idx_list, dtype=np.int64), pd.DataFrame(rows)


def sequence_scores_from_epochs(epoch_proba: np.ndarray, seq_idx: np.ndarray) -> np.ndarray:
    """Average per-epoch probabilities over each sequence (makes epoch models comparable)."""
    return epoch_proba[seq_idx].mean(axis=1)


def recording_groups(meta, rows: np.ndarray) -> list[np.ndarray]:
    """Epoch rows of each recording, in time order."""
    sub = meta.iloc[rows]
    out = []
    for _, g in sub.groupby(["subject_id", "condition"]):
        out.append(g.sort_values("epoch_index").index.to_numpy())
    return out


def sample_items(groups: list[np.ndarray], L: int, n_per_rec: int, rng) -> np.ndarray:
    """Random sequence starts per recording -> (n_items, L) epoch indices."""
    items = []
    for g in groups:
        starts = rng.integers(0, len(g) - L + 1, size=n_per_rec)
        items += [g[s:s + L] for s in starts]
    return np.array(items)


def strided_items(groups: list[np.ndarray], L: int, stride: int) -> np.ndarray:
    return np.array([g[s:s + L] for g in groups for s in range(0, len(g) - L + 1, stride)])
