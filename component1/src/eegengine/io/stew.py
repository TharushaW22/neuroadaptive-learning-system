"""STEW (Simultaneous Task EEG Workload) dataset loader and label builder.

Dataset facts (Lim et al., 2018):
  * 48 subjects, Emotiv EPOC, 14 channels, 128 Hz.
  * subXX_lo.txt = 2.5 min resting state, subXX_hi.txt = 2.5 min SIMKAP multitasking.
  * Each file: 19200 rows (samples) x 14 columns (channels), whitespace separated, no header.
  * ratings.txt: "subject, rest_rating, task_rating" on a 1-9 workload scale.
    Subjects 5, 24 and 42 have no rating line -> their ratings are NaN.

Labels built here (STEW only provides WORKLOAD):
  * T1 workload_binary : lo = 0, hi = 1                       (available for all 48 subjects)
  * T2 workload_level  : rating 1-3 -> 0 (low), 4-6 -> 1 (medium), 7-9 -> 2 (high)
                         missing rating -> -1 and level_mask = False (never drop the subject).

The raw values carry a large Emotiv DC offset (~4000 uV). The loader returns the values
unchanged (in microvolts); DC removal is done by `eegengine.preprocess.dc.remove_dc`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNELS: list[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
FS: int = 128                    # sampling rate (Hz)
N_SAMPLES: int = 19200           # 150 s * 128 Hz
N_SUBJECTS: int = 48
SUBJECTS: list[int] = list(range(1, N_SUBJECTS + 1))
CONDITIONS: tuple[str, str] = ("lo", "hi")  # lo = rest, hi = SIMKAP task
LEVEL_NAMES: list[str] = ["low", "medium", "high"]

# Project root = component1/ (this file is src/eegengine/io/stew.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STEW_DIR = PROJECT_ROOT / "data" / "raw" / "STEW"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    """One STEW recording (one subject, one condition)."""

    subject_id: int
    condition: str               # "lo" (rest) or "hi" (task)
    rating: float                # 1-9 self-reported workload, NaN if missing
    data: np.ndarray             # shape (14, 19200), microvolts, DC offset still present
    fs: int = FS
    ch_names: tuple[str, ...] = tuple(CHANNELS)
    path: str = ""

    @property
    def label_binary(self) -> int:
        """T1: 0 = rest (lo), 1 = task (hi)."""
        return 0 if self.condition == "lo" else 1

    @property
    def label_level(self) -> int:
        """T2: 0 low / 1 medium / 2 high, or -1 when the rating is missing."""
        return rating_to_level(self.rating)

    @property
    def level_mask(self) -> bool:
        """True when the T2 label is valid (rating present)."""
        return self.label_level >= 0

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs

    @property
    def key(self) -> str:
        return f"sub{self.subject_id:02d}_{self.condition}"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def rating_to_level(rating: float) -> int:
    """Map a 1-9 workload rating to 3 levels: 1-3 low(0), 4-6 medium(1), 7-9 high(2).

    Returns -1 for a missing (NaN) rating so it can be masked out of the loss/metrics.
    """
    if rating is None or np.isnan(rating):
        return -1
    r = int(round(rating))
    if not 1 <= r <= 9:
        raise ValueError(f"Rating {rating} outside the 1-9 scale")
    if r <= 3:
        return 0
    if r <= 6:
        return 1
    return 2


def load_ratings(stew_dir: Path | str = DEFAULT_STEW_DIR) -> dict[int, tuple[float, float]]:
    """Read ratings.txt -> {subject: (rest_rating, task_rating)}.

    Subjects without a line (5, 24, 42) get (NaN, NaN).
    """
    path = Path(stew_dir) / "ratings.txt"
    ratings: dict[int, tuple[float, float]] = {s: (np.nan, np.nan) for s in SUBJECTS}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected 'subject, rest, task', got {line!r}")
        subject, rest, task = int(parts[0]), float(parts[1]), float(parts[2])
        if subject not in ratings:
            raise ValueError(f"{path}:{line_no}: unknown subject {subject}")
        ratings[subject] = (rest, task)
    return ratings


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def recording_path(subject: int, condition: str, stew_dir: Path | str = DEFAULT_STEW_DIR) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    return Path(stew_dir) / f"sub{subject:02d}_{condition}.txt"


def read_signal_file(path: Path | str) -> np.ndarray:
    """Read one STEW text file and return a (channels, samples) float64 array."""
    # sep=r"\s+" uses pandas' fast C whitespace parser (much faster than np.loadtxt).
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    arr = df.to_numpy().T  # file is samples x channels -> transpose to channels x samples
    if arr.shape != (len(CHANNELS), N_SAMPLES):
        raise ValueError(f"{path}: expected shape (14, 19200) after transpose, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: contains NaN/Inf values")
    return arr


def load_recording(
    subject: int,
    condition: str,
    stew_dir: Path | str = DEFAULT_STEW_DIR,
    ratings: dict[int, tuple[float, float]] | None = None,
) -> Recording:
    ratings = ratings if ratings is not None else load_ratings(stew_dir)
    rest, task = ratings[subject]
    path = recording_path(subject, condition, stew_dir)
    return Recording(
        subject_id=subject,
        condition=condition,
        rating=rest if condition == "lo" else task,
        data=read_signal_file(path),
        path=str(path),
    )


def load_stew(
    stew_dir: Path | str = DEFAULT_STEW_DIR,
    subjects: Iterable[int] | None = None,
    conditions: Iterable[str] = CONDITIONS,
    cache_path: Path | str | None = None,
) -> list[Recording]:
    """Load STEW recordings (default: all 48 subjects x 2 conditions = 96).

    If `cache_path` is given, the full set of signals is cached to a .npz file so later
    loads take well under a second instead of parsing 96 text files.
    """
    subjects = list(subjects) if subjects is not None else SUBJECTS
    conditions = list(conditions)
    ratings = load_ratings(stew_dir)

    cache = _read_cache(cache_path) if cache_path else None
    recs: list[Recording] = []
    for s in subjects:
        for c in conditions:
            key = f"sub{s:02d}_{c}"
            if cache is not None and key in cache:
                rest, task = ratings[s]
                recs.append(Recording(s, c, rest if c == "lo" else task, cache[key],
                                      path=str(recording_path(s, c, stew_dir))))
            else:
                recs.append(load_recording(s, c, stew_dir, ratings))

    if cache_path and (cache is None or any(r.key not in cache for r in recs)):
        _write_cache(cache_path, recs, cache)
    return recs


def _read_cache(cache_path: Path | str) -> dict[str, np.ndarray] | None:
    p = Path(cache_path)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _write_cache(cache_path: Path | str, recs: list[Recording], old: dict | None) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(old or {})
    arrays.update({r.key: r.data for r in recs})
    np.savez(p, **arrays)


# ---------------------------------------------------------------------------
# Tidy views / conversions
# ---------------------------------------------------------------------------
def recordings_to_frame(recs: list[Recording]) -> pd.DataFrame:
    """One row per recording with all labels and masks (no signal arrays)."""
    return pd.DataFrame(
        {
            "subject_id": r.subject_id,
            "condition": r.condition,
            "rating": r.rating,
            "label_binary": r.label_binary,
            "label_level": r.label_level,
            "level_mask": r.level_mask,
            "n_channels": r.data.shape[0],
            "n_samples": r.data.shape[1],
            "fs": r.fs,
            "duration_s": r.duration_s,
            "path": r.path,
        }
        for r in recs
    )


def to_mne_raw(rec: Recording, remove_dc: bool = True, verbose: str = "ERROR"):
    """Convert a Recording to an mne.io.RawArray with the standard 10-20 montage.

    MNE expects volts, so microvolts are scaled by 1e-6. With `remove_dc=True`, the
    per-channel mean is subtracted first (the Emotiv ~4000 uV offset).
    """
    import mne  # imported lazily: MNE is slow to import and not needed for plain loading

    data = rec.data - rec.data.mean(axis=1, keepdims=True) if remove_dc else rec.data
    info = mne.create_info(list(rec.ch_names), sfreq=rec.fs, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=verbose)
    raw.set_montage(mne.channels.make_standard_montage(_montage_name()))
    return raw


def _montage_name() -> str:
    """Standard 10-20 montage. MNE >= 1.13 renamed 'standard_1020' to 'colin27_1020'."""
    import mne

    names = mne.channels.get_builtin_montages()
    return "colin27_1020" if "colin27_1020" in names else "standard_1020"
