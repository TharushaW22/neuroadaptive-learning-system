"""Gatekeeping for real-data workflows: refuse when real labelled data is missing/invalid."""
from __future__ import annotations

import pandas as pd

from src.data.dataset_audit import STATUS_READY, run_audit
from src.utils.config import resolve_path


class RealDataUnavailableError(RuntimeError):
    """Raised when a research-mode command is run without valid real labelled data."""


def get_clip_manifest(cfg: dict, probe: bool = True) -> pd.DataFrame:
    """Return included clips (real files + real labels). Raises if dataset not ready."""
    report = run_audit(cfg, probe=probe, write=True)
    if report["status"] != STATUS_READY:
        raise RealDataUnavailableError(
            f"Dataset status is {report['status']}. " + " ".join(report.get("messages", [])) +
            " Run: python scripts/audit_dataset.py  for details.")
    clips = pd.read_csv(resolve_path(cfg, "audit_dir") / "audit_clips.csv",
                        dtype={"clip_id": str, "participant_id": str})
    clips = clips[clips["included"].astype(str).str.lower() == "true"].copy()
    if clips.empty:
        raise RealDataUnavailableError("No included clips after audit.")
    return clips.reset_index(drop=True)


def assert_no_split_leakage(df: pd.DataFrame, split_col: str = "split", group_cols=("participant_id", "video_id")) -> None:
    """Raise if any participant or video appears in more than one split."""
    for col in group_cols:
        if col not in df.columns:
            continue
        sub = df[df[col].astype(str) != "unknown"]
        n = sub.groupby(col)[split_col].nunique()
        bad = n[n > 1]
        if len(bad):
            raise ValueError(f"Leakage: {len(bad)} {col} value(s) occur in multiple splits, e.g. {list(bad.index[:5])}")
