"""Versioned feature schema shared by extraction, training and inference.

Every feature row (webcam, dataset video or synthetic) has the same columns:
  META      - identity, provenance and timing
  LABELS    - original DAiSEE labels (NaN when unlabelled, e.g. webcam)
  FEATURES  - numerical visual features (NaN when not measurable)
  QUALITY   - per-frame quality flags

Changing any column list REQUIRES bumping SCHEMA_VERSION; saved models record the
version and the exact feature list, and loading refuses mismatches.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"

VALID_SOURCES = ("webcam", "daisee", "synthetic", "video_file")

META_COLUMNS = [
    "schema_version",
    "participant_id",
    "video_id",
    "frame_index",
    "timestamp_s",
    "source",
    "is_synthetic",
    "split",
    "label_policy",
]

LABEL_COLUMNS = ["label_engagement", "label_boredom", "label_confusion", "label_frustration"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "eye": ["ear_left", "ear_right", "ear_mean", "blink_event", "blink_rate_per_min"],
    "gaze": ["gaze_h", "gaze_v"],
    "head_pose": ["head_yaw", "head_pitch", "head_roll"],
    "facial": ["mouth_aspect_ratio", "brow_raise", "face_scale", "face_center_x", "face_center_y"],
}

FEATURE_COLUMNS: list[str] = [c for group in FEATURE_GROUPS.values() for c in group]

QUALITY_COLUMNS = [
    "face_detected",
    "num_faces",
    "multiple_faces",
    "brightness",
    "low_light",
    "over_exposed",
    "face_too_small",
    "landmarks_complete",
    "quality_ok",
]

ALL_COLUMNS = META_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS + QUALITY_COLUMNS

LABEL_TARGETS = {"engagement": "label_engagement", "boredom": "label_boredom",
                 "confusion": "label_confusion", "frustration": "label_frustration"}


class SchemaError(ValueError):
    """Raised when a feature table does not conform to the schema."""


def schema_fingerprint(feature_columns: Iterable[str] = FEATURE_COLUMNS) -> str:
    payload = json.dumps({"version": SCHEMA_VERSION, "features": list(feature_columns)})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def schema_description() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta_columns": META_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "feature_columns": FEATURE_COLUMNS,
        "quality_columns": QUALITY_COLUMNS,
        "fingerprint": schema_fingerprint(),
    }


def empty_row() -> dict:
    """A row with every column present; numeric values NaN, flags False."""
    row: dict = {c: math.nan for c in ALL_COLUMNS}
    row.update(
        schema_version=SCHEMA_VERSION, participant_id="unknown", video_id="unknown",
        frame_index=-1, source="webcam", is_synthetic=False, split="none",
        label_policy="unlabelled", face_detected=False, num_faces=0, multiple_faces=False,
        low_light=False, over_exposed=False, face_too_small=False,
        landmarks_complete=False, quality_ok=False,
    )
    return row


def validate_feature_frame(df: pd.DataFrame, allow_synthetic: bool = True,
                           require_labels: bool = False, label_target: str = "engagement") -> pd.DataFrame:
    """Validate and coerce a feature table. Returns a cleaned copy; raises SchemaError."""
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing schema columns: {missing}")
    out = df[ALL_COLUMNS].copy()
    versions = set(out["schema_version"].astype(str).unique())
    if versions - {SCHEMA_VERSION}:
        raise SchemaError(f"Schema version mismatch: found {sorted(versions)}, expected {SCHEMA_VERSION}")
    bad_src = set(out["source"].astype(str).unique()) - set(VALID_SOURCES)
    if bad_src:
        raise SchemaError(f"Unknown source values: {sorted(bad_src)}")

    out["is_synthetic"] = out["is_synthetic"].astype(str).str.lower().isin(["true", "1"])
    if (out["source"] == "synthetic").any() and not out.loc[out["source"] == "synthetic", "is_synthetic"].all():
        raise SchemaError("Rows with source='synthetic' must have is_synthetic=True")
    if not allow_synthetic and out["is_synthetic"].any():
        raise SchemaError("Synthetic rows are not permitted in real research data")

    for c in FEATURE_COLUMNS + LABEL_COLUMNS + ["timestamp_s", "brightness", "num_faces"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["face_detected", "multiple_faces", "low_light", "over_exposed",
              "face_too_small", "landmarks_complete", "quality_ok"]:
        out[c] = out[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce").fillna(-1).astype(int)
    out["participant_id"] = out["participant_id"].astype(str)
    out["video_id"] = out["video_id"].astype(str)

    if out["timestamp_s"].isna().any() or np.isinf(out["timestamp_s"]).any():
        raise SchemaError("Invalid (NaN/inf) timestamps present")
    if (out["timestamp_s"] < 0).any():
        raise SchemaError("Negative timestamps present")

    if require_labels:
        col = LABEL_TARGETS[label_target]
        if out[col].isna().any():
            raise SchemaError(f"Missing labels in {col}")
        vals = set(out[col].unique())
        if not vals <= {0.0, 1.0, 2.0, 3.0}:
            raise SchemaError(f"Labels outside 0-3 in {col}: {sorted(vals)}")
    return out
