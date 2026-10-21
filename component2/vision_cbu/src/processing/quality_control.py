"""Per-frame and per-window quality assessment."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd


def is_valid_frame(frame) -> bool:
    return (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
            and frame.shape[0] >= 32 and frame.shape[1] >= 32 and frame.dtype == np.uint8)


def frame_brightness(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def assess_frame_quality(frame, result, vcfg: dict) -> dict:
    """Quality flags for one frame. `result` is a LandmarkResult (or None)."""
    q = {"face_detected": False, "num_faces": 0, "multiple_faces": False, "brightness": math.nan,
         "low_light": False, "over_exposed": False, "face_too_small": False,
         "landmarks_complete": False, "quality_ok": False}
    if not is_valid_frame(frame):
        return q
    b = frame_brightness(frame)
    q["brightness"] = b
    q["low_light"] = b < vcfg.get("min_brightness", 40)
    q["over_exposed"] = b > vcfg.get("max_brightness", 220)
    if result is not None:
        q["num_faces"] = int(result.num_faces)
        q["face_detected"] = result.num_faces > 0
        q["multiple_faces"] = result.num_faces > 1
        q["landmarks_complete"] = result.landmarks is not None and result.landmarks.shape[0] >= 468
        if result.bbox is not None:
            q["face_too_small"] = (result.bbox[3] - result.bbox[1]) < vcfg.get("min_face_scale", 0.08)
    q["quality_ok"] = bool(q["face_detected"] and q["landmarks_complete"] and not q["multiple_faces"]
                           and not q["low_light"] and not q["over_exposed"] and not q["face_too_small"])
    return q


def quality_score(rows: pd.DataFrame) -> float:
    """Fraction of quality-OK frames; 0.0 for empty input."""
    if rows is None or len(rows) == 0:
        return 0.0
    return float(rows["quality_ok"].astype(bool).mean())


def quality_warnings(row: dict) -> list[str]:
    msgs = []
    if not row.get("face_detected"):
        msgs.append("No face detected")
    if row.get("multiple_faces"):
        msgs.append("Multiple faces detected - only the largest face is analysed")
    if row.get("low_light"):
        msgs.append("Poor lighting (too dark)")
    if row.get("over_exposed"):
        msgs.append("Over-exposed image")
    if row.get("face_too_small"):
        msgs.append("Face too small / too far from camera")
    if row.get("face_detected") and not row.get("landmarks_complete"):
        msgs.append("Landmarks unavailable (face-box-only backend) - most features are missing")
    return msgs
