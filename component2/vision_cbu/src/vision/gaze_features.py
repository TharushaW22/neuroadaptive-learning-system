"""Uncalibrated gaze-direction PROXIES from iris position inside the eye opening.

gaze_h: -1 (iris at image-left eye corner) .. +1 (image-right corner); 0 ~ centred.
gaze_v: -1 (towards upper lid) .. +1 (towards lower lid).

Limitations (see docs/limitations_and_ethics.md): no per-user calibration, no screen
geometry, sensitive to head pose and eyelid shape. These are relative signals, NOT
validated gaze estimates, and they do not identify where on a screen a person looks.
Requires iris landmarks (478-point mesh); returns NaN otherwise.
"""
from __future__ import annotations

import math

import numpy as np

from src.vision.landmark_indices import (LEFT_EYE_EAR, LEFT_IRIS_CENTER, N_MESH_WITH_IRIS,
                                         RIGHT_EYE_EAR, RIGHT_IRIS_CENTER)


def _eye_ratio(lm: np.ndarray, eye_idx: list[int], iris_idx: int, scale: np.ndarray) -> tuple[float, float]:
    p = lm[eye_idx, :2] * scale
    iris = lm[iris_idx, :2] * scale
    xs = sorted([p[0][0], p[3][0]])
    width = xs[1] - xs[0]
    upper = (p[1][1] + p[2][1]) / 2.0
    lower = (p[4][1] + p[5][1]) / 2.0
    gap = lower - upper
    if width < 1e-6:
        return math.nan, math.nan
    h = (iris[0] - (xs[0] + xs[1]) / 2.0) / (width / 2.0)
    v = (iris[1] - (upper + lower) / 2.0) / (gap / 2.0) if gap > 1e-3 else math.nan
    return float(np.clip(h, -1.5, 1.5)), (float(np.clip(v, -1.5, 1.5)) if not math.isnan(v) else math.nan)


def compute_gaze_proxy(landmarks: np.ndarray | None, width: int, height: int) -> tuple[float, float]:
    if landmarks is None or landmarks.shape[0] < N_MESH_WITH_IRIS:
        return math.nan, math.nan
    scale = np.array([width, height], dtype=np.float32)
    rh, rv = _eye_ratio(landmarks, RIGHT_EYE_EAR, RIGHT_IRIS_CENTER, scale)
    lh, lv = _eye_ratio(landmarks, LEFT_EYE_EAR, LEFT_IRIS_CENTER, scale)
    hs = [x for x in (rh, lh) if not math.isnan(x)]
    vs = [x for x in (rv, lv) if not math.isnan(x)]
    return (float(np.mean(hs)) if hs else math.nan, float(np.mean(vs)) if vs else math.nan)
