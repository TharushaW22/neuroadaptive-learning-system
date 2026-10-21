"""Facial geometry and simple facial-behaviour PROXIES.

These are geometric ratios (mouth opening, brow-to-eye distance, face size/position).
They are NOT FACS action units and are not validated expression measures.
"""
from __future__ import annotations

import math

import numpy as np

from src.vision import landmark_indices as L


def compute_facial_features(landmarks: np.ndarray | None, bbox, width: int, height: int) -> dict:
    out = {"mouth_aspect_ratio": math.nan, "brow_raise": math.nan, "face_scale": math.nan,
           "face_center_x": math.nan, "face_center_y": math.nan}
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        out["face_scale"] = float(y1 - y0)
        out["face_center_x"] = float((x0 + x1) / 2)
        out["face_center_y"] = float((y0 + y1) / 2)
    if landmarks is None or landmarks.shape[0] < 468:
        return out
    px = landmarks[:, :2] * np.array([width, height], dtype=np.float32)
    mouth_w = np.linalg.norm(px[L.MOUTH_LEFT_INNER] - px[L.MOUTH_RIGHT_INNER])
    if mouth_w > 1e-6:
        out["mouth_aspect_ratio"] = float(np.linalg.norm(px[L.MOUTH_UPPER_INNER] - px[L.MOUTH_LOWER_INNER]) / mouth_w)
    face_h = np.linalg.norm(px[L.FOREHEAD] - px[L.CHIN])
    if face_h > 1e-6:
        r_eye_y = px[L.RIGHT_EYE_EAR][:, 1].mean()
        l_eye_y = px[L.LEFT_EYE_EAR][:, 1].mean()
        out["brow_raise"] = float(((r_eye_y - px[L.RIGHT_BROW, 1]) + (l_eye_y - px[L.LEFT_BROW, 1])) / 2 / face_h)
    out["face_scale"] = float(face_h / height)
    return out
