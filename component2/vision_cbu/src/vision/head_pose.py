"""Head pose (yaw, pitch, roll in degrees) via cv2.solvePnP.

3D reference: all 468 vertices of MediaPipe's canonical face model (face_model.py), matched
one-to-one with the detected Face Mesh landmarks. Using every point (rather than a handful)
makes the estimate robust to noise in individual landmarks; in a mirror-image check on a real
photo, yaw flipped sign and pitch agreed to within about 1 degree.

Camera: approximate pinhole (focal length = image width, principal point = image centre, no
distortion). Angles are approximate and not validated against motion-capture ground truth;
treat them as relative signals over time.

Sign convention (verified in tests by rotating the reference model by known angles):
  yaw   > 0  face turned towards the image's right side
  pitch > 0  head tilted down (nose moves down in the image)
  roll  > 0  head tilted clockwise as seen in the image
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.vision.face_model import N_VERTICES, camera_frame

_MODEL = camera_frame()     # (468, 3)


def camera_matrix(width: int, height: int) -> np.ndarray:
    return np.array([[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]], dtype=np.float64)


def estimate_head_pose(landmarks: np.ndarray | None, width: int, height: int) -> tuple[float, float, float]:
    if landmarks is None or landmarks.shape[0] < N_VERTICES:
        return math.nan, math.nan, math.nan
    img_pts = (landmarks[:N_VERTICES, :2].astype(np.float64) * np.array([width, height], dtype=np.float64))
    if not np.isfinite(img_pts).all():
        return math.nan, math.nan, math.nan
    try:
        ok, rvec, tvec = cv2.solvePnP(_MODEL, img_pts, camera_matrix(width, height), np.zeros((4, 1)),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return math.nan, math.nan, math.nan
    if not ok or tvec[2, 0] <= 0:          # face must be in front of the camera
        return math.nan, math.nan, math.nan
    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]      # rotation about camera x, y, z
    return _wrap(-yaw), _wrap(pitch), _wrap(roll)


def _wrap(a: float) -> float:
    """Wrap into (-90, 90] to remove the 180-degree ambiguity of the decomposition."""
    a = (float(a) + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a <= -90.0:
        a += 180.0
    return a
