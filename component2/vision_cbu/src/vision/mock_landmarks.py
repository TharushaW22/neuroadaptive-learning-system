"""Deterministic mock landmarks/detector for tests and demo mode (no camera, no model file).

The mock face is a perspective projection of all 468 vertices of MediaPipe's canonical face
model (face_model.py) through the same pinhole camera head_pose.py assumes, so known head
rotations can be applied and recovered. Eye openness, iris position and mouth opening are
then set explicitly. This is a software-testing aid, not a model of any person's face.
"""
from __future__ import annotations

import numpy as np

from src.vision import landmark_indices as L
from src.vision.face_landmarks import LandmarkResult, _bbox_from_landmarks
from src.vision.face_model import N_VERTICES, camera_frame, rotation_matrix
from src.vision.head_pose import camera_matrix

_MODEL = camera_frame()                               # (468, 3) cm, nose at origin
_FACE_HEIGHT_CM = float(_MODEL[L.CHIN, 1] - _MODEL[L.FOREHEAD, 1])


def _perp(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit 'up' normal of segment a->b in image coordinates, and its length."""
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / max(length, 1e-9)
    return np.array([u[1], -u[0]]), length


def make_mock_landmarks(width: int = 640, height: int = 480, eye_open: float = 1.0,
                        gaze_shift: float = 0.0, mouth_open: float = 0.1,
                        center=(320.0, 240.0), scale: float = 0.4, n_points: int = 478,
                        yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Return (n_points, 3) normalised landmarks (n_points 468 or 478).

    eye_open 1.0 -> EAR 0.30 (0 -> closed); gaze_shift in [-1, 1] moves the irises by
    0.4 * gaze_shift of the half eye width; mouth_open = mouth aspect ratio; yaw/pitch/roll
    in degrees (head_pose.py convention); scale 0.4 -> forehead-chin height ~270 px.
    """
    K = camera_matrix(width, height)
    f = K[0, 0]
    tz = f * _FACE_HEIGHT_CM / (270.0 * scale / 0.4)
    pts3 = _MODEL @ rotation_matrix(-yaw, pitch, roll).T
    pts3 = pts3 + np.array([(center[0] - K[0, 2]) * tz / f, (center[1] - K[1, 2]) * tz / f, tz])
    px = (pts3[:, :2] / pts3[:, 2:3]) * f + K[:2, 2]           # (468, 2) pixels

    out = np.zeros((max(n_points, N_VERTICES), 3), dtype=np.float64)
    out[:N_VERTICES, :2] = px

    for (p1, p2, p3, p4, p5, p6) in (L.RIGHT_EYE_EAR, L.LEFT_EYE_EAR):
        a, b = px[p1], px[p4]
        n, w_ = _perp(a, b)
        half_gap = 0.15 * w_ * eye_open                      # EAR = 2*half_gap / w_ = 0.30 * eye_open
        for idx, frac, sgn in ((p2, 1 / 3, 1), (p3, 2 / 3, 1), (p6, 1 / 3, -1), (p5, 2 / 3, -1)):
            out[idx, :2] = a + frac * (b - a) + sgn * half_gap * n
    if n_points >= 478:
        for eye, iris in ((L.RIGHT_EYE_EAR, L.RIGHT_IRIS_CENTER), (L.LEFT_EYE_EAR, L.LEFT_IRIS_CENTER)):
            a, b = px[eye[0]], px[eye[3]]
            lo, hi = (a, b) if a[0] <= b[0] else (b, a)
            out[iris, :2] = (a + b) / 2 + gaze_shift * 0.4 * (hi - lo) / 2
            out[iris + 1:iris + 5, :2] = out[iris, :2]       # iris contour points
    ml, mr = px[L.MOUTH_LEFT_INNER], px[L.MOUTH_RIGHT_INNER]
    nm, mw = _perp(ml, mr)
    out[L.MOUTH_UPPER_INNER, :2] = (ml + mr) / 2 + nm * mw * mouth_open / 2
    out[L.MOUTH_LOWER_INNER, :2] = (ml + mr) / 2 - nm * mw * mouth_open / 2

    out = out[:n_points]
    out[:, 0] /= width
    out[:, 1] /= height
    return out.astype(np.float32)


class MockLandmarkDetector:
    """Returns mock landmarks. `script` is an optional list of dicts with keys
    num_faces / eye_open / gaze_shift / yaw / pitch / roll, consumed one per call (cycled)."""

    backend = "mock"

    def __init__(self, script: list[dict] | None = None):
        self.script = script or [{}]
        self.calls = 0
        self.closed = False

    def detect(self, frame_bgr) -> LandmarkResult:
        step = self.script[self.calls % len(self.script)]
        self.calls += 1
        n = int(step.get("num_faces", 1))
        if n == 0:
            return LandmarkResult(None, 0, None, self.backend)
        h, w = frame_bgr.shape[:2]
        lm = make_mock_landmarks(w, h, eye_open=step.get("eye_open", 1.0), gaze_shift=step.get("gaze_shift", 0.0),
                                 yaw=step.get("yaw", 0.0), pitch=step.get("pitch", 0.0), roll=step.get("roll", 0.0))
        return LandmarkResult(lm, n, _bbox_from_landmarks(lm), self.backend)

    def close(self) -> None:
        self.closed = True
