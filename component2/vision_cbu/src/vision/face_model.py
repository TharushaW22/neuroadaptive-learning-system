"""3D reference face geometry for head pose and mock landmarks.

Uses MediaPipe's canonical face model (Apache License 2.0), stored in
data/canonical_face_vertices.csv. Its 468 vertices are numbered exactly like the Face Mesh
landmarks, so every detected landmark has a 3D counterpart. Units are approximately cm;
the file's frame is right-handed with y up and z towards the viewer.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "data" / "canonical_face_vertices.csv"
N_VERTICES = 468
NOSE_TIP = 1


@lru_cache(maxsize=1)
def canonical_vertices() -> np.ndarray:
    """(468, 3) vertices in the file's own frame."""
    arr = np.loadtxt(DATA_FILE, delimiter=",", comments="#", skiprows=4, dtype=np.float64)
    if arr.shape != (N_VERTICES, 4) or not np.array_equal(arr[:, 0], np.arange(N_VERTICES)):
        raise RuntimeError(f"Corrupt canonical face model file: {DATA_FILE}")
    return arr[:, 1:4]


def camera_frame(indices=None) -> np.ndarray:
    """Model points in OpenCV camera convention (x right, y down, z away from camera),
    centred on the nose tip. indices=None -> all 468 vertices."""
    v = canonical_vertices()
    pts = v - v[NOSE_TIP]
    pts = pts * np.array([1.0, -1.0, -1.0])
    return pts if indices is None else pts[list(indices)]


def rotation_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch): the composition that cv2.RQDecomp3x3 inverts exactly
    (angles about the camera x, y and z axes)."""
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx
