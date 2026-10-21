"""Eye Aspect Ratio (Soukupova & Cech, 2016) and blink-event / blink-rate estimation.

EAR thresholds vary between people, cameras and head poses; the default threshold is a
starting point and should be calibrated. A single blink (or its absence) says nothing
about attention on its own.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.vision.landmark_indices import LEFT_EYE_EAR, RIGHT_EYE_EAR


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """pts: (6,2) pixel coords ordered p1..p6. EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = pts
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return math.nan
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


def compute_ear(landmarks: np.ndarray | None, width: int, height: int) -> tuple[float, float, float]:
    """Return (ear_left, ear_right, ear_mean) for the subject's left/right eye; NaN if unavailable."""
    if landmarks is None or landmarks.shape[0] < 468:
        return math.nan, math.nan, math.nan
    scale = np.array([width, height], dtype=np.float32)
    right = eye_aspect_ratio(landmarks[RIGHT_EYE_EAR, :2] * scale)
    left = eye_aspect_ratio(landmarks[LEFT_EYE_EAR, :2] * scale)
    mean = float(np.nanmean([left, right])) if not (math.isnan(left) and math.isnan(right)) else math.nan
    return left, right, mean


class BlinkDetector:
    """Counts a blink when EAR stays below threshold for [min_frames, max_frames] samples
    and then re-opens. Missing EAR (NaN) aborts any closure in progress."""

    def __init__(self, threshold: float = 0.21, min_frames: int = 2, max_frames: int = 15,
                 window_s: float = 60.0, min_elapsed_s: float = 10.0):
        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.window_s = window_s
        self.min_elapsed_s = min_elapsed_s
        self.reset()

    def reset(self) -> None:
        self._closed = 0
        self._blinks: deque[float] = deque()
        self._t0: float | None = None
        self.total_blinks = 0

    def update(self, ear: float, timestamp_s: float) -> bool:
        if self._t0 is None:
            self._t0 = timestamp_s
        if ear is None or math.isnan(ear):
            self._closed = 0
            return False
        if ear < self.threshold:
            self._closed += 1
            return False
        blink = self.min_frames <= self._closed <= self.max_frames
        self._closed = 0
        if blink:
            self._blinks.append(timestamp_s)
            self.total_blinks += 1
        return blink

    def blink_rate(self, timestamp_s: float) -> float:
        """Blinks/min over the trailing window; NaN until min_elapsed_s of data exist."""
        if self._t0 is None:
            return math.nan
        while self._blinks and self._blinks[0] < timestamp_s - self.window_s:
            self._blinks.popleft()
        elapsed = min(self.window_s, timestamp_s - self._t0)
        if elapsed < self.min_elapsed_s:
            return math.nan
        return 60.0 * len(self._blinks) / elapsed
