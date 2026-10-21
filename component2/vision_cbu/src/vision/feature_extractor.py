"""Per-frame feature extraction producing schema-conformant rows."""
from __future__ import annotations

import math

import numpy as np

from src.data.feature_schema import empty_row
from src.processing.quality_control import assess_frame_quality, is_valid_frame
from src.vision.eye_features import BlinkDetector, compute_ear
from src.vision.facial_features import compute_facial_features
from src.vision.face_landmarks import FaceLandmarkDetector, LandmarkResult
from src.vision.gaze_features import compute_gaze_proxy
from src.vision.head_pose import estimate_head_pose


class FrameFeatureExtractor:
    """Stateful (blink tracking) extractor for ONE stream (one webcam session or one video).

    Call reset() between videos. The detector can be injected (tests/mock mode).
    """

    def __init__(self, cfg: dict, detector=None, source: str = "webcam",
                 participant_id: str = "unknown", video_id: str = "unknown", split: str = "none"):
        self.cfg = cfg
        v = cfg["vision"]
        self.vcfg = v
        self.detector = detector if detector is not None else FaceLandmarkDetector(cfg)
        self.blinks = BlinkDetector(v["ear_blink_threshold"], v["blink_min_consecutive_frames"],
                                    v.get("blink_max_consecutive_frames", 15), v["blink_rate_window_s"])
        self.source, self.participant_id, self.video_id, self.split = source, participant_id, video_id, split
        self._last_ts: float | None = None

    @property
    def backend(self) -> str:
        return getattr(self.detector, "backend", "unknown")

    def reset(self, participant_id: str | None = None, video_id: str | None = None, split: str | None = None):
        self.blinks.reset()
        self._last_ts = None
        if participant_id is not None:
            self.participant_id = participant_id
        if video_id is not None:
            self.video_id = video_id
        if split is not None:
            self.split = split

    def extract(self, frame, timestamp_s: float, frame_index: int = -1,
                labels: dict | None = None) -> tuple[dict, LandmarkResult | None]:
        if timestamp_s is None or not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError(f"Invalid timestamp: {timestamp_s!r}")
        if self._last_ts is not None and timestamp_s < self._last_ts:
            raise ValueError(f"Non-monotonic timestamp {timestamp_s} < {self._last_ts}")
        self._last_ts = timestamp_s

        row = empty_row()
        row.update(participant_id=self.participant_id, video_id=self.video_id, frame_index=int(frame_index),
                   timestamp_s=float(timestamp_s), source=self.source,
                   is_synthetic=self.source == "synthetic", split=self.split)
        if labels:
            row["label_policy"] = self.cfg["labels"].get("video_label_policy", "broadcast_video_label")
            for k, val in labels.items():
                row[f"label_{k}"] = val

        if not is_valid_frame(frame):
            self.blinks.update(math.nan, timestamp_s)
            row.update(assess_frame_quality(frame, None, self.vcfg))
            return row, None

        h, w = frame.shape[:2]
        try:
            result = self.detector.detect(frame)
        except Exception:
            result = LandmarkResult(None, 0, None, self.backend)
        row.update(assess_frame_quality(frame, result, self.vcfg))

        lm = result.landmarks if result.face_detected else None
        el, er, em = compute_ear(lm, w, h)
        row.update(ear_left=el, ear_right=er, ear_mean=em)
        blink = self.blinks.update(em, timestamp_s)
        row["blink_event"] = float(blink) if not math.isnan(em) else math.nan
        row["blink_rate_per_min"] = self.blinks.blink_rate(timestamp_s)
        row["gaze_h"], row["gaze_v"] = compute_gaze_proxy(lm, w, h)
        row["head_yaw"], row["head_pitch"], row["head_roll"] = estimate_head_pose(lm, w, h)
        row.update(compute_facial_features(lm, result.bbox, w, h))
        return row, result

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()


def numeric_summary(rows: list[dict], cols: list[str]) -> dict:
    arr = np.array([[r.get(c, math.nan) for c in cols] for r in rows], dtype=float)
    with np.errstate(all="ignore"):
        return {c: float(np.nanmean(arr[:, i])) if len(arr) else math.nan for i, c in enumerate(cols)}
