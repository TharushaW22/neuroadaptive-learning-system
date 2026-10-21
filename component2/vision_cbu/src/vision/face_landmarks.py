"""Face detection + landmarks with graceful backend fallback.

Backends (tried in this order for `landmark_backend: auto`):
  1. mediapipe_solutions - legacy FaceMesh API (MediaPipe <= 0.10.14 on Python 3.11)
  2. mediapipe_tasks     - FaceLandmarker Tasks API; needs models/face_landmarker.task
  3. opencv_haar         - face boxes only (no landmarks -> landmark features are NaN)
A detector can also be injected (e.g. MockLandmarkDetector) for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.logging_config import get_logger
from src.vision.landmark_indices import LEFT_EYE_EAR, RIGHT_EYE_EAR

log = get_logger("face_landmarks")


@dataclass
class LandmarkResult:
    landmarks: np.ndarray | None      # (N,3) normalised x,y in [0,1], z relative; primary face only
    num_faces: int
    bbox: tuple[float, float, float, float] | None   # normalised x0,y0,x1,y1 of primary face
    backend: str

    @property
    def face_detected(self) -> bool:
        return self.num_faces > 0


def _bbox_from_landmarks(lm: np.ndarray) -> tuple[float, float, float, float]:
    return float(lm[:, 0].min()), float(lm[:, 1].min()), float(lm[:, 0].max()), float(lm[:, 1].max())


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class FaceLandmarkDetector:
    def __init__(self, cfg: dict, backend: str | None = None):
        vcfg = cfg["vision"]
        self.max_faces = int(vcfg.get("max_num_faces", 2))
        self.min_det = float(vcfg.get("min_detection_confidence", 0.5))
        self.min_track = float(vcfg.get("min_tracking_confidence", 0.5))
        requested = backend or vcfg.get("landmark_backend", "auto")
        from src.utils.config import resolve_path
        self.model_path = resolve_path(cfg, "face_landmarker_model")
        self._impl = None
        self.backend = "none"
        order = ["mediapipe_solutions", "mediapipe_tasks", "opencv_haar"] if requested == "auto" else [requested]
        errors = []
        for name in order:
            try:
                getattr(self, f"_init_{name}")()
                self.backend = name
                break
            except Exception as exc:  # backend unavailable -> try next
                errors.append(f"{name}: {exc}")
        if self.backend == "none":
            log.warning("No face backend available: %s", errors)
        elif errors:
            log.info("Face backend '%s' selected (unavailable: %s)", self.backend, "; ".join(errors))

    # ---- backend initialisers -------------------------------------------------
    def _init_mediapipe_solutions(self):
        import mediapipe as mp
        fm = mp.solutions.face_mesh  # AttributeError on new MediaPipe builds
        self._impl = fm.FaceMesh(static_image_mode=False, max_num_faces=self.max_faces,
                                 refine_landmarks=True, min_detection_confidence=self.min_det,
                                 min_tracking_confidence=self.min_track)

    def _init_mediapipe_tasks(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"{self.model_path} missing (run scripts/download_face_model.py)")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=self.max_faces,
            min_face_detection_confidence=self.min_det, min_face_presence_confidence=self.min_det)
        self._impl = vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    def _init_opencv_haar(self):
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(str(path))
        if clf.empty():
            raise RuntimeError("Haar cascade could not be loaded")
        self._impl = clf

    # ---- detection ------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> LandmarkResult:
        if self._impl is None:
            return LandmarkResult(None, 0, None, "none")
        h, w = frame_bgr.shape[:2]
        if self.backend == "opencv_haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._impl.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                return LandmarkResult(None, 0, None, self.backend)
            boxes = [(x / w, y / h, (x + fw) / w, (y + fh) / h) for x, y, fw, fh in faces]
            return LandmarkResult(None, len(boxes), max(boxes, key=_area), self.backend)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces: list[np.ndarray] = []
        if self.backend == "mediapipe_solutions":
            res = self._impl.process(rgb)
            for f in (res.multi_face_landmarks or []):
                faces.append(np.array([[p.x, p.y, p.z] for p in f.landmark], dtype=np.float32))
        else:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self._impl.detect(image)
            for f in (res.face_landmarks or []):
                faces.append(np.array([[p.x, p.y, p.z] for p in f], dtype=np.float32))
        if not faces:
            return LandmarkResult(None, 0, None, self.backend)
        primary = max(faces, key=lambda lm: _area(_bbox_from_landmarks(lm)))
        return LandmarkResult(primary, len(faces), _bbox_from_landmarks(primary), self.backend)

    def close(self) -> None:
        if self._impl is not None and hasattr(self._impl, "close"):
            try:
                self._impl.close()
            except Exception:  # pragma: no cover
                pass
        self._impl = None


def draw_landmarks(frame_bgr: np.ndarray, result: LandmarkResult, draw_mesh: bool = True) -> np.ndarray:
    """Return an annotated copy: face box, sparse mesh and eye contours."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if result.bbox is not None:
        x0, y0, x1, y1 = result.bbox
        cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (80, 200, 120), 2)
    if result.landmarks is not None:
        lm = result.landmarks
        if draw_mesh:
            for x, y, _ in lm[::4]:
                cv2.circle(out, (int(x * w), int(y * h)), 1, (200, 200, 200), -1)
        for idx in (RIGHT_EYE_EAR, LEFT_EYE_EAR):
            pts = np.array([[int(lm[i, 0] * w), int(lm[i, 1] * h)] for i in idx], dtype=np.int32)
            cv2.polylines(out, [pts], True, (255, 180, 0), 1)
        if lm.shape[0] >= 478:
            for i in (468, 473):
                cv2.circle(out, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 3, (0, 140, 255), -1)
    if result.num_faces > 1:
        cv2.putText(out, f"{result.num_faces} faces detected", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
