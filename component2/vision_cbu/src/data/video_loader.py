"""Video indexing, metadata probing and frame sampling (OpenCV)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.data.annotation_parser import clip_key

DEFAULT_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def index_videos(video_dir: Path, extensions=DEFAULT_EXTS) -> dict[str, Path]:
    """Map clip key -> path for every video under video_dir (recursive).

    Duplicate stems are reported by `find_duplicate_stems`; the first path wins here.
    """
    exts = {e.lower() for e in extensions}
    index: dict[str, Path] = {}
    if not video_dir.exists():
        return index
    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            index.setdefault(clip_key(p.name), p)
    return index


def find_duplicate_stems(video_dir: Path, extensions=DEFAULT_EXTS) -> dict[str, list[str]]:
    exts = {e.lower() for e in extensions}
    seen: dict[str, list[str]] = {}
    if video_dir.exists():
        for p in video_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                seen.setdefault(clip_key(p.name), []).append(str(p))
    return {k: v for k, v in seen.items() if len(v) > 1}


def participant_from_path(video_path: Path, video_root: Path) -> str | None:
    """Participant ID from the official DAiSEE layout <split>/<subject>/<clip>/<clip>.avi.

    Returns the folder two levels above the file (the subject folder) only when that
    layout is present. Returns None rather than guessing when the layout differs.
    """
    try:
        rel = video_path.relative_to(video_root).parts
    except ValueError:
        return None
    # rel e.g. ('DataSet','Train','110001','1100011002','1100011002.avi') or ('Train','110001',...)
    if len(rel) >= 3 and rel[-2] == video_path.stem:
        subject = rel[-3]
        if subject.isdigit():
            return subject
    return None


def split_from_path(video_path: Path) -> str | None:
    for part in (s.lower() for s in video_path.parts):
        if part == "train":
            return "train"
        if part in ("validation", "val"):
            return "validation"
        if part == "test":
            return "test"
    return None


def probe_video(path: Path) -> dict:
    """Read real metadata from the container. Never fabricates values."""
    info = {"path": str(path), "readable": False, "frame_count": None, "fps": None,
            "width": None, "height": None, "duration_s": None, "error": None}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "cannot open video"
            return info
        ok, frame = cap.read()
        if not ok or frame is None:
            info["error"] = "cannot decode first frame"
            return info
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info.update(readable=True,
                    fps=float(fps) if fps and fps > 0 else None,
                    frame_count=int(n) if n and n > 0 else None,
                    width=int(frame.shape[1]), height=int(frame.shape[0]))
        if info["fps"] and info["frame_count"]:
            info["duration_s"] = round(info["frame_count"] / info["fps"], 3)
        return info
    finally:
        cap.release()


def iter_sampled_frames(path: Path, target_fps: float, max_frames: int | None = None
                        ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, frame_bgr) sampled at ~target_fps.

    Timestamps come from the container position when available, otherwise index/fps.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video: {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / target_fps))) if target_fps else 1
        idx = emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step == 0:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else idx / src_fps
                yield idx, float(ts), frame
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
