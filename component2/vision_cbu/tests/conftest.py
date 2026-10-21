"""Shared fixtures. Nothing here needs a webcam, a download, or the real DAiSEE dataset."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, with_root  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Config whose relative paths point into an isolated temporary project root."""
    c = with_root(load_config(), tmp_path)
    c["temporal"]["epochs"] = 3
    c["evaluation"]["latency_repeats"] = 3
    for key in ("daisee_videos", "daisee_annotations", "features_dir", "sequences_dir", "demo_dir", "models_dir"):
        (tmp_path / c["paths"][key]).mkdir(parents=True, exist_ok=True)
    return c


def write_video(path: Path, n_frames: int = 30, fps: float = 30.0, size=(160, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), 90 + (i % 20), np.uint8))
    w.release()


@pytest.fixture
def fake_daisee(cfg, tmp_path):
    """Tiny DAiSEE-shaped dataset: DataSet/<Split>/<subject>/<clip>/<clip>.avi + Labels CSVs.

    Generated test fixture only - not real data. Column names mimic the official files,
    including the trailing space DAiSEE has on 'Frustration '.
    """
    vroot = tmp_path / cfg["paths"]["daisee_videos"] / "DataSet"
    aroot = tmp_path / cfg["paths"]["daisee_annotations"]
    rng = np.random.default_rng(0)
    layout = {"Train": ["110001", "110002", "110003", "110004"], "Validation": ["210001"], "Test": ["310001", "310002"]}
    files = {}
    for split, subjects in layout.items():
        lines = ["ClipID,Boredom,Engagement,Confusion,Frustration "]
        for s in subjects:
            for k in range(2):
                clip = f"{s}{k:04d}"
                write_video(vroot / split / s / clip / f"{clip}.avi", n_frames=60)
                lines.append(f"{clip}.avi,{rng.integers(0, 4)},{(k + int(s[-1])) % 4},{rng.integers(0, 4)},{rng.integers(0, 4)}")
        name = {"Train": "TrainLabels.csv", "Validation": "ValidationLabels.csv", "Test": "TestLabels.csv"}[split]
        (aroot / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[split] = aroot / name
    return {"videos": vroot, "annotations": aroot, "files": files}
