import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_features
from src.utils.privacy import (ConsentRequiredError, delete_sessions, pseudonym_for, pseudonymize, save_session,
                               to_csv_bytes, to_json_bytes)
from src.vision.feature_extractor import FrameFeatureExtractor
from src.vision.mock_landmarks import MockLandmarkDetector
from src.webcam.camera import Camera, CameraError, MockCamera

ROOT = Path(__file__).resolve().parents[1]


def test_pseudonymization_stable_and_salted(cfg):
    a = pseudonymize("alice", "salt1")
    assert a == pseudonymize("alice", "salt1") and a != pseudonymize("alice", "salt2")
    assert a.startswith("P_") and "alice" not in a
    assert pseudonym_for(cfg, "bob") == pseudonym_for(cfg, "bob")


def test_session_saving_requires_enable_and_consent(cfg):
    df = generate_synthetic_features(1, 1, 2, 5)
    with pytest.raises(ConsentRequiredError):
        save_session(df, cfg, consent=True)                 # logging disabled by default
    cfg["privacy"]["enable_session_logging"] = True
    with pytest.raises(ConsentRequiredError):
        save_session(df, cfg, consent=False)
    p = save_session(df, cfg, consent=True, session_label="P_x")
    assert p.exists() and p.suffix == ".csv"
    assert delete_sessions(cfg) == 1 and not p.exists()


def test_defaults_do_not_store_video(cfg):
    assert cfg["privacy"]["store_raw_video"] is False
    assert cfg["privacy"]["enable_session_logging"] is False
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    assert "VideoWriter" not in src and "imwrite" not in src     # no code path writes video/frames


def test_csv_and_json_exports():
    df = generate_synthetic_features(1, 1, 2, 5)
    back = pd.read_csv(pd.io.common.BytesIO(to_csv_bytes(df)))
    assert list(back.columns) == list(df.columns) and len(back) == len(df)
    payload = json.loads(to_json_bytes(df.head(3), synthetic=True))
    assert payload["is_synthetic"] and "DEMO ONLY" in payload["notice"] and len(payload["data"]) == 3


def test_mock_camera_frames_and_cleanup(cfg):
    cam = MockCamera(sample_every_n=2)
    fe = FrameFeatureExtractor(cfg, detector=MockLandmarkDetector())
    with cam:
        rows = [fe.extract(f, ts, i)[0] for i, (ts, f) in enumerate(cam.frames(max_frames=5))]
    assert len(rows) == 5 and cam.n_read == 9 and not cam.is_open and cam.released_count == 1
    assert all(r["quality_ok"] for r in rows)


def test_camera_released_on_exception_and_failure():
    cam = MockCamera()
    with pytest.raises(RuntimeError):
        with cam:
            for _ in cam.frames():
                raise RuntimeError("boom")
    assert not cam.is_open and cam.released_count == 1
    cam2 = MockCamera(fail_after=3)
    with pytest.raises(CameraError):
        with cam2:
            list(cam2.frames(max_consecutive_failures=5))
    assert not cam2.is_open
    with pytest.raises(CameraError):
        MockCamera().read()                                  # not opened


def test_missing_physical_camera_raises_cleanly(cfg):
    cfg["camera"]["index"] = 97                              # surely absent
    cfg["camera"]["backend"] = "any"
    cam = Camera(cfg)
    with pytest.raises(CameraError):
        cam.open()
    assert not cam.is_open


def test_mock_camera_duration_uses_camera_clock():
    with MockCamera(fps=30.0) as cam:
        frames = list(cam.frames(max_seconds=1.0))
    assert len(frames) == 30 and not cam.is_open
