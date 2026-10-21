import math

import numpy as np
import pandas as pd
import pytest

from src.data.feature_schema import (ALL_COLUMNS, FEATURE_COLUMNS, FEATURE_GROUPS, SCHEMA_VERSION, SchemaError,
                                     empty_row, schema_fingerprint, validate_feature_frame)
from src.data.synthetic import generate_synthetic_features
from src.vision.eye_features import BlinkDetector, compute_ear, eye_aspect_ratio
from src.vision.feature_extractor import FrameFeatureExtractor
from src.vision.gaze_features import compute_gaze_proxy
from src.vision.head_pose import estimate_head_pose
from src.vision.mock_landmarks import MockLandmarkDetector, make_mock_landmarks


def test_groups_cover_features_exactly():
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert len(schema_fingerprint()) == 16


def test_empty_row_has_all_columns():
    r = empty_row()
    assert set(r) == set(ALL_COLUMNS) and r["schema_version"] == SCHEMA_VERSION


def test_validate_synthetic_frame_and_real_rejection():
    df = generate_synthetic_features(2, 1, 4, 5)
    out = validate_feature_frame(df)
    assert out["is_synthetic"].all()
    with pytest.raises(SchemaError):
        validate_feature_frame(df, allow_synthetic=False)


def test_missing_columns_and_bad_version():
    df = generate_synthetic_features(1, 1, 2, 5)
    with pytest.raises(SchemaError):
        validate_feature_frame(df.drop(columns=["ear_mean"]))
    with pytest.raises(SchemaError):
        validate_feature_frame(df.assign(schema_version="0.0.1"))


def test_invalid_timestamps_rejected():
    df = generate_synthetic_features(1, 1, 2, 5)
    df.loc[3, "timestamp_s"] = np.nan
    with pytest.raises(SchemaError):
        validate_feature_frame(df)
    df.loc[3, "timestamp_s"] = -1
    with pytest.raises(SchemaError):
        validate_feature_frame(df)


def test_label_range_validation():
    df = generate_synthetic_features(1, 1, 2, 5)
    df["label_engagement"] = 7
    with pytest.raises(SchemaError):
        validate_feature_frame(df, require_labels=True)


def test_ear_values():
    open_pts = np.array([[0, 0], [10, -3], [20, -3], [30, 0], [20, 3], [10, 3]], float)
    assert eye_aspect_ratio(open_pts) == pytest.approx(0.2)
    lm = make_mock_landmarks(eye_open=1.0)
    _, _, m = compute_ear(lm, 640, 480)
    assert m == pytest.approx(0.30, abs=0.01)
    assert all(math.isnan(v) for v in compute_ear(None, 640, 480))


def test_blink_detector_counts_blinks_and_rate():
    b = BlinkDetector(threshold=0.2, min_frames=2, max_frames=5, window_s=60, min_elapsed_s=1)
    seq = [0.3, 0.1, 0.1, 0.3, 0.3, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]  # 1 blink, 1 too short, 1 too long
    events = [b.update(e, i * 0.1) for i, e in enumerate(seq)]
    assert sum(events) == 1
    assert b.blink_rate(1.3) == pytest.approx(60 / 1.3, rel=1e-6)
    assert not b.update(float("nan"), 1.4)


def test_gaze_and_head_pose_on_mock_face():
    lm = make_mock_landmarks(gaze_shift=0.0)
    h, v = compute_gaze_proxy(lm, 640, 480)
    assert abs(h) < 1e-3 and abs(v) < 1e-3
    h2, _ = compute_gaze_proxy(make_mock_landmarks(gaze_shift=0.5), 640, 480)
    assert h2 == pytest.approx(0.2, abs=1e-3)   # iris shifted by 0.2 of the half eye-width
    yaw, pitch, roll = estimate_head_pose(lm, 640, 480)
    assert abs(yaw) < 0.1 and abs(pitch) < 0.1 and abs(roll) < 0.1
    assert math.isnan(compute_gaze_proxy(lm[:468], 640, 480)[0])   # no iris -> NaN, not a guess
    assert all(math.isnan(x) for x in estimate_head_pose(lm[:100], 640, 480))


@pytest.mark.parametrize("yaw,pitch,roll", [(20, 0, 0), (-30, 0, 0), (0, 15, 0), (0, -15, 0), (0, 0, 10),
                                            (25, -10, 8), (-35, 20, -12), (40, 25, -20)])
@pytest.mark.parametrize("eye_open", [1.0, 0.05])
def test_head_pose_recovers_known_rotation(yaw, pitch, roll, eye_open):
    lm = make_mock_landmarks(yaw=yaw, pitch=pitch, roll=roll, eye_open=eye_open)
    est = estimate_head_pose(lm, 640, 480)
    assert est == pytest.approx((yaw, pitch, roll), abs=0.5)


def test_head_pose_sign_convention():
    from src.vision import landmark_indices as L

    def nose_offset(**kw):
        lm = make_mock_landmarks(**kw)
        centre = (lm[L.FOREHEAD, :2] + lm[L.CHIN, :2]) / 2
        return (lm[L.NOSE_TIP, :2] - centre) * [640, 480]
    neutral = nose_offset()
    assert nose_offset(yaw=20)[0] > neutral[0] + 5          # yaw > 0: face turns to image right
    assert nose_offset(pitch=15)[1] > neutral[1] + 5        # pitch > 0: nose moves down (looking down)
    lm = make_mock_landmarks(roll=10)
    assert lm[263, 1] > lm[33, 1]                           # roll > 0: clockwise (image-right eye lower)


def test_canonical_face_model_file():
    from src.vision.face_model import N_VERTICES, camera_frame, canonical_vertices
    v = canonical_vertices()
    assert v.shape == (N_VERTICES, 3) and np.isfinite(v).all()
    m = camera_frame()
    assert np.allclose(m[1], 0) and m[152, 1] > 0 and m[10, 1] < 0    # nose origin, chin down, forehead up


def test_extractor_row_conforms_and_handles_bad_input(cfg):
    fe = FrameFeatureExtractor(cfg, detector=MockLandmarkDetector([{}, {"num_faces": 0}, {"num_faces": 2}]))
    frame = np.full((480, 640, 3), 120, np.uint8)
    r1, _ = fe.extract(frame, 0.0, 0)
    r2, _ = fe.extract(frame, 0.2, 1)
    r3, _ = fe.extract(frame, 0.4, 2)
    r4, _ = fe.extract(np.zeros((5, 5), np.uint8), 0.6, 3)            # invalid frame
    r5, _ = fe.extract(np.full((480, 640, 3), 10, np.uint8), 0.8, 4)   # too dark
    df = validate_feature_frame(pd.DataFrame([r1, r2, r3, r4, r5]))
    assert list(df["quality_ok"]) == [True, False, False, False, False]
    assert df.loc[2, "multiple_faces"] and df.loc[4, "low_light"]
    assert math.isnan(r2["ear_mean"])
    with pytest.raises(ValueError):
        fe.extract(frame, 0.1, 5)          # non-monotonic timestamp
    with pytest.raises(ValueError):
        fe.extract(frame, float("nan"), 6)
