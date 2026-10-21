import numpy as np
import pandas as pd
import pytest

from src.data.feature_schema import FEATURE_COLUMNS
from src.data.sequence_builder import (SequenceSet, build_sequences, clean_timestamps, validate_sequence_lengths,
                                       windows_for_video)
from src.data.synthetic import generate_synthetic_features
from src.data.validate_dataset import assert_no_split_leakage
from src.models.train import prepare_data
from src.processing.preprocessing import Normalizer, forward_fill_within_video


def test_window_count_and_stride():
    vals = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
    w = windows_for_video(vals, np.ones(50, bool), window=20, stride=10, min_valid_ratio=0.5)
    assert [s for s, _, _ in w] == [0, 10, 20, 30]
    assert all(X.shape == (20, 3) and m.sum() == 20 for _, X, m in w)


def test_short_video_is_padded_with_mask():
    w = windows_for_video(np.ones((12, 2), np.float32), np.ones(12, bool), 20, 10, 0.5)
    assert len(w) == 1
    _, X, m = w[0]
    assert m.sum() == 12 and X[12:].sum() == 0


def test_low_quality_windows_dropped():
    q = np.zeros(20, bool)
    q[:5] = True
    assert windows_for_video(np.ones((20, 2), np.float32), q, 20, 10, 0.6) == []
    with pytest.raises(ValueError):
        windows_for_video(np.ones((20, 2)), q, 0, 1, 0.5)


def test_invalid_timestamps_cleaned():
    df = pd.DataFrame({"timestamp_s": [0.4, 0.0, np.nan, 0.2, 0.2, -1, np.inf]})
    out, dropped = clean_timestamps(df)
    assert list(out["timestamp_s"]) == [0.0, 0.2, 0.4] and dropped == 4


def test_windows_never_cross_videos_and_labels_broadcast():
    df = generate_synthetic_features(4, 2, 8, 5, missing_rate=0.0)
    seq, stats = build_sequences(df, FEATURE_COLUMNS, 10, 5, 0.5, "label_engagement")
    assert stats["videos"] == 8 and len(seq) > 0
    vid_label = df.groupby("video_id")["label_engagement"].first()
    for v, y in zip(seq.video_ids, seq.y):
        assert y == vid_label[v]
    validate_sequence_lengths(seq.X, 10, len(FEATURE_COLUMNS))
    with pytest.raises(ValueError):
        validate_sequence_lengths(seq.X, 11, len(FEATURE_COLUMNS))


def test_sequence_save_load_roundtrip(tmp_path):
    df = generate_synthetic_features(2, 1, 6, 5)
    seq, _ = build_sequences(df, FEATURE_COLUMNS, 10, 5, 0.3, "label_engagement")
    seq.save(tmp_path / "s.npz")
    back = SequenceSet.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(np.nan_to_num(back.X), np.nan_to_num(seq.X))
    assert back.feature_columns == FEATURE_COLUMNS


def test_bounded_forward_fill_within_video():
    df = pd.DataFrame({"video_id": ["a"] * 6 + ["b"] * 2, "timestamp_s": list(range(6)) + [0, 1],
                       "f": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0, np.nan, 5.0], "quality_ok": [True] * 8})
    out = forward_fill_within_video(df, ["f"], max_gap=2)
    assert list(out["f"].iloc[:6].fillna(-1)) == [1, 1, 1, -1, -1, 2]
    assert np.isnan(out["f"].iloc[6])      # never filled from video 'a'


def test_normalizer_fit_train_only_and_no_nan():
    tr = np.array([[1.0, np.nan], [3.0, 4.0]])
    n = Normalizer("standard").fit(tr)
    out = n.transform(np.array([[np.nan, np.nan]]))
    assert not np.isnan(out).any()
    n2 = Normalizer.from_dict(n.to_dict())
    np.testing.assert_allclose(n2.transform(tr), n.transform(tr))
    with pytest.raises(RuntimeError):
        Normalizer().transform(tr)


def test_prepare_data_participant_level_splits(cfg):
    df = generate_synthetic_features(12, 2, 8, 5)
    data = prepare_data(cfg, df, "synthetic")
    allp = {"train": set(data.train.participant_ids), "val": set(data.val.participant_ids),
            "test": set(data.test.participant_ids)}
    assert not (allp["train"] & allp["test"]) and not (allp["train"] & allp["val"])
    combined = pd.DataFrame({"participant_id": np.concatenate([data.train.participant_ids, data.test.participant_ids]),
                             "video_id": np.concatenate([data.train.video_ids, data.test.video_ids]),
                             "split": ["train"] * len(data.train) + ["test"] * len(data.test)})
    assert_no_split_leakage(combined)
    assert not np.isnan(data.test.X).any()
