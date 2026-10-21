"""Tests for the STEW loader and label builder."""

import numpy as np
import pytest

from eegengine.io.stew import (
    CHANNELS,
    DEFAULT_STEW_DIR,
    FS,
    N_SAMPLES,
    load_ratings,
    load_recording,
    load_stew,
    rating_to_level,
    recordings_to_frame,
    to_mne_raw,
)

MISSING = {5, 24, 42}

needs_data = pytest.mark.skipif(not (DEFAULT_STEW_DIR / "ratings.txt").exists(),
                                reason="STEW data not found in data/raw/STEW")


@pytest.fixture(scope="session")
def recs(tmp_path_factory):
    return load_stew(cache_path=tmp_path_factory.mktemp("cache") / "stew.npz")


# ----------------------------------------------------------------- labels (no data needed)
@pytest.mark.parametrize("rating,level", [(1, 0), (3, 0), (4, 1), (6, 1), (7, 2), (9, 2), (np.nan, -1)])
def test_rating_to_level(rating, level):
    assert rating_to_level(rating) == level


def test_rating_out_of_range():
    with pytest.raises(ValueError):
        rating_to_level(10)


# ----------------------------------------------------------------- ratings file
@needs_data
def test_ratings_missing_subjects():
    r = load_ratings()
    assert len(r) == 48
    for s in MISSING:
        assert np.isnan(r[s][0]) and np.isnan(r[s][1])
    rated = [s for s in r if s not in MISSING]
    assert all(not np.isnan(r[s][0]) and not np.isnan(r[s][1]) for s in rated)
    assert r[1] == (2.0, 8.0)       # first line of ratings.txt
    assert r[48] == (1.0, 7.0)      # last line


# ----------------------------------------------------------------- recordings
@needs_data
def test_all_recordings_loaded(recs):
    assert len(recs) == 96
    assert {r.subject_id for r in recs} == set(range(1, 49))
    assert sum(r.condition == "lo" for r in recs) == 48


@needs_data
def test_shapes_and_metadata(recs):
    for r in recs:
        assert r.data.shape == (14, N_SAMPLES)
        assert r.fs == FS == 128
        assert list(r.ch_names) == CHANNELS
        assert r.duration_s == pytest.approx(150.0)
        assert np.isfinite(r.data).all()


@needs_data
def test_binary_labels(recs):
    for r in recs:
        assert r.label_binary == (1 if r.condition == "hi" else 0)


@needs_data
def test_missing_ratings_are_masked_not_dropped(recs):
    for r in recs:
        if r.subject_id in MISSING:
            assert np.isnan(r.rating)
            assert r.label_level == -1 and not r.level_mask
        else:
            assert 1 <= r.rating <= 9
            assert r.label_level in (0, 1, 2) and r.level_mask
    # missing-rating subjects are still present for the binary task
    assert {r.subject_id for r in recs if r.subject_id in MISSING} == MISSING


@needs_data
def test_each_recording_uses_its_own_rating(recs):
    by_key = {r.key: r for r in recs}
    assert by_key["sub01_lo"].rating == 2 and by_key["sub01_lo"].label_level == 0
    assert by_key["sub01_hi"].rating == 8 and by_key["sub01_hi"].label_level == 2


@needs_data
def test_file_orientation_matches_raw_text(recs):
    """Row 0 of the file = sample 0 of every channel (guards against a missed transpose)."""
    first_line = (DEFAULT_STEW_DIR / "sub01_lo.txt").open().readline().split()
    np.testing.assert_allclose(recs[0].data[:, 0], np.array(first_line, dtype=float))


@needs_data
def test_dc_offset_present_in_raw(recs):
    # Emotiv offset ~4000 uV; the loader must not silently alter values.
    assert 2000 < recs[0].data.mean() < 6000


@needs_data
def test_cache_roundtrip(tmp_path):
    a = load_stew(subjects=[1, 2], cache_path=tmp_path / "c.npz")
    b = load_stew(subjects=[1, 2], cache_path=tmp_path / "c.npz")
    assert (tmp_path / "c.npz").exists()
    for x, y in zip(a, b):
        assert x.key == y.key
        assert np.array_equal(x.data, y.data)
        assert x.rating == y.rating


@needs_data
def test_frame(recs):
    df = recordings_to_frame(recs)
    assert len(df) == 96
    assert df.loc[df.subject_id.isin(MISSING), "level_mask"].eq(False).all()
    assert df.label_binary.sum() == 48


@needs_data
def test_mne_conversion():
    raw = to_mne_raw(load_recording(1, "lo"))
    assert raw.ch_names == CHANNELS
    assert raw.info["sfreq"] == 128
    assert raw.n_times == N_SAMPLES
    data = raw.get_data()
    assert abs(data.mean()) < 1e-9                 # DC removed (volts)
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(14)])
    assert np.isfinite(pos).all() and (np.abs(pos).sum(axis=1) > 0).all()  # montage applied
