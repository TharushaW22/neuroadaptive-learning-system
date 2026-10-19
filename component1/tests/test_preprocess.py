"""Phase 2 tests: filters, epoching, bad-channel repair, normalisation, labels."""

import numpy as np
import pandas as pd
import pytest

from eegengine.config import load_config
from eegengine.io.stew import CHANNELS, DEFAULT_STEW_DIR, load_recording, load_stew
from eegengine.labels import add_labels, label_counts, within_task_label
from eegengine.preprocess.bad_channels import detect_bad_channels, repair_bad_channels
from eegengine.preprocess.epoching import make_epochs
from eegengine.preprocess.filters import bandpass, filter_recording, notch
from eegengine.preprocess.normalize import epoch_zscore, rest_baseline_features, rest_baseline_signal
from eegengine.preprocess.pipeline import config_hash, epoch_quality_config, preprocess_recording
from eegengine.quality.checks import QualityConfig, assess_window
from tests.test_quality import clean_window

FS = 128
needs_data = pytest.mark.skipif(not (DEFAULT_STEW_DIR / "ratings.txt").exists(), reason="no STEW data")


def sine(f, seconds=30, amp=1.0):
    t = np.arange(seconds * FS) / FS
    return amp * np.sin(2 * np.pi * f * t)[None, :]


def gain(x_in, x_out):
    mid = slice(x_in.shape[1] // 4, 3 * x_in.shape[1] // 4)  # ignore filter edges
    return x_out[:, mid].std() / x_in[:, mid].std()


# ----------------------------------------------------------------- filters
@pytest.mark.parametrize("f,lo,hi", [(10, 0.97, 1.03), (30, 0.97, 1.03), (2, 0.95, 1.05)])
def test_bandpass_passes_in_band(f, lo, hi):
    x = sine(f)
    assert lo < gain(x, bandpass(x, FS)) < hi


@pytest.mark.parametrize("f", [0.05, 58.0])
def test_bandpass_attenuates_out_of_band(f):
    x = sine(f, seconds=120)
    assert gain(x, bandpass(x, FS)) < 0.1


def test_bandpass_is_zero_phase():
    x = sine(10)
    y = bandpass(x, FS)
    mid = slice(1000, 3000)
    lag = np.argmax(np.correlate(y[0, mid], x[0, mid], mode="full")) - (len(x[0, mid]) - 1)
    assert lag == 0


def test_notch_removes_50hz():
    x = sine(50)
    assert gain(x, notch(x, FS, 50.0)) < 0.1
    assert gain(sine(20), notch(sine(20), FS, 50.0)) > 0.97


def test_filter_recording_removes_dc():
    x = 4000 + np.random.default_rng(0).standard_normal((14, 19200))
    assert np.abs(filter_recording(x, FS, {}).mean(axis=1)).max() < 0.5


# ----------------------------------------------------------------- epoching
@pytest.mark.parametrize("win,step,n", [(4.0, 2.0, 74), (2.0, 1.0, 149), (4.0, 4.0, 37)])
def test_epoch_count(win, step, n):
    x = np.random.default_rng(0).standard_normal((14, 19200))
    ep, starts = make_epochs(x, FS, win, step)
    assert ep.shape == (n, 14, int(win * FS))
    np.testing.assert_array_equal(ep[1], x[:, starts[1]:starts[1] + int(win * FS)])


# ----------------------------------------------------------------- quality: blink tolerance
def test_frontal_blink_is_warn_not_bad():
    x = clean_window()                               # realistic 1/f EEG-like noise
    for ch in ("AF3", "AF4", "F7", "F8"):           # 4 frontal channels blink together
        x[CHANNELS.index(ch), 200:240] += 200 * np.hanning(40)
    res = assess_window(x, FS, QualityConfig())
    assert res["n_bad"] >= 3 and res["n_hard"] == 0 and res["flag"] == "warn"


def test_non_frontal_artifact_is_bad():
    x = clean_window()                               # realistic 1/f EEG-like noise
    for ch in ("O1", "O2", "P7"):
        x[CHANNELS.index(ch), 200:240] += 200 * np.hanning(40)
    assert assess_window(x, FS, QualityConfig())["flag"] == "bad"


# ----------------------------------------------------------------- bad channels
def test_bad_channel_detected_and_interpolated():
    rng = np.random.default_rng(0)
    t = np.arange(60 * FS) / FS
    common = 10 * np.sin(2 * np.pi * 10 * t)                # shared alpha -> interpolation can rebuild it
    x = common + rng.standard_normal((14, t.size)) * 3
    k = CHANNELS.index("T7")
    x[k] = rng.standard_normal(t.size) * 200                 # broken electrode
    qcfg = QualityConfig()
    cfg = {"min_bad_fraction": 0.6, "max_std_ratio": 4.0, "max_interpolated": 3}
    found = detect_bad_channels(x, FS, CHANNELS, qcfg, cfg)
    assert [d["channel"] for d in found] == ["T7"]
    y, log = repair_bad_channels(x, FS, CHANNELS, qcfg, cfg)
    assert log["interpolated"] == ["T7"]
    assert np.isfinite(y).all() and y[k].std() < 30          # rebuilt from neighbours
    np.testing.assert_allclose(y[0], x[0])                   # good channels untouched


# ----------------------------------------------------------------- normalisation
def test_epoch_zscore():
    X = np.random.default_rng(0).standard_normal((5, 14, 512)) * 7 + 3
    Z = epoch_zscore(X)
    np.testing.assert_allclose(Z.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(Z.std(-1), 1, atol=1e-6)


def _toy_meta():
    return pd.DataFrame({"subject_id": [1, 1, 1, 1, 2, 2, 2, 2],
                         "condition": ["lo", "lo", "hi", "hi"] * 2})


def test_rest_baseline_signal_uses_only_rest():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 14, 256))
    X[[2, 3, 6, 7]] *= 5                                     # task epochs much larger
    meta = _toy_meta()
    Z = rest_baseline_signal(X, meta)
    rest = Z[[0, 1]]
    np.testing.assert_allclose(rest.std(axis=(0, 2)), 1, atol=1e-6)
    assert Z[[2, 3]].std() > 3                               # task keeps its larger scale


def test_rest_baseline_features():
    F = pd.DataFrame({"a": [1.0, 3.0, 10.0, 12.0, 0.0, 2.0, 5.0, 5.0]})
    Z = rest_baseline_features(F, _toy_meta())
    np.testing.assert_allclose(Z["a"].to_numpy()[[0, 1, 4, 5]], [-1, 1, -1, 1], atol=1e-6)


# ----------------------------------------------------------------- labels
def test_within_task_label_rules():
    assert within_task_label("hi", 5) == 0
    assert within_task_label("hi", 7) == 1
    assert within_task_label("lo", 2) == -1
    assert within_task_label("hi", np.nan) == -1


def test_add_labels_and_counts():
    meta = pd.DataFrame({"subject_id": [1, 1, 5, 5], "condition": ["lo", "hi", "lo", "hi"],
                         "rating": [2, 8, np.nan, np.nan]})
    m = add_labels(meta)
    assert m.y_binary.tolist() == [0, 1, 0, 1]
    assert m.y_level.tolist() == [0, 2, -1, -1]
    assert m.y_within.tolist() == [-1, 1, -1, -1]
    c = label_counts(m)
    assert c["workload_binary"]["n"] == 4 and c["workload_within_task"]["n"] == 1


@needs_data
def test_real_label_counts():
    recs = load_stew(cache_path=DEFAULT_STEW_DIR.parents[1] / "processed" / "stew_raw_cache.npz")
    meta = add_labels(pd.DataFrame({"subject_id": r.subject_id, "condition": r.condition,
                                    "rating": r.rating} for r in recs))
    c = label_counts(meta)
    assert c["workload_binary"]["classes"] == {"low": 48, "high": 48}
    assert c["workload_level"]["by_condition"]["lo"] == {"low": 42, "medium": 3, "high": 0}
    assert c["workload_level"]["by_condition"]["hi"] == {"low": 0, "medium": 20, "high": 25}
    assert c["workload_within_task"]["classes"] == {"moderate": 20, "high": 25}
    assert c["workload_within_task"]["n_subjects"] == 45


# ----------------------------------------------------------------- full pipeline on one recording
@needs_data
def test_pipeline_one_recording():
    cfg = load_config("preprocess.yaml")
    res = preprocess_recording(load_recording(1, "lo"), cfg)
    assert res["epochs"].shape == (74, 14, 512)
    assert np.isfinite(res["epochs"]).all()
    assert len(res["meta"]) == 74
    assert res["meta"][0]["t_start"] == 0.0 and res["meta"][1]["t_start"] == 2.0
    assert "T7" in res["bad_channel_log"]["interpolated"]   # the broken electrode seen in Phase 1
    assert {m["quality_flag"] for m in res["meta"]} <= {"good", "warn", "bad"}


def test_config_hash_changes_with_epoching():
    cfg = load_config("preprocess.yaml")
    other = {**cfg, "epoching": {"window_s": 2.0, "step_s": 1.0}}
    assert config_hash(cfg) != config_hash(other)
    assert epoch_quality_config(other).window_s == 2.0
