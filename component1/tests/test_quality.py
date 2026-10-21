"""Tests for the signal quality checks, using synthetic signals with known defects."""

import numpy as np
import pytest

from eegengine.io.stew import CHANNELS
from eegengine.preprocess.dc import remove_dc
from eegengine.quality.checks import QualityConfig, assess_recording, assess_window, robust_z, window_starts

FS = 128


def clean_window(seconds=4, seed=0):
    """14 channels of pink-ish noise (8-12 uV std per channel), resembling clean EEG."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    white = rng.standard_normal((14, n))
    # 1/f shaping in the frequency domain, then a 0-40 Hz band limit
    f = np.fft.rfftfreq(n, 1 / FS)
    shape = np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1)), 0) * (f < 40)
    x = np.fft.irfft(np.fft.rfft(white, axis=1) * shape, n=n, axis=1)
    amp = rng.uniform(8, 12, size=(14, 1))   # channels differ a little, as in real EEG
    return amp * x / x.std(axis=1, keepdims=True)


@pytest.fixture
def cfg():
    return QualityConfig()


def test_clean_window_is_good(cfg):
    res = assess_window(clean_window(), FS, cfg)
    assert res["flag"] == "good" and res["n_bad"] == 0 and res["score"] == 1.0


def test_flatline_detected(cfg):
    x = clean_window()
    x[3] = 0.0
    res = assess_window(x, FS, cfg)
    assert res["bad"]["flat"][3] and res["bad"]["flat"].sum() == 1
    assert res["flag"] == "warn"


def test_excessive_amplitude_detected(cfg):
    x = clean_window()
    x[0, 100:120] += 400.0   # blink-like deflection on AF3
    res = assess_window(x, FS, cfg)
    assert res["bad"]["amplitude"][0]


def test_abnormal_variance_detected(cfg):
    x = clean_window()
    x[5] *= 8.0              # one channel 64x the variance of the others
    res = assess_window(x, FS, cfg)
    assert res["bad"]["variance"][5]
    assert res["bad"]["variance"].sum() == 1


def test_line_noise_detected(cfg):
    x = clean_window()
    t = np.arange(x.shape[1]) / FS
    x[7] += 20 * np.sin(2 * np.pi * 50 * t)
    res = assess_window(x, FS, cfg)
    assert res["bad"]["line"][7]


def test_high_frequency_noise_detected(cfg):
    x = clean_window()
    t = np.arange(x.shape[1]) / FS
    for f0 in (32, 36, 40, 44):                     # EMG-like power in 30-45 Hz
        x[9] += 8 * np.sin(2 * np.pi * f0 * t)
    res = assess_window(x, FS, cfg)
    assert res["bad"]["hf"][9]


def test_many_bad_channels_flag_bad(cfg):
    x = clean_window()
    x[:4] = 0.0
    assert assess_window(x, FS, cfg)["flag"] == "bad"


def test_window_starts():
    s = window_starts(19200, FS, 4.0, 2.0)
    assert s[0] == 0 and s[1] == 256 and s[-1] + 512 <= 19200
    assert len(s) == 74      # (150 - 4) / 2 + 1


def test_assess_recording_marks_not_deletes(cfg):
    x = np.concatenate([clean_window(seed=i) for i in range(5)], axis=1)  # 20 s
    x[2, 4 * FS:5 * FS] = 0.0          # flatline inside window 1-2 only
    rows, flags = assess_recording(x, FS, CHANNELS, cfg)
    assert len(rows) == len(window_starts(x.shape[1], FS, 4, 2))  # every window kept
    assert rows[0]["flag"] == "good"
    assert any(r["flag"] != "good" for r in rows)
    assert flags.shape == (len(rows), 5, 14)


def test_robust_z_constant():
    assert np.all(robust_z(np.ones(14)) == 0)


def test_remove_dc():
    rng = np.random.default_rng(0)
    x = 4000 + rng.standard_normal((14, 19200))
    y = remove_dc(x, FS)
    assert np.abs(y.mean(axis=1)).max() < 0.1
    assert y.shape == x.shape


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        QualityConfig.from_dict({"not_a_key": 1})
