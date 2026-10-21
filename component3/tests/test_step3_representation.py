"""Step 3 (no data needed): band decomposition, feature space, replacement strategies."""

import numpy as np
import pytest

from xaicomp.replacement.strategies import GaussianLogPower, compose, conditional_gains, nearest_donors
from xaicomp.representation.eeg_bands import (band_weights, decompose, log_power, reconstruction_error,
                                              residual_fraction)
from xaicomp.representation.groups import InterpretableSpace

BANDS6 = {"slow": (0.0, 1.0), "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
          "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
FS = 128


def test_filter_bank_is_partition_of_unity():
    W = band_weights(4096, FS, BANDS6, 1.0)
    f = np.fft.rfftfreq(4096, 1 / FS)
    np.testing.assert_allclose(W.sum(0)[f <= 44.5], 1.0, atol=1e-12)
    assert (W >= -1e-12).all() and (W <= 1 + 1e-12).all()


def test_sine_lands_in_its_band_and_reconstruction_is_exact():
    t = np.arange(19200) / FS
    x = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t) + 0.3 * np.sin(2 * np.pi * 0.3 * t)])
    comps, res = decompose(x, FS, BANDS6, 1.0)
    assert comps.shape == (2, 6, 19200)
    p = (comps ** 2).mean(-1)
    assert p[0].argmax() == list(BANDS6).index("alpha")
    assert p[1].argmax() == list(BANDS6).index("beta") and p[1, 0] > 0.01      # 0.3 Hz -> slow band
    assert reconstruction_error(x, comps, res) < 1e-10
    assert residual_fraction(x, res) < 1e-6


def test_space_feature_order_and_aggregation():
    sp = InterpretableSpace(["A", "B", "C"], ["slow", "alpha"], {"front": ["A", "B"], "back": ["C"]})
    assert sp.feature_names == ["A|slow", "A|alpha", "B|slow", "B|alpha", "C|slow", "C|alpha"]
    v = np.arange(6.0)
    ch, names = sp.aggregate(v, "channel")
    assert names == ["A", "B", "C"] and ch.tolist() == [1, 5, 9]
    bd, _ = sp.aggregate(v, "band")
    assert bd.tolist() == [6, 9]
    rg, rn = sp.aggregate(v, "region")
    assert rn == ["front", "back"] and rg.sum() == v.sum()
    with pytest.raises(ValueError):
        InterpretableSpace(["A"], ["x"], {"r": ["Z"]})


def test_d5_regions_cover_all_14_channels_once():
    from xaicomp.config import load_context
    ctx = load_context("eeg_T1_A8")
    chans = [c for v in ctx["regions"].values() for c in v]
    assert len(chans) == 14 and len(set(chans)) == 14


def test_compose_present_absent_and_donor():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2, 50)).astype(np.float32)
    res = rng.normal(size=(3, 50)).astype(np.float32)
    donor = rng.normal(size=(3, 2, 50)).astype(np.float32)
    full = compose(x, res, np.ones((1, 6)), donor)[0]
    np.testing.assert_allclose(full, x.sum(1) + res, atol=1e-5)
    empty0 = compose(x, res, np.zeros((1, 6)), None)[0]
    np.testing.assert_allclose(empty0, res, atol=1e-6)
    m = np.zeros((1, 6)); m[0, 1] = 1                                     # only channel 0, band 1 present
    mixed = compose(x, res, m, donor)[0]
    expect = res + donor.sum(1); expect[0] += x[0, 1] - donor[0, 1]
    np.testing.assert_allclose(mixed, expect, atol=1e-5)
    g = np.full((1, 3, 2), 2.0, np.float32)
    np.testing.assert_allclose(compose(x, res, np.zeros((1, 6)), donor, g)[0], res + 2 * donor.sum(1), atol=1e-5)


def test_gaussian_conditional_matches_formula():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    Z = rng.normal(size=(5000, 3)) @ A.T + np.array([1.0, -2.0, 0.5])
    g = GaussianLogPower(Z)
    present = np.array([True, False, True])
    mean, cov = g.conditional(present, np.array([1.5, 0.0, 0.0])[present])
    S, Ab = present, ~present
    K = g.cov[np.ix_(Ab, S)] @ np.linalg.inv(g.cov[np.ix_(S, S)])
    np.testing.assert_allclose(mean, g.mu[Ab] + K @ (np.array([1.5, 0.0]) - g.mu[S]))
    np.testing.assert_allclose(cov, g.cov[np.ix_(Ab, Ab)] - K @ g.cov[np.ix_(S, Ab)], atol=1e-12)
    m0, c0 = g.conditional(np.zeros(3, bool), np.zeros(0))
    np.testing.assert_allclose(m0, g.mu)


def test_conditional_gains_hit_the_sampled_power_exactly():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 4))
    g = GaussianLogPower(Z)
    donor_z = rng.normal(size=(2, 4))
    masks = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], bool)
    gains, targets = conditional_gains(masks, np.zeros(4), donor_z, g, rng)
    for i in range(2):
        A = ~masks[i]
        for k in range(2):
            np.testing.assert_allclose(donor_z[k, A] + 2 * np.log10(gains[i, k, A]), targets[i, k, A], atol=1e-5)
        assert np.all(gains[i, :, masks[i]] == 1)


def test_nearest_donors_uses_only_present_features():
    pool = np.array([[0.0, 9.0], [1.0, 0.0], [5.0, 0.0]])
    idx = nearest_donors(np.array([True, False]), np.array([0.1, 0.0]), pool, np.ones(2), 1, np.random.default_rng(0))
    assert idx.tolist() == [0]


def test_log_power():
    x = np.ones((2, 3, 10)) * 10
    np.testing.assert_allclose(log_power(x), 2.0)
