"""Phase 3 tests: connectivity, band powers, graph metrics, feature table structure."""

import numpy as np
import pytest
from scipy.signal import coherence as scipy_coherence

from eegengine.config import load_config
from eegengine.features.connectivity import band_filter, coherence_matrix, plv_matrix, upper_triangle_columns
from eegengine.features.entropy import hjorth, spectral_entropy
from eegengine.features.extract import FAMILIES, extract_features, family_columns
from eegengine.features.graph import graph_metrics, threshold_matrix
from eegengine.features.spectral import BANDS, band_power, welch_psd
from eegengine.features.wavelet import wavelet_features
from eegengine.io.stew import CHANNELS

FS = 128
rng = np.random.default_rng(0)


# ----------------------------------------------------------------- PLV
def test_plv_identical_signals_is_one():
    x = rng.standard_normal(512)
    X = np.tile(x, (1, 3, 1))
    P = plv_matrix(band_filter(X, FS, (8, 13)))
    np.testing.assert_allclose(P[0], 1.0, atol=1e-6)


def test_plv_constant_phase_lag_is_one():
    t = np.arange(512) / FS
    X = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 10 * t + 1.0)])[None]
    assert plv_matrix(X)[0, 0, 1] > 0.99


def test_plv_random_phase_is_near_zero():
    X = rng.standard_normal((50, 2, 4096))        # independent noise, long epochs
    P = plv_matrix(band_filter(X, FS, (8, 13)))
    assert P[:, 0, 1].mean() < 0.1


def test_plv_symmetric_unit_diagonal():
    P = plv_matrix(band_filter(rng.standard_normal((2, 14, 512)), FS, (4, 8)))
    np.testing.assert_allclose(P, P.transpose(0, 2, 1), atol=1e-10)
    np.testing.assert_allclose(np.diagonal(P, axis1=1, axis2=2), 1.0, atol=1e-10)


# ----------------------------------------------------------------- coherence
def test_coherence_matches_scipy():
    X = rng.standard_normal((1, 2, 512))
    X[0, 1] = 0.7 * X[0, 0] + 0.3 * rng.standard_normal(512)
    ours = coherence_matrix(X, FS, {"alpha": (8, 13)}, nperseg=128)["alpha"][0, 0, 1]
    f, c = scipy_coherence(X[0, 0], X[0, 1], fs=FS, nperseg=128, window="hann", detrend="constant")
    ref = c[(f >= 8) & (f <= 13)].mean()
    assert abs(ours - ref) < 1e-6


def test_coherence_bounds():
    C = coherence_matrix(rng.standard_normal((3, 14, 512)), FS, {"beta": (13, 30)})["beta"]
    assert (C >= -1e-6).all() and (C <= 1 + 1e-6).all()
    np.testing.assert_allclose(np.diagonal(C, axis1=1, axis2=2), 1.0, atol=1e-5)


# ----------------------------------------------------------------- band power
def test_relative_band_powers_sum_to_one():
    X = rng.standard_normal((4, 14, 512))
    f, p = welch_psd(X, FS)
    bp = band_power(f, p)
    rel = bp / bp.sum(-1, keepdims=True)
    np.testing.assert_allclose(rel.sum(-1), 1.0, atol=1e-12)


def test_band_powers_sum_to_total_1_45():
    X = rng.standard_normal((4, 14, 512))
    f, p = welch_psd(X, FS)
    total = p[..., (f >= 1) & (f <= 45)].sum(-1) * (f[1] - f[0])
    np.testing.assert_allclose(band_power(f, p).sum(-1), total, rtol=1e-10)


def test_alpha_sine_lands_in_alpha():
    t = np.arange(512) / FS
    X = np.sin(2 * np.pi * 10 * t)[None, None, :]
    f, p = welch_psd(X, FS)
    bp = band_power(f, p)[0, 0]
    assert np.argmax(bp) == list(BANDS).index("alpha")
    # Parseval: power of a unit sine = 0.5
    assert abs(bp.sum() - 0.5) < 0.05


# ----------------------------------------------------------------- entropy / hjorth / wavelet
def test_spectral_entropy_white_vs_sine():
    t = np.arange(4096) / FS
    f, p_sine = welch_psd(np.sin(2 * np.pi * 10 * t)[None, None], FS)
    _, p_white = welch_psd(rng.standard_normal((1, 1, 4096)), FS)
    assert spectral_entropy(f, p_white)[0, 0] > 0.9
    assert spectral_entropy(f, p_sine)[0, 0] < 0.3


def test_hjorth_mobility_of_sine():
    t = np.arange(4096) / FS
    _, mob, _ = hjorth(np.sin(2 * np.pi * 8 * t)[None, None])
    # mobility ~ 2*pi*f/fs for a pure sine
    assert abs(mob[0, 0] - 2 * np.pi * 8 / FS) < 0.01


def test_wavelet_relative_energy_sums_to_one():
    cols = wavelet_features(rng.standard_normal((3, 14, 512)), CHANNELS)
    rel = sum(cols[k] for k in cols if k.startswith("wavelet_relenergy_") and k.endswith("_AF3"))
    np.testing.assert_allclose(rel, 1.0, atol=1e-10)


# ----------------------------------------------------------------- graph
def test_proportional_threshold_keeps_fraction():
    W = rng.uniform(size=(14, 14))
    W = (W + W.T) / 2
    A = threshold_matrix(W, "proportional", 0.3)
    n_edges = (A[np.triu_indices(14, 1)] > 0).sum()
    assert n_edges == round(0.3 * 91)
    assert np.all(np.diag(A) == 0)


def test_graph_metrics_complete_graph():
    W = np.ones((14, 14))
    res = graph_metrics(threshold_matrix(W, "none"), "none")
    np.testing.assert_allclose(res["strength"], 13)
    np.testing.assert_allclose(res["clustering"], 1.0)
    np.testing.assert_allclose(res["betweenness"], 0.0)
    assert res["efficiency"] == pytest.approx(1.0)
    assert res["entropy"] == pytest.approx(1.0)       # all nodes equal strength


def test_graph_metrics_disconnected_efficiency():
    A = np.zeros((14, 14))
    A[0, 1] = A[1, 0] = 1.0
    res = graph_metrics(A, "absolute")
    assert res["efficiency"] == pytest.approx(1 / 91)  # only one reachable pair (of 91) at distance 1


# ----------------------------------------------------------------- full extractor
def test_extract_features_structure():
    cfg = load_config("features.yaml")
    X = rng.standard_normal((3, 14, 512)) * 10
    F, arrays = extract_features(X, FS, CHANNELS, cfg)
    assert len(F) == 3 and np.isfinite(F.to_numpy()).all()
    prefixes = {c.split("_", 1)[0] for c in F.columns}
    assert prefixes == set(FAMILIES)
    assert len(family_columns(F.columns, "plv")) == 91 * 4
    assert len(family_columns(F.columns, "spectral")) == 14 * 5 * 2 + 14 * 3 + 3
    assert arrays["plv"].shape == (3, 4, 14, 14) and arrays["bandpower"].shape == (3, 14, 5)


def test_family_selection():
    cfg = {**load_config("features.yaml"), "families": ["spectral"]}
    F, _ = extract_features(rng.standard_normal((2, 14, 512)), FS, CHANNELS, cfg)
    assert all(c.startswith("spectral_") for c in F.columns)


def test_upper_triangle_names():
    M = np.zeros((1, 1, 14, 14), dtype=np.float32)
    M[0, 0, 0, 1] = 0.5
    cols = upper_triangle_columns(M, "plv", ["alpha"], CHANNELS)
    assert len(cols) == 91 and cols["plv_alpha_AF3-F7"][0] == 0.5
