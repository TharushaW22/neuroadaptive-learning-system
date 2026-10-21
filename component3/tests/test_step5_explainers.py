"""Step 5: the four explainers on real windows (tiny budgets) — common format and sanity checks."""

import numpy as np
import pytest

from xaicomp.config import load_c3_config
from xaicomp.explainers.base import Explanation
from xaicomp.explainers.registry import explain

TINY = {"kernel_shap": {"nsamples": 120, "k_donors": 2}, "conditional_shap": {"nsamples": 120, "k_donors": 2},
        "lime": {"num_samples": 120, "k_donors": 2}, "ig": {"n_steps": 16, "k_donors": 2}, "ig_zero": {"n_steps": 16}}


@pytest.fixture(scope="module")
def results(adapter):
    cfg = load_c3_config("explainers.yaml")["methods"]
    out = {}
    for m in TINY:
        v = adapter.view("s02_hi_e10", TINY[m].get("k_donors", 0), 0)
        out[m] = explain(m, v, 0, TINY[m], cfg[m])
    return out


def test_common_output_format(results):
    for m, e in results.items():
        assert isinstance(e, Explanation) and e.values.shape == (84,) and np.isfinite(e.values).all()
        assert e.seconds > 0 and e.n_evals > 0 and 0 <= e.fx <= 1


def test_shap_efficiency_is_exact(results):
    for m in ("kernel_shap", "conditional_shap"):
        assert results[m].extras["efficiency_error"] < 1e-6


def test_ig_completeness_small(results):
    for m in ("ig", "ig_zero"):
        assert results[m].extras["completeness_error_rel"] < 0.1


def test_lime_reports_surrogate_r2(results):
    assert 0 <= results["lime"].extras["surrogate_r2"] <= 1


def test_ig_zero_is_deterministic_and_ig_donor_is_not(adapter):
    cfg = load_c3_config("explainers.yaml")["methods"]
    a = explain("ig_zero", adapter.view("s14_lo_e20", 0, 0), 0, TINY["ig_zero"], cfg["ig_zero"]).values
    b = explain("ig_zero", adapter.view("s14_lo_e20", 0, 1), 1, TINY["ig_zero"], cfg["ig_zero"]).values
    np.testing.assert_allclose(a, b)
    c = explain("ig", adapter.view("s14_lo_e20", 2, 0), 0, TINY["ig"], cfg["ig"]).values
    d = explain("ig", adapter.view("s14_lo_e20", 2, 1), 1, TINY["ig"], cfg["ig"]).values
    assert not np.allclose(c, d)


def test_same_seed_reproduces(adapter):
    cfg = load_c3_config("explainers.yaml")["methods"]
    a = explain("kernel_shap", adapter.view("s03_lo_e00", 2, 5), 5, TINY["kernel_shap"], cfg["kernel_shap"]).values
    b = explain("kernel_shap", adapter.view("s03_lo_e00", 2, 5), 5, TINY["kernel_shap"], cfg["kernel_shap"]).values
    np.testing.assert_allclose(a, b)


def test_save_load_roundtrip(results, tmp_path):
    from xaicomp.paths import RESULTS_DIR
    p = RESULTS_DIR / "_scratch" / "pytest" / "e.npz"
    results["lime"].save(p)
    e = Explanation.load(p)
    np.testing.assert_allclose(e.values, results["lime"].values)
    assert e.extras["surrogate_r2"] == results["lime"].extras["surrogate_r2"]


def test_conditional_power_check_small(adapter):
    from xaicomp.replacement.checks import conditional_power_check
    r = conditional_power_check(adapter.view("s02_hi_e10", 2, 0), n_masks=3)
    assert r["mean_abs_err_log10"] < 0.1                  # < 1 dB on average
