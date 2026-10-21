"""Step 9: ICC, degradation, redundancy, randomisation control."""

import numpy as np
import pandas as pd
import pytest

from xaicomp.reliability.degradation import attribution_only_degradation, degrade, summarise
from xaicomp.reliability.randomisation import method_sanity, metric_collapse
from xaicomp.reliability.redundancy import redundancy_matrix, redundant_groups
from xaicomp.reliability.test_retest import icc_2_1, icc_label


def test_icc_known_cases():
    rng = np.random.default_rng(0)
    t = rng.normal(size=50)
    assert icc_2_1(np.c_[t, t, t]) == pytest.approx(1.0)
    assert icc_2_1(np.c_[t + rng.normal(0, 0.1, 50), t + rng.normal(0, 0.1, 50)]) > 0.95
    assert abs(icc_2_1(rng.normal(size=(200, 2)))) < 0.2
    assert np.isnan(icc_2_1(np.ones((2, 2))))
    assert icc_label(0.8) == "good" and icc_label(float("nan")) == "not assessable"


def test_degrade_endpoints():
    rng = np.random.default_rng(0)
    a = np.arange(1.0, 11.0)
    np.testing.assert_allclose(degrade(a, 0.0, rng), a / np.linalg.norm(a))
    assert abs(np.corrcoef(degrade(a, 1.0, rng), a)[0, 1]) < 0.9


def test_attribution_metrics_fall_monotonically_with_degradation():
    rng = np.random.default_rng(1)
    base = {i: np.exp(rng.normal(0, 2, 84)) * rng.choice([-1, 1], 84) for i in range(8)}
    ex = {(f"i{i}", "m", s): base[i] + rng.normal(0, 0.05, 84) for i in range(8) for s in (0, 1, 2)}
    noisy = {(f"i{i}", "m"): base[i] + rng.normal(0, 0.05, 84) for i in range(8)}
    s = summarise(attribution_only_degradation(ex, noisy, [0, 1, 2]), 0.8).set_index("criterion")
    assert s.loc["sparsity", "monotonic_decrease"] and s.loc["stability", "monotonic_decrease"]
    assert s.loc["robustness", "monotonic_decrease"]


def test_redundancy_groups():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    df = pd.DataFrame({"a": x, "b": x * 2 + 0.001 * rng.normal(size=100), "c": rng.normal(size=100), "d": np.ones(100)})
    R = redundancy_matrix(df, ["a", "b", "c", "d"])
    assert "d" not in R.columns
    g = redundant_groups(R, 0.9)
    assert sorted(map(sorted, g)) == [["a", "b"], ["c"]]


def test_randomisation_checks():
    rng = np.random.default_rng(0)
    tr = {(f"i{i}", "good", 0): rng.normal(size=30) for i in range(5)}
    tr.update({(f"i{i}", "bad", 0): np.arange(30.0) for i in range(5)})
    ctrl = {(f"i{i}", "good", 0): rng.normal(size=30) for i in range(5)}
    ctrl.update({(f"i{i}", "bad", 0): np.arange(30.0) + 0.01 * rng.normal(size=30) for i in range(5)})
    s = method_sanity(tr, ctrl).set_index("method")
    assert s.loc["good", "passes_sanity_check"] and not s.loc["bad", "passes_sanity_check"]
    a = pd.DataFrame({"instance_id": ["x", "y"], "fidelity_on": [0.2, 0.3]})
    b = pd.DataFrame({"instance_id": ["x", "y"], "fidelity_on": [0.01, 0.0]})
    assert metric_collapse(a, b, ["fidelity_on"]).iloc[0]["collapses"]
