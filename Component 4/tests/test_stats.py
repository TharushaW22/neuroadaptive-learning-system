import numpy as np
from evaluation.stats import paired_test


def test_identical_means_no_significance():
    a = np.random.default_rng(0).normal(0.5, 0.1, 200)
    b = a.copy()
    r = paired_test(a, b)
    assert r["p"] > 0.5 or np.isnan(r["p"])