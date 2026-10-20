"""Steps 10-11: corrected agreement, mixed model, subject-level statistics, Bayesian ROPE."""

import numpy as np
import pandas as pd
import pytest

from xaicomp.disagreement.corrected_agreement import corrected_agreement
from xaicomp.disagreement.mixed_effects import fit_mixed, value_function_effect
from xaicomp.stats.aggregate import subject_level, wide
from xaicomp.stats.bayesian import pairwise_bayes
from xaicomp.stats.frequentist import (bootstrap_matrix, cluster_bootstrap, friedman, kendalls_w, pairwise_wilcoxon,
                                       rank_biserial)


def test_corrected_agreement_formula_and_flags():
    ag = pd.DataFrame({"instance_id": ["a", "b"] * 2, "seed": [0, 0, 1, 1], "method_a": ["x"] * 4,
                       "method_b": ["y"] * 4, "rank_corr": [0.4, 0.4, 0.4, 0.4]})
    st = pd.DataFrame({"instance_id": ["a", "b", "a", "b"], "method": ["x", "x", "y", "y"], "stability": [0.5, 0.5, 0.8, 0.8]})
    r = corrected_agreement(ag, st).iloc[0]
    assert r["corrected_agreement"] == pytest.approx(0.4 / np.sqrt(0.4))
    assert not r["corrected_gt_1"]
    st2 = st.assign(stability=0.1)
    assert corrected_agreement(ag, st2).iloc[0]["corrected_gt_1"]
    ag3 = ag.assign(method_b="ig_zero")
    st3 = pd.concat([st, st.assign(method="ig_zero", stability=1.0)])
    assert np.isnan(corrected_agreement(ag3, st3).iloc[0]["corrected_agreement"])


def _toy_disagreement(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(12):
        u = rng.normal(0, 0.05)
        for i in range(6):
            conf, r2, bp, bad = rng.uniform(0.5, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 0.3)
            for pair in ("kernel_shap vs lime", "conditional_shap vs kernel_shap"):
                a, b = pair.split(" vs ")
                rows.append({"instance_id": f"{s}_{i}", "subject_id": s, "pair": pair, "method_a": a, "method_b": b,
                             "confidence": conf, "lime_r2": r2, "local_bp_corr": bp, "frac_bad": bad,
                             "disagreement": 0.5 + 0.3 * bp - 0.2 * r2 + u + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_mixed_model_recovers_signs():
    res = fit_mixed(_toy_disagreement())
    assert "coefficients" in res
    c = {r["term"]: r["coef"] for r in res["coefficients"]}
    assert c["z_local_bp_corr"] > 0 and c["z_lime_r2"] < 0
    vfe = value_function_effect(_toy_disagreement())
    assert vfe["available"] and vfe["spearman_with_local_bp_corr"] > 0


def test_mixed_model_handles_too_little_data():
    assert fit_mixed(_toy_disagreement().head(4))["converged"] is False


def _toy_im():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        for i in range(3):
            for m, shift in (("a", 0.3), ("b", 0.0), ("c", 0.29)):
                rows.append({"instance_id": f"{s}_{i}", "subject_id": s, "method": m,
                             "fidelity_on": shift + rng.normal(0, 0.02), "cost_seconds": 1 + shift})
    return pd.DataFrame(rows)


def test_subject_level_and_frequentist_tests():
    subj = subject_level(_toy_im(), ["fidelity_on", "cost_seconds"], ["a", "b", "c"])
    assert len(subj) == 30
    W = wide(subj, "fidelity_on", ["a", "b", "c"])
    fr = friedman(W)
    assert fr["p_value"] < 0.01 and 0 < fr["kendalls_w"] <= 1
    wx = pairwise_wilcoxon(W, "+")
    ab = wx[(wx.method_a == "a") & (wx.method_b == "b")].iloc[0]
    assert ab["rank_biserial_a_better"] == pytest.approx(1.0) and ab["significant"]
    wc = pairwise_wilcoxon(wide(subj, "cost_seconds", ["a", "b"]), "-")
    assert wc.iloc[0]["rank_biserial_a_better"] < 0                   # a is more expensive -> worse
    bm, bd = cluster_bootstrap(W, 500, 0.95, 0)
    assert ((bm["ci_low"] <= bm["mean"]) & (bm["mean"] <= bm["ci_high"])).all()
    assert rank_biserial(np.array([1.0, 2, -0.5])) > 0
    assert kendalls_w(np.array([[1, 2, 3], [1, 2, 3.0]])) == pytest.approx(1.0)
    B = bootstrap_matrix(subj, ["fidelity_on"], ["a", "b", "c"], 50, 0)
    assert B.shape == (50, 3, 1)


def test_bayes_rope_direction_and_equivalence():
    subj = subject_level(_toy_im(), ["fidelity_on"], ["a", "b", "c"])
    by = pairwise_bayes(wide(subj, "fidelity_on", ["a", "b", "c"]), "+", 0.05, 5000, 0).set_index(["method_a", "method_b"])
    assert by.loc[("a", "b"), "p_a_better"] > 0.95
    assert by.loc[("a", "c"), "p_rope"] > 0.95 and by.loc[("a", "c"), "verdict"] == "practically the same"
