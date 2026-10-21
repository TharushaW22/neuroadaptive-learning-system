"""Steps 6-7: fidelity, stability, robustness, sparsity, agreement, cost — checked on a toy model with
known feature importance (tests/fixtures.py)."""

import numpy as np
import pytest
from fixtures import ToyView, toy_weights

from xaicomp.metrics.agreement import agreement, pairwise_agreement
from xaicomp.metrics.cost import cost_summary, ms_per_eval
from xaicomp.metrics.fidelity import auc, curve_masks, fidelity_for_instance, ks
from xaicomp.metrics.robustness import noise_for_window, robustness
from xaicomp.metrics.sparsity import gini, n_mass, sparsity
from xaicomp.metrics.stability import stability, topk_jaccard


def test_curve_masks_and_ks():
    assert ks(84, 5).tolist()[-2:] == [80, 84] and ks(10, 5).tolist() == [0, 5, 10]
    d, i = curve_masks(np.array([2, 0, 1]), 3, 1)
    assert d[0].all() and not d[-1].any() and not i[0].any() and i[-1].all()
    assert d[1].tolist() == [True, True, False] and i[1].tolist() == [False, False, True]
    assert auc(np.array([1.0, 1.0])) == 1.0 and auc(np.array([0.0, 1.0])) == 0.5


def test_random_order_falls_between_best_and_worst():
    w = toy_weights(16, 3)
    view = ToyView(w, bias=0.5)
    pred_is_target = view.fx() >= 0.5
    rows, curves = fidelity_for_instance(view, {"truth": w, "anti": -w}, pred_is_target, 2, 10,
                                         ["off_manifold"], 0)
    r = {x["key"]: x for x in rows}
    t = r["truth"]
    assert t["auc_del"] <= t["auc_del_random"] <= t["auc_del_lerf"]
    assert t["auc_ins"] >= t["auc_ins_random"] >= t["auc_ins_lerf"]
    assert t["fidelity"] > 0 > r["anti"]["fidelity"]
    assert len(curves["off_manifold"]["truth"]["deletion"]) == len(ks(16, 2))


def test_stability_and_jaccard():
    a = np.arange(20.0)
    assert stability([a, a, a])["stability"] == pytest.approx(1.0)
    assert stability([a, -a])["stability"] == pytest.approx(-1.0)
    assert topk_jaccard(a, a, 5) == 1.0
    assert np.isnan(stability([a])["stability"])


def test_robustness_and_noise_scale():
    a = np.arange(10.0)
    assert robustness(a, a + 1e-9)["robustness"] == pytest.approx(1.0)
    n = noise_for_window(np.array([1.0, 10.0]), (2, 20000), 0.1, 0)
    np.testing.assert_allclose(n.std(axis=1), [0.1, 1.0], rtol=0.05)


def test_sparsity():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    one_hot = np.zeros(10); one_hot[3] = -5
    assert gini(one_hot) == pytest.approx(0.9)
    assert n_mass(one_hot) == 1 and n_mass(np.ones(10)) == 8
    assert set(sparsity(one_hot)) == {"sparsity", "n_features_80"}


def test_agreement():
    a = np.array([5.0, -4, 3, 0.1, 0.2, 0.0])
    r = agreement(a, a, k=3)
    assert r["rank_corr"] == pytest.approx(1) and r["feature_agree"] == 1 and r["sign_agree"] == 1
    r2 = agreement(a, -a, k=3)
    assert r2["feature_agree"] == 1 and r2["sign_agree"] == 0
    assert len(pairwise_agreement({"x": a, "y": a, "z": -a}, 3)) == 3


def test_cost():
    assert cost_summary([1.0, 3.0], [10, 30]) == {"cost_seconds": 2.0, "cost_evals": 20.0}
    assert ms_per_eval([1.0], [100]) == pytest.approx(10.0)
