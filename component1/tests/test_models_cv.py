"""Tests for the deep models, the masked multi-task loss, calibration, metrics and the CV leakage guard."""

import numpy as np
import pandas as pd
import pytest
import torch

from eegengine.evaluation.ablation import holm, paired_wilcoxon
from eegengine.evaluation.metrics import auroc, classification_metrics, expected_calibration_error
from eegengine.models.classical import make_pipeline
from eegengine.models.deep.eegnet import EEGNet
from eegengine.models.deep.gat import GATLayer
from eegengine.models.deep.hybrid import HybridModel, masked_multitask_loss
from eegengine.training.calibration import fit_temperature, softmax
from eegengine.training.cv import (assert_no_subject_overlap, inner_validation_split, kfold_splits, loso_splits,
                                   subject_fold_assignment)
from eegengine.training.sequences import recording_groups, strided_items

torch.manual_seed(0)


# ----------------------------------------------------------------- CV leakage guard
def test_loso_never_mixes_subjects():
    groups = np.repeat(np.arange(1, 11), 5)
    folds = list(loso_splits(groups))
    assert len(folds) == 10
    for s, tr, te in folds:
        assert set(groups[te]) == {s} and s not in set(groups[tr])


def test_kfold_assignment_fixed_and_disjoint():
    subs = list(range(1, 49))
    a1, a2 = subject_fold_assignment(subs, 5, 42), subject_fold_assignment(subs, 5, 42)
    assert a1 == a2                                         # deterministic -> same folds for every model
    assert sorted(set(a1.values())) == [0, 1, 2, 3, 4]
    groups = np.repeat(subs, 3)
    seen = []
    for _, tr, te in kfold_splits(groups, a1):
        assert not set(groups[tr]) & set(groups[te])
        seen += list(np.unique(groups[te]))
    assert sorted(seen) == subs                             # every subject tested exactly once


def test_guard_raises_on_overlap():
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_subject_overlap(np.array([1, 2, 3]), np.array([3, 4]))


def test_inner_validation_split_by_subject():
    groups = np.repeat(np.arange(20), 4)
    fit, val = inner_validation_split(groups, 0.15, 0)
    assert not set(groups[fit]) & set(groups[val]) and len(np.unique(groups[val])) == 3


def test_scaler_is_fit_inside_pipeline():
    """The StandardScaler is a pipeline step, so it only ever sees the training fold."""
    pipe = make_pipeline("lr")
    assert pipe.steps[0][0] == "scale"
    Xtr, Xte = np.random.default_rng(0).normal(0, 1, (50, 3)), np.full((5, 3), 100.0)
    pipe.fit(Xtr, np.r_[np.zeros(25), np.ones(25)])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, Xtr.mean(0))   # not influenced by Xte
    pipe.predict(Xte)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, Xtr.mean(0))


# ----------------------------------------------------------------- sequences
def test_sequences_stay_within_recording():
    meta = pd.DataFrame({"subject_id": [1] * 10 + [2] * 10, "condition": ["lo"] * 10 + ["hi"] * 10,
                         "epoch_index": list(range(10)) * 2})
    items = strided_items(recording_groups(meta, np.arange(20)), 4, 2)
    assert items.shape == (8, 4)
    for it in items:
        assert meta.loc[it, "subject_id"].nunique() == 1 and np.all(np.diff(it) == 1)


# ----------------------------------------------------------------- deep models
def test_eegnet_shapes():
    out = EEGNet(2)(torch.randn(3, 14, 512))
    assert out["workload"].shape == (3, 2)


@pytest.mark.parametrize("branches", [("cnn",), ("cnn", "lstm"), ("cnn", "lstm", "transformer"),
                                      ("cnn", "lstm", "gnn"), ("cnn", "lstm", "transformer", "gnn"), ("gnn",)])
def test_hybrid_branches(branches):
    m = HybridModel(3, branches=branches)
    out = m(torch.randn(2, 8, 14, 512), torch.randn(2, 8, 14, 5), torch.rand(2, 8, 14, 14))
    assert out["workload"].shape == (2, 3)
    assert set(out) == {"workload", "attention", "fatigue", "readiness"}


def test_hybrid_accepts_shorter_sequences():
    m = HybridModel(2).eval()
    out = m(torch.randn(1, 3, 14, 512), torch.randn(1, 3, 14, 5), torch.rand(1, 3, 14, 14))
    assert out["workload"].shape == (1, 2)


def test_masked_loss_ignores_unlabelled_heads():
    m = HybridModel(2, branches=("gnn",))
    out = m(torch.randn(4, 2, 14, 512), torch.randn(4, 2, 14, 5), torch.rand(4, 2, 14, 14))
    y = torch.tensor([0, 1, 0, 1])
    only_w = masked_multitask_loss(out, {"workload": y}, {"workload": torch.ones(4, dtype=torch.bool)})
    with_masked = masked_multitask_loss(out, {"workload": y, "attention": y},
                                        {"workload": torch.ones(4, dtype=torch.bool), "attention": torch.zeros(4, dtype=torch.bool)})
    assert torch.isclose(only_w, with_masked)
    with_masked.backward()
    assert m.heads["attention"].weight.grad is None or torch.all(m.heads["attention"].weight.grad == 0)


def test_gat_respects_adjacency():
    """With no edges (only self-loops), each node's output depends on its own features only."""
    layer = GATLayer(5, 4, heads=2).eval()
    x = torch.randn(1, 14, 5)
    adj = torch.zeros(1, 14, 14)
    y1 = layer(x, adj)
    x2 = x.clone()
    x2[0, 3] += 10.0
    y2 = layer(x2, adj)
    changed = (y1 - y2).abs().sum(-1)[0] > 1e-6
    assert changed.tolist() == [i == 3 for i in range(14)]


# ----------------------------------------------------------------- calibration + metrics
def test_temperature_scaling_reduces_nll_on_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    logits = np.where(rng.random(2000) < 0.7, y, 1 - y)[:, None] * np.array([[-1, 1]]) * 6.0  # 70 % right, very confident
    logits = logits + np.array([3.0, -3.0])
    T = fit_temperature(logits, y)
    assert T > 1.5
    nll = lambda t: -np.log(softmax(logits, t)[np.arange(2000), y] + 1e-12).mean()   # noqa: E731
    assert nll(T) < nll(1.0)
    e_before, _ = expected_calibration_error(y, softmax(logits))
    e_after, _ = expected_calibration_error(y, softmax(logits, T))
    assert e_after < e_before


def test_metrics_basics():
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]])
    m = classification_metrics(y, p, 2)
    assert m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0 and m["auroc"] == 1.0
    assert np.isnan(auroc(np.array([1, 1]), p[:2], 2))


def test_wilcoxon_and_holm():
    a = {"experiment_id": "a", "per_subject": {str(i): 0.8 + 0.01 * i for i in range(12)}, "per_subject_metric": "macro_f1"}
    b = {"experiment_id": "b", "per_subject": {str(i): 0.7 for i in range(12)}}
    r = paired_wilcoxon(a, b)
    assert r["n_a_better"] == 12 and r["p_value"] < 0.01
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
