import numpy as np
import pytest
import torch

from src.data.feature_schema import FEATURE_COLUMNS
from src.models.baseline import aggregate_windows, aggregated_feature_names, baseline_predict_proba, build_baseline
from src.models.labels import LabelSchemeError, get_label_scheme
from src.models.temporal_model import TemporalClassifier

F = len(FEATURE_COLUMNS)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_temporal_output_shape(cell, bidirectional):
    m = TemporalClassifier(F, 4, hidden_size=16, num_layers=2, cell=cell, bidirectional=bidirectional)
    x = torch.randn(5, 20, F)
    mask = torch.ones(5, 20)
    mask[0, 12:] = 0
    assert m(x, mask).shape == (5, 4)


def test_temporal_padding_does_not_change_prediction():
    torch.manual_seed(0)
    m = TemporalClassifier(F, 3, hidden_size=8).eval()
    x = torch.randn(1, 10, F)
    mask = torch.ones(1, 10)
    mask[0, 6:] = 0
    x2 = x.clone()
    x2[0, 6:] = 999.0                    # garbage in padded positions
    torch.testing.assert_close(m(x, mask), m(x2, mask))


def test_temporal_rejects_wrong_shapes():
    m = TemporalClassifier(F, 2)
    with pytest.raises(ValueError):
        m(torch.randn(3, F))
    with pytest.raises(ValueError):
        m(torch.randn(2, 5, F + 1))
    with pytest.raises(ValueError):
        TemporalClassifier(F, 2, cell="transformer")


def test_baseline_aggregation_and_proba(cfg):
    X = np.random.randn(8, 20, F).astype(np.float32)
    mask = np.ones((8, 20), np.float32)
    mask[:, 15:] = 0
    agg = aggregate_windows(X, mask)
    assert agg.shape == (8, 4 * F) and len(aggregated_feature_names(FEATURE_COLUMNS)) == 4 * F
    np.testing.assert_allclose(agg[:, :F], X[:, :15].mean(1), rtol=1e-5, atol=1e-5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    model = build_baseline(cfg).fit(agg, y)
    p = baseline_predict_proba(model, X, mask, n_classes=4)
    assert p.shape == (8, 4) and np.allclose(p.sum(1), 1) and (p[:, 2:] == 0).all()


def test_label_schemes(cfg):
    s = get_label_scheme(cfg)
    assert s.name == "daisee_original" and s.class_names == [f"engagement_{i}" for i in range(4)]
    cfg["labels"]["scheme"] = "daisee_binary"
    b = get_label_scheme(cfg)
    assert [b.transform(i) for i in range(4)] == [0, 0, 1, 1]
    cfg["labels"]["scheme"] = "operational"
    with pytest.raises(LabelSchemeError):          # no documented protocol -> refused
        get_label_scheme(cfg)
    cfg["labels"]["operational_mapping"] = {0: "DISTRACTED", 1: "REDUCED_ATTENTION", 2: "FOCUSED", 3: "FOCUSED"}
    cfg["labels"]["operational_mapping_document"] = "docs/does_not_exist.md"
    with pytest.raises(LabelSchemeError):
        get_label_scheme(cfg)
