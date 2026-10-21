"""Step 2 / D1: fold retraining with C1's unchanged run_fold (one fold, 1 epoch, TEST folder only)."""

import pytest

from xaicomp.paths import resolve_c3


@pytest.mark.slow
def test_one_fold_tiny_budget_trains_saves_and_loads(data_ready):
    from xaicomp.data.eeg_data import load_data
    from xaicomp.models.fold_training import train_fold_models
    from xaicomp.models.loader import load_fold

    out = "artifacts/fold_models_test/pytest_T1_A8"
    s = train_fold_models(load_data(), "workload_binary", "A8_cnn_lstm_trans", out, folds=[0], max_epochs=1, threads=2,
                          log=lambda *_: None)
    assert s["tiny_budget_test"] is True and 0 in s["folds"]
    assert (resolve_c3(out) / "fold0.pt").exists() and (resolve_c3(out) / "fold0_predictions.csv").exists()
    lm = load_fold(out, 0)
    assert lm.source == "fold" and lm.in_sample is False and lm.fold == 0
    assert sorted(lm.meta["test_subjects"]) == sorted([2, 11, 15, 18, 23, 26, 28, 38, 39, 48])
    assert s["c1_reference"]["pooled"]["macro_f1"] == pytest.approx(0.774, abs=1e-3)


def test_model_bank_falls_back_to_in_sample_deployed():
    from xaicomp.config import load_context
    from xaicomp.models.loader import ModelBank, fold_models_complete
    ctx = load_context("eeg_T1_A8")
    bank = ModelBank(ctx)
    if fold_models_complete(ctx["fold_models_dir"]):
        assert bank.source == "fold" and not bank.in_sample
    else:
        assert bank.source == "deployed" and bank.in_sample


def test_randomised_model_differs_and_keeps_scalers():
    import torch
    from xaicomp.models.loader import load_deployed, randomise
    lm = load_deployed("T1_A8_cnn_lstm_trans")
    rm = randomise(lm, 1234)
    a = dict(lm.model.named_parameters())["heads.workload.weight"]
    b = dict(rm.model.named_parameters())["heads.workload.weight"]
    assert not torch.allclose(a, b)
    assert (rm.channel_scale == lm.channel_scale).all() and rm.weights == "randomised"
    assert dict(load_deployed("T1_A8_cnn_lstm_trans").model.named_parameters())["heads.workload.weight"].equal(a)
