"""Step 3 (real data): the EEG adapter reproduces C1's predictions and is differentiable."""

import numpy as np
import torch


def test_windows_match_c1_test_sequences(adapter):
    w = adapter.instances()
    assert len(w) == 96 * 34                          # C1: 3264 T1 sequences (stride 2)
    assert adapter.space.n_features == 84


def test_predict_matches_c1_deep_predictor(adapter):
    from eegengine.inference.predictors import load_predictor
    from xaicomp.data.eeg_data import load_data

    if adapter.bank.source != "deployed":
        return                                        # parity is defined against the deployed checkpoint
    d = load_data()
    p = load_predictor("T1_A8_cnn_lstm_trans")
    W = adapter._windows
    for iid in W.index[::400]:
        rows = W.loc[iid, "epoch_rows"]
        c1 = p.predict([{"epoch": d.X[r], "bandpower": d.bandpower[r], "adj": d.adj[r]} for r in rows])
        c3 = adapter.predict_segments(adapter.window_signal(iid)[None], int(W.loc[iid, "fold"]))[0]
        assert np.abs(c1 - c3).max() <= 1e-5


def test_components_reconstruct_window_and_residual_small(adapter):
    from xaicomp.representation.eeg_bands import residual_fraction
    iid = adapter._windows.index[100]
    x, r = adapter.window_components(iid)
    seg = adapter.window_signal(iid)
    np.testing.assert_allclose(x.sum(1) + r, seg, atol=1e-3)
    assert residual_fraction(seg, r) < 0.01


def test_torch_module_matches_numpy_path_and_gradients_finite(adapter):
    v = adapter.view(adapter._windows.index[7], 0, 0)
    module, x, args = v.torch_parts()
    x = x.clone().requires_grad_(True)
    out = module(x, *args)
    out.sum().backward()
    assert abs(out.item() - v.fx()) < 1e-5
    assert torch.isfinite(x.grad).all()


def test_value_function_edges(adapter):
    v = adapter.view(adapter._windows.index[3], 2, 0)
    full = v.value(np.ones((1, 84), bool), "donor")[0]
    assert abs(full - v.fx()) < 1e-5                   # every component present = the instance itself
    assert np.isfinite(v.value(np.zeros((2, 84), bool), "conditional")).all()
    assert np.isfinite(v.value(np.zeros((1, 84), bool), "empirical")).all()


def test_noisy_view_adds_noise_exactly(adapter):
    iid = adapter._windows.index[11]
    v0 = adapter.view(iid, 0, 0)
    v1 = adapter.view(iid, 0, 0, noise_seed=0, noise_scale=0.1)
    diff = v1.segment() - v0.segment()
    lm = adapter.model_for(v0.fold)
    ratio = diff.std(axis=1) / (0.1 * lm.channel_scale)
    assert np.all((ratio > 0.8) & (ratio < 1.2))
