# 6. Deep models

All deep code is plain PyTorch (`src/eegengine/models/deep/`), configured by
`configs/experiments/deep.yaml`, trained by `src/eegengine/training/trainer.py` and run per experiment
by `src/eegengine/training/deep_runner.py:run_deep` from `scripts/run_deep.py`.

## 6.1 Input representations
| Input | Shape | Built by | Scaling (fit on the fit subjects of the fold only) |
|---|---|---|---|
| raw epochs | `(L, 14, 512)` per sample (L epochs × 14 channels × 4 s at 128 Hz) | `data/processed/epochs_*.npz` | divided by a per-channel robust scale = median over training epochs of each epoch's channel std, clipped to ±20 (`trainer.py:Scalers`, `DeepInputs.batch`) |
| node features (GAT) | `(L, 14, 5)` | log10 band power per channel (delta…gamma) from `features_*_arrays.npz` | z-scored per channel and band with training-fold mean/std |
| edges (GAT) | `(L, 14, 14)` | PLV averaged over the 4 bands (`trainer.py:edge_matrix`, `input.edge_source: plv`, `edge_band: mean`) | none (PLV is already 0–1) |
| sequences | `L = 8` consecutive epochs of the same recording = 18 s | `training/sequences.py:strided_items` / `sample_items` | — |

Sequences never cross recordings, so each has a single label. Test windows: stride 2 epochs (4 s); 3264
test sequences for T1 (`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `n_sequences`). EEGNet is a
single-epoch model: it is trained on epochs (`L = 1`) and its calibrated probabilities are averaged over
the same 8-epoch test windows, so all models are scored on identical windows.

## 6.2 EEGNet (baseline)
**WHAT.** A compact CNN for EEG (Lawhern et al., 2018, *J. Neural Eng.* — cited in
`src/eegengine/models/deep/eegnet.py`). Layer by layer for one epoch `(1, 14, 512)`
(shapes verified by instantiating the model):

| # | Layer | Parameters | Output shape |
|---|---|---|---|
| 1 | temporal Conv2d, F1 = 8 filters, kernel (1, 32), padding 16, no bias | learns 8 band-pass filters | (8, 14, 513) |
| 2 | BatchNorm2d | | (8, 14, 513) |
| 3 | depthwise spatial Conv2d (14, 1), D = 2 per filter, groups = 8 | learns 2 spatial filters per band (like CSP) | (16, 1, 513) |
| 4 | BatchNorm2d → ELU → AvgPool (1, 4) → Dropout 0.25 | | (16, 1, 128) |
| 5 | separable conv = depthwise temporal Conv2d (1, 16), groups 16 + pointwise Conv2d 1×1 → F2 = 16 | combines feature maps over time | (16, 1, 129) |
| 6 | BatchNorm2d → ELU → AvgPool (1, 8) → Dropout 0.25 | | (16, 1, 16) |
| 7 | flatten → Linear(256, n_classes) | | (n_classes) |

Parameters: 1586 [`results/deep_T1_eegnet_kfold5/metrics.json` → `n_params`].
**WHY.** It is the standard, widely replicated EEG deep-learning baseline: few parameters (suits small
datasets), physiologically interpretable filters, and it works across paradigms.
**DEVIATION FROM PROPOSAL.** The original EEGNet uses a temporal kernel of fs/2 = 64 samples. The kernel was set
to 32 (0.25 s) because a 64-sample kernel hit a very slow CPU convolution path (≈ 26 s per training step vs
≈ 1.3 s measured during development; D17).

## 6.3 CNN-LSTM
`src/eegengine/models/deep/cnn_lstm.py:build_cnn_lstm` = the hybrid with only the CNN and LSTM branches (A7),
so baseline and ablation share one implementation:
1. **CNN encoder** per epoch = EEGNet blocks 1–6 + `Linear(256, 64)` → embedding e_t ∈ ℝ⁶⁴ (`eegnet.py:EEGNetEncoder`, 17520 parameters).
2. **LSTM** (1 layer, hidden 64, batch-first) over e_1…e_8; Z_lstm = last hidden state (33280 parameters).
3. Z_cnn = mean of e_t; fusion `[Z_cnn; Z_lstm]` → MLP → heads (§6.4).
Parameters: 59576 [`results/deep_T1_A7_cnn_lstm_kfold5/metrics.json` → `n_params`].

## 6.4 Hybrid CNN + LSTM + Transformer + GAT (multi-task)
`src/eegengine/models/deep/hybrid.py:HybridModel`. Input `x (B, L, 14, 512)`, `node (B, L, 14, 5)`,
`adj (B, L, 14, 14)`. Every branch can be switched off (`branches` in the config).

**CNN branch** — the shared `EEGNetEncoder` applied to each of the L epochs → `e (B, L, 64)`;
`Z_cnn = mean_t e_t` (64-d).

**LSTM branch** — `nn.LSTM(64, 64)` over the embedding sequence; `Z_lstm = h_L` (64-d).

**Transformer branch** — sinusoidal positional encoding
`PE(pos, 2i) = sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos / 10000^{2i/d})`, d = 64, added to e;
2 `TransformerEncoderLayer`s (4 heads, feed-forward 128, dropout 0.1, batch-first);
`Z_trans = mean over positions` (64-d). Self-attention lets every epoch attend to every other epoch of the window.
Branch parameters: 66944.

**GAT branch** (`src/eegengine/models/deep/gat.py`, plain PyTorch, dense 14 × 14 adjacency). For node i with
features h_i and neighbours j:
```
e_ij  = LeakyReLU_0.2( a_srcᵀ W h_i + a_dstᵀ W h_j ) + log(A_ij)      (A = PLV, self-loops = 1)
α_ij  = softmax_j(e_ij)   over edges with A_ij > 0 (others masked to −∞), dropout 0.1
h_i'  = Σ_j α_ij W h_j     per head
```
Layer 1: 5 → 16 per head × 4 heads, concatenated (64) + ELU; layer 2: 64 → 32, 4 heads averaged + ELU
(`GATEncoder`, 448 + 8448 parameters). Node embeddings are mean-pooled over the 14 nodes, then over the
L epochs → `Z_gnn` (32-d). The `log(A_ij)` term is a prior: strongly phase-locked pairs get more attention,
while the learned part decides which connections matter (Veličković et al., 2018, cited in `gat.py`).
Tested: with no edges each node's output depends only on itself (`tests/test_models_cv.py:test_gat_respects_adjacency`).

**Fusion** — concatenate the enabled Z's (A10: 64 + 64 + 64 + 32 = 224) → `Linear(224, 64) → ELU →
Dropout(0.3)` (14400 parameters).

**Four task heads** — `workload: Linear(64, n_classes)`, `attention / fatigue / readiness: Linear(64, 2)` each (520 parameters in total).

**Masked multi-task loss** (`hybrid.py:masked_multitask_loss`):
`L = Σ_h w_h · CE_h( logits_h[m_h], y_h[m_h]; class weights c_h )`, summed over the heads h that have at least one
labelled sample in the batch (mask m_h). A head with no labels contributes exactly 0 and receives no gradient
(tested: `test_masked_loss_ignores_unlabelled_heads`). On STEW only the workload mask is ever true, so
attention/fatigue/readiness heads are present but **inert**, ready for the team-collected labels.

**Parameter count per model:**
| Model | Branches | Params |
|---|---|---|
| EEGNet | — | 1586 [`results/deep_T1_eegnet_kfold5/metrics.json` → `n_params`] |
| A6 CNN | cnn | 22200 [`results/deep_T1_A6_cnn_kfold5/metrics.json` → `n_params`] |
| A7 CNN-LSTM | cnn, lstm | 59576 [`results/deep_T1_A7_cnn_lstm_kfold5/metrics.json` → `n_params`] |
| A8 CNN-LSTM-Transformer | cnn, lstm, transformer | 130616 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `n_params`] |
| A9 CNN-LSTM-GAT | cnn, lstm, gnn | 70520 [`results/deep_T1_A9_cnn_lstm_gnn_kfold5/metrics.json` → `n_params`] |
| A10 full hybrid | all four | 141560 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `n_params`] |

All models accept any sequence length L ≥ 1, so the live engine can predict before 8 epochs exist
(`test_hybrid_accepts_shorter_sequences`).

## 6.5 Training
`trainer.py:train_model` / `run_fold`, settings from `configs/experiments/deep.yaml` → `train` / `augment`:
| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 0.001 [`configs/experiments/deep.yaml` → `train.lr`], weight decay 0.01 [`configs/experiments/deep.yaml` → `train.weight_decay`], gradient-norm clip 1.0 |
| Batch | 16 [`configs/experiments/deep.yaml` → `train.batch_size`] sequences (EEGNet: 4× = 64 epochs) |
| Items per epoch | 8 [`configs/experiments/deep.yaml` → `train.epoch_items_per_recording`] fresh random sequence starts per training recording each epoch (EEGNet 32 epochs) — fixed cost per epoch, more variety than a fixed list |
| Epochs | at most 25 [`configs/experiments/deep.yaml` → `train.max_epochs`], early stopping patience 5 [`configs/experiments/deep.yaml` → `train.patience`] on validation loss; best weights restored |
| Validation | 15 % of the training subjects (`val_frac` 0.15 [`configs/experiments/deep.yaml` → `train.val_frac`]), subject-wise, re-drawn until it contains every class; validation stride 4 epochs |
| Class weights | inverse class frequency over the fit recordings, normalised to mean 1 |
| Augmentation | Gaussian noise σ = 0.1 [`configs/experiments/deep.yaml` → `augment.noise_std`] (in channel-scale units), channel dropout p = 0.1 [`configs/experiments/deep.yaml` → `augment.channel_dropout`] (same channel zeroed in all epochs of a sequence), circular time shift up to ±16 [`configs/experiments/deep.yaml` → `augment.max_shift`] samples (125 ms) |
| Seeds | `seed 42 + fold` for Python, NumPy and PyTorch (`trainer.py:set_torch_seed`) |
| Hardware | CPU only (Intel i5-1135G7, 4 cores; D17), 4 threads [`configs/experiments/deep.yaml` → `train.threads`] |

CPU time per experiment (train + evaluate, 5 folds) is in the table below (column "Train+eval s").
**Known issue — early stopping on a tiny validation set.** The validation split holds only ≈ 6 subjects. In
fold 1 of T1 *every* deep model had its best validation loss at epoch 0 and stopped at epoch 6; most T3
folds stop at epochs 0–2 (column "Best epoch per fold"). Early stopping (and temperature scaling) are
therefore noisy, which is one reason the deep models under-perform (§7.7).

## 6.6 Results and architecture ablation (A6–A10), 5 subject folds, 18-s windows
<!-- BEGIN TABLE:deep -->
| Task | Model | Params | Fold macro-F1 | Fold bal. acc | Pooled AUROC | Recording macro-F1 | ECE before → after | CPU ms / window (1 thread) | Best epoch per fold | Train+eval s | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | EEGNet | 1586 | 0.772 ± 0.099 | 0.773 ± 0.098 | 0.843 | 0.812 | 0.086 → 0.105 | 0.9 | [23, 0, 22, 8, 23] | 410 | `deep_T1_eegnet_kfold5` |
| T1 | A6 CNN | 22200 | 0.701 ± 0.138 | 0.709 ± 0.127 | 0.782 | 0.768 | 0.031 → 0.082 | 4.4 | [22, 0, 0, 3, 10] | 359 | `deep_T1_A6_cnn_kfold5` |
| T1 | A7 CNN-LSTM | 59576 | 0.730 ± 0.091 | 0.736 ± 0.085 | 0.807 | 0.771 | 0.093 → 0.084 | 9.8 | [23, 0, 14, 4, 10] | 667 | `deep_T1_A7_cnn_lstm_kfold5` |
| T1 | A8 CNN-LSTM-Transformer | 130616 | 0.773 ± 0.045 | 0.775 ± 0.045 | 0.835 | 0.833 | 0.074 → 0.053 | 11.3 | [9, 3, 18, 4, 14] | 805 | `deep_T1_A8_cnn_lstm_trans_kfold5` |
| T1 | A9 CNN-LSTM-GAT | 70520 | 0.751 ± 0.077 | 0.751 ± 0.077 | 0.827 | 0.812 | 0.088 → 0.068 | 6.5 | [5, 0, 23, 7, 17] | 842 | `deep_T1_A9_cnn_lstm_gnn_kfold5` |
| T1 | A10 full hybrid (CNN+LSTM+Transformer+GAT) | 141560 | 0.758 ± 0.103 | 0.765 ± 0.093 | 0.853 | 0.802 | 0.098 → 0.055 | 21.8 | [20, 0, 9, 15, 14] | 783 | `deep_T1_A10_hybrid_kfold5` |
| T3 | EEGNet | 1586 | 0.514 ± 0.076 | 0.536 ± 0.085 | 0.543 | 0.464 | 0.048 → 0.061 | 3.2 | [5, 21, 0, 0, 0] | 115 | `deep_T3_eegnet_kfold5` |
| T3 | A6 CNN | 22200 | 0.458 ± 0.076 | 0.468 ± 0.084 | 0.447 | 0.434 | 0.111 → 0.103 | 4.9 | [1, 14, 0, 2, 2] | 222 | `deep_T3_A6_cnn_kfold5` |
| T3 | A7 CNN-LSTM | 59576 | 0.433 ± 0.089 | 0.476 ± 0.067 | 0.485 | 0.440 | 0.125 → 0.134 | 6.3 | [1, 12, 0, 2, 6] | 1700 | `deep_T3_A7_cnn_lstm_kfold5` |
| T3 | A8 CNN-LSTM-Transformer | 130616 | 0.412 ± 0.097 | 0.500 ± 0.043 | 0.504 | 0.527 | 0.125 → 0.129 | 4.7 | [1, 14, 0, 2, 0] | 138 | `deep_T3_A8_cnn_lstm_trans_kfold5` |
| T3 | A9 CNN-LSTM-GAT | 70520 | 0.487 ± 0.058 | 0.493 ± 0.058 | 0.484 | 0.398 | 0.113 → 0.171 | 7.7 | [1, 16, 0, 1, 2] | 260 | `deep_T3_A9_cnn_lstm_gnn_kfold5` |
| T3 | A10 full hybrid (CNN+LSTM+Transformer+GAT) | 141560 | 0.389 ± 0.062 | 0.492 ± 0.023 | 0.501 | 0.466 | 0.138 → 0.133 | 5.3 | [1, 12, 0, 1, 1] | 138 | `deep_T3_A10_hybrid_kfold5` |
<!-- END TABLE:deep -->

Paired Wilcoxon tests (`scripts/deep_stats.py`; per-subject macro-F1 on T1, accuracy on T3):
<!-- BEGIN TABLE:stats_deep -->
| # | Group | A | B | Per-subject metric | n | mean A | mean B | A better / B better / ties | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | deep_ablation/T1 | deep_T1_A10_hybrid_kfold5 | deep_T1_A6_cnn_kfold5 | macro_f1 | 48 | 0.7267 | 0.6778 | 28/15/5 | 0.089 | 0.443 |
| 1 | deep_ablation/T1 | deep_T1_A10_hybrid_kfold5 | deep_T1_A7_cnn_lstm_kfold5 | macro_f1 | 48 | 0.7267 | 0.7022 | 30/16/2 | 0.188 | 0.752 |
| 2 | deep_ablation/T1 | deep_T1_A10_hybrid_kfold5 | deep_T1_A8_cnn_lstm_trans_kfold5 | macro_f1 | 48 | 0.7267 | 0.7569 | 24/20/4 | 0.930 | 1.000 |
| 3 | deep_ablation/T1 | deep_T1_A10_hybrid_kfold5 | deep_T1_A9_cnn_lstm_gnn_kfold5 | macro_f1 | 48 | 0.7267 | 0.7264 | 26/19/3 | 0.710 | 1.000 |
| 4 | deep_ablation/T1 | deep_T1_A10_hybrid_kfold5 | deep_T1_eegnet_kfold5 | macro_f1 | 48 | 0.7267 | 0.7443 | 21/22/5 | 0.664 | 1.000 |
| 5 | classical_vs_deep/T1 | deep_T1_A10_hybrid_kfold5 | cls_T1_xgb_all_none_kfold5[windows] | macro_f1 | 48 | 0.7267 | 0.9071 | 8/36/4 | 4.03e-06 | 4.83e-06 |
| 6 | classical_vs_deep/T1 | deep_T1_A8_cnn_lstm_trans_kfold5 | cls_T1_xgb_all_none_kfold5[windows] | macro_f1 | 48 | 0.7569 | 0.9071 | 6/38/4 | 2.42e-06 | 4.83e-06 |
| 7 | deep_ablation/T3 | deep_T3_A10_hybrid_kfold5 | deep_T3_A6_cnn_kfold5 | accuracy | 45 | 0.4837 | 0.4699 | 23/20/2 | 0.814 | 1.000 |
| 8 | deep_ablation/T3 | deep_T3_A10_hybrid_kfold5 | deep_T3_A7_cnn_lstm_kfold5 | accuracy | 45 | 0.4837 | 0.4876 | 19/18/8 | 0.898 | 1.000 |
| 9 | deep_ablation/T3 | deep_T3_A10_hybrid_kfold5 | deep_T3_A8_cnn_lstm_trans_kfold5 | accuracy | 45 | 0.4837 | 0.5190 | 18/19/8 | 0.763 | 1.000 |
| 10 | deep_ablation/T3 | deep_T3_A10_hybrid_kfold5 | deep_T3_A9_cnn_lstm_gnn_kfold5 | accuracy | 45 | 0.4837 | 0.5026 | 23/22/0 | 0.892 | 1.000 |
| 11 | deep_ablation/T3 | deep_T3_A10_hybrid_kfold5 | deep_T3_eegnet_kfold5 | accuracy | 45 | 0.4837 | 0.5444 | 21/22/2 | 0.432 | 1.000 |
| 12 | classical_vs_deep/T3 | deep_T3_A10_hybrid_kfold5 | cls_T3_xgb_all_none_kfold5[windows] | accuracy | 45 | 0.4837 | 0.5784 | 17/22/6 | 0.418 | 0.801 |
| 13 | classical_vs_deep/T3 | deep_T3_eegnet_kfold5 | cls_T3_xgb_all_none_kfold5[windows] | accuracy | 45 | 0.5444 | 0.5784 | 21/24/0 | 0.400 | 0.801 |

Source: `results/stats_deep.json` (row # = index in `tests`)
<!-- END TABLE:stats_deep -->

Findings (T1): the order A6 CNN 0.701 [`results/deep_T1_A6_cnn_kfold5/metrics.json` → `per_fold.macro_f1.mean`] →
A7 +LSTM 0.730 [`results/deep_T1_A7_cnn_lstm_kfold5/metrics.json` → `per_fold.macro_f1.mean`] →
A8 +Transformer 0.773 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `per_fold.macro_f1.mean`] →
A9 +GAT instead of Transformer 0.751 [`results/deep_T1_A9_cnn_lstm_gnn_kfold5/metrics.json` → `per_fold.macro_f1.mean`] →
A10 all 0.758 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `per_fold.macro_f1.mean`] suggests temporal context helps
and the GAT branch does not, but **none of the differences to A10 is significant** (smallest p = 0.089 [`results/stats_deep.json` → `tests.0.p_value`]).
EEGNet with 1.1 % of the hybrid's parameters is as good as the best sequence model
(0.772 [`results/deep_T1_eegnet_kfold5/metrics.json` → `per_fold.macro_f1.mean`]). On T3 all deep models are at chance.

## 6.7 Calibration — temperature scaling
**WHAT.** `p = softmax(z / T)` with a single scalar T > 0 (Guo et al., 2017, cited in
`src/eegengine/training/calibration.py`). T > 1 softens over-confident predictions; the argmax (and
therefore accuracy/F1) does not change. T is fit by minimising the negative log-likelihood on the
**inner validation subjects** (LBFGS on log T; `calibration.py:fit_temperature`), clipped to [0.05, 20];
if the validation set has a single class T = 1.
**ECE.** `ECE = Σ_b (n_b/n) · |acc_b − conf_b|` over 15 equal-width confidence bins
(`evaluation/metrics.py:expected_calibration_error`).
**PRODUCED.** ECE before → after is in the table above. Temperature scaling improved ECE for the hybrid on T1
(0.098 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `calibration.ece_before`] →
0.055 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `calibration.ece_after`]) and A8
(0.074 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `calibration.ece_before`] → 0.053 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `calibration.ece_after`])
but made EEGNet worse (0.086 [`results/deep_T1_eegnet_kfold5/metrics.json` → `calibration.ece_before`] →
0.105 [`results/deep_T1_eegnet_kfold5/metrics.json` → `calibration.ece_after`]) and A6 worse
(0.031 [`results/deep_T1_A6_cnn_kfold5/metrics.json` → `calibration.ece_before`] → 0.082 [`results/deep_T1_A6_cnn_kfold5/metrics.json` → `calibration.ece_after`]).
Reason: T is estimated from ≈ 6 validation subjects, whose confidence behaviour differs from the test
subjects'. Fitted temperatures per fold:
<!-- BEGIN TABLE:temps -->
| Task | Model | Temperature T per fold (bounds 0.05–20) | Source (calibration.temperatures) |
|---|---|---|---|
| T1 | EEGNet | 1.00, 7.80, 1.25, 1.73, 0.97 | `deep_T1_eegnet_kfold5` |
| T1 | A6 CNN | 1.58, 10.14, 2.40, 1.72, 0.79 | `deep_T1_A6_cnn_kfold5` |
| T1 | A7 CNN-LSTM | 1.17, 7.26, 1.32, 1.91, 0.71 | `deep_T1_A7_cnn_lstm_kfold5` |
| T1 | A8 CNN-LSTM-Transformer | 0.93, 2.22, 1.36, 1.91, 0.71 | `deep_T1_A8_cnn_lstm_trans_kfold5` |
| T1 | A9 CNN-LSTM-GAT | 1.17, 2.29, 1.64, 1.98, 1.05 | `deep_T1_A9_cnn_lstm_gnn_kfold5` |
| T1 | A10 full hybrid (CNN+LSTM+Transformer+GAT) | 1.39, 1.87, 1.75, 1.89, 1.02 | `deep_T1_A10_hybrid_kfold5` |
| T3 | EEGNet | 0.85, 0.70, 9.06, 0.53, 0.41 | `deep_T3_eegnet_kfold5` |
| T3 | A6 CNN | 0.66, 0.82, 20.00, 10.98, 1.31 | `deep_T3_A6_cnn_kfold5` |
| T3 | A7 CNN-LSTM | 0.74, 0.66, 2.47, 0.70, 1.28 | `deep_T3_A7_cnn_lstm_kfold5` |
| T3 | A8 CNN-LSTM-Transformer | 1.74, 1.22, 20.00, 0.59, 0.66 | `deep_T3_A8_cnn_lstm_trans_kfold5` |
| T3 | A9 CNN-LSTM-GAT | 20.00, 0.57, 20.00, 0.05, 1.32 | `deep_T3_A9_cnn_lstm_gnn_kfold5` |
| T3 | A10 full hybrid (CNN+LSTM+Transformer+GAT) | 1.80, 0.73, 1.99, 2.07, 0.52 | `deep_T3_A10_hybrid_kfold5` |
<!-- END TABLE:temps -->

Several T3 folds hit the bounds (T = 20 flattens the probabilities to ≈ 0.5; T = 0.05 in one A9 fold) —
a symptom of models that learned nothing transferable on T3.

![Reliability A10 T1](figures/reliability_deep_T1_A10_hybrid_kfold5.png)

*Figure 6.1 — Reliability diagram, full hybrid A10, T1 (before vs after temperature scaling).*

![Reliability EEGNet T1](figures/reliability_deep_T1_eegnet_kfold5.png)

*Figure 6.2 — Reliability diagram, EEGNet T1: calibration got worse after temperature scaling.*

## 6.8 Latency per window (CPU)
<!-- BEGIN TABLE:latency -->
| Model | Kind | Feature extraction ms | Model ms (single window, median) | Classical end-to-end ms | Source |
|---|---|---|---|---|---|
| Logistic Regression | classical | 40.53 | 1.32 | 41.85 | `cls_T1_lr_all_none_kfold5` |
| SVM (RBF) | classical | 40.53 | 2.75 | 43.28 | `cls_T1_svm_all_none_kfold5` |
| Random Forest | classical | 40.53 | 10.56 | 51.09 | `cls_T1_rf_all_none_kfold5` |
| XGBoost | classical | 40.53 | 0.95 | 41.48 | `cls_T1_xgb_all_none_kfold5` |
| EEGNet | deep (1 epoch(s) per forward pass) | — | 0.88 (p95 1.16) | — | `deep_T1_eegnet_kfold5` |
| A6 CNN | deep (8 epoch(s) per forward pass) | — | 4.37 (p95 6.64) | — | `deep_T1_A6_cnn_kfold5` |
| A7 CNN-LSTM | deep (8 epoch(s) per forward pass) | — | 9.76 (p95 13.07) | — | `deep_T1_A7_cnn_lstm_kfold5` |
| A8 CNN-LSTM-Transformer | deep (8 epoch(s) per forward pass) | — | 11.30 (p95 13.71) | — | `deep_T1_A8_cnn_lstm_trans_kfold5` |
| A9 CNN-LSTM-GAT | deep (8 epoch(s) per forward pass) | — | 6.51 (p95 8.44) | — | `deep_T1_A9_cnn_lstm_gnn_kfold5` |
| A10 full hybrid (CNN+LSTM+Transformer+GAT) | deep (8 epoch(s) per forward pass) | — | 21.78 (p95 40.99) | — | `deep_T1_A10_hybrid_kfold5` |

End-to-end engine latency per window (filtering 20 s context + quality + features + model + JSON), from the offline demo streams produced by `scripts/export_demo.py` (model `T1_xgb_all`, 592 windows): median 98.4 ms, p95 121.4 ms — saved to `results/engine_latency.json`.
<!-- END TABLE:latency -->

![Latency](figures/summary_latency.png)

*Figure 6.3 — CPU latency per window (generated from the result files).*

## 6.9 Deployable checkpoints and the T3 hybrid issue
After cross-validation, deployable models were trained on all labelled subjects (`scripts/run_deep.py --final`,
`scripts/train_final_classical.py`):
<!-- BEGIN TABLE:models_deployed -->
| Model id (models/) | Task | Version | Temperature | CV experiment it represents | That experiment's fold macro-F1 |
|---|---|---|---|---|---|
| `T1_A10_hybrid` | workload_binary | A10_hybrid-T1-20261019 | 1.62 | deep_T1_A10_hybrid_kfold5 | 0.758 |
| `T1_A8_cnn_lstm_trans` | workload_binary | A8_cnn_lstm_trans-T1-20261019 | 1.10 | deep_T1_A8_cnn_lstm_trans_kfold5 | 0.773 |
| `T1_xgb_all` | workload_binary | xgb-all-T1-20261019 | — | cls_T1_xgb_all_none_loso | 0.859 |
| `T3_A10_hybrid` | workload_within_task | A10_hybrid-T3-20261019 | 20.00 | deep_T3_A10_hybrid_kfold5 | 0.389 |
| `T3_xgb_all` | workload_within_task | xgb-all-T3-20261019 | — | cls_T3_xgb_all_none_loso | — |

Source: `models/*.json` (written by `scripts/train_final_classical.py` and `scripts/run_deep.py --final`)
<!-- END TABLE:models_deployed -->

**Is the T3 hybrid "early stop at epoch 0" fixed? — No, and it cannot be fixed by tuning alone.** The final
T3 hybrid stopped at epoch 0 with T = 20.00 [`models/T3_A10_hybrid.json` → `temperature`] (the upper bound), so its
probabilities are ≈ 0.5. In cross-validation the same model is at chance (pooled AUROC
0.501 [`results/deep_T3_A10_hybrid_kfold5/metrics.json` → `pooled.auroc`]). Two code fixes were made before the
full runs — the validation split must contain both classes, and T = 1 when it does not
(`trainer.py:run_fold`, `train_model`) — but the underlying cause is that T3 contains no signal these models
can learn from 45 recordings. The near-0.5 output is the honest behaviour. It is documented in D18.

## 6.10 Limitations
Small, untuned models (no hyper-parameter search); 5 folds instead of LOSO; validation sets of ≈ 6 subjects
make early stopping and calibration noisy; CPU-only training capped at 25 epochs; kernel 32 instead of 64;
no data from more subjects; the GAT uses a single averaged PLV graph (band-specific graphs not tried).
