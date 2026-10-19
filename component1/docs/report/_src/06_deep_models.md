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
<!-- TABLE:deep -->

Paired Wilcoxon tests (`scripts/deep_stats.py`; per-subject macro-F1 on T1, accuracy on T3):
<!-- TABLE:stats_deep -->

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
<!-- TABLE:temps -->

Several T3 folds hit the bounds (T = 20 flattens the probabilities to ≈ 0.5; T = 0.05 in one A9 fold) —
a symptom of models that learned nothing transferable on T3.

![Reliability A10 T1](figures/reliability_deep_T1_A10_hybrid_kfold5.png)

*Figure 6.1 — Reliability diagram, full hybrid A10, T1 (before vs after temperature scaling).*

![Reliability EEGNet T1](figures/reliability_deep_T1_eegnet_kfold5.png)

*Figure 6.2 — Reliability diagram, EEGNet T1: calibration got worse after temperature scaling.*

## 6.8 Latency per window (CPU)
<!-- TABLE:latency -->

![Latency](figures/summary_latency.png)

*Figure 6.3 — CPU latency per window (generated from the result files).*

## 6.9 Deployable checkpoints and the T3 hybrid issue
After cross-validation, deployable models were trained on all labelled subjects (`scripts/run_deep.py --final`,
`scripts/train_final_classical.py`):
<!-- TABLE:models_deployed -->

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
