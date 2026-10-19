# 7. Evaluation and results

## 7.1 Metrics — definitions and why
Implemented in `src/eegengine/evaluation/metrics.py:classification_metrics` (scikit-learn).

| Metric | Definition | Why it is used |
|---|---|---|
| **Macro-F1** (primary) | mean over classes of `F1_c = 2·P_c·R_c / (P_c + R_c)` | weights every class equally, so a model cannot score well by favouring the majority class; standard for imbalanced multi-class EEG tasks (T2/T3 are imbalanced) |
| **Balanced accuracy** | mean over classes of recall `TP_c / (TP_c + FN_c)` | chance level is exactly 1/n_classes regardless of imbalance — easy to compare with 0.5 |
| **Precision (macro)** | mean of `TP_c / (TP_c + FP_c)` | how trustworthy a predicted class is |
| **Recall (macro)** | mean of `TP_c / (TP_c + FN_c)` | how many true cases are found (= balanced accuracy) |
| **AUROC** | area under the ROC curve; binary: of P(high); 3-class: macro one-vs-rest over the classes present | threshold-free ranking quality; 0.5 = chance |
| **Confusion matrix** | counts of true vs predicted classes | shows *which* errors happen |
| **ECE** | `Σ_b (n_b/n)·|acc_b − conf_b|`, 15 bins | whether probabilities can be trusted (needed by the fusion component) |
| **Latency** | median ms per window on CPU | real-time feasibility (an output is due every 2 s) |

Three views of the out-of-fold predictions: **per fold** (mean ± std over folds where defined), **pooled**
(all predictions together) and **per recording** (mean probability per recording → one decision; for T3 one
decision per subject). Per-subject scores for the Wilcoxon tests use macro-F1 when every subject has both
classes (T1, T2) and accuracy otherwise (T3) (`metrics.py:per_subject_scores`).

## 7.2 Master results table (every experiment in `results/`)
<!-- BEGIN TABLE:master -->
| Experiment id (results/<id>/metrics.json) | Task | Kind | Model | Features | Norm | CV | Folds | Fold macro-F1 | Fold bal. acc | Pooled macro-F1 | Pooled precision | Pooled recall | Pooled AUROC | Recording macro-F1 | 18-s window fold macro-F1 (kfold5 classical) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cls_T1_lr_all_none_kfold5 | T1 | classical | lr | all | none | kfold5 | 5 | 0.841 ± 0.037 | 0.841 ± 0.036 | 0.841 | 0.843 | 0.841 | 0.912 | 0.927 | 0.896 ± 0.051 |
| cls_T1_lr_all_none_loso | T1 | classical | lr | all | none | loso | 48 | 0.836 ± 0.134 | 0.845 ± 0.118 | 0.845 | 0.845 | 0.845 | 0.918 | 0.948 | — |
| cls_T1_lr_all_rest_baseline_loso | T1 | classical | lr | all | rest_baseline | loso | 48 | 0.882 ± 0.142 | 0.892 ± 0.121 | 0.891 | 0.906 | 0.892 | 0.959 | 0.927 | — |
| cls_T1_rf_all_none_kfold5 | T1 | classical | rf | all | none | kfold5 | 5 | 0.860 ± 0.035 | 0.860 ± 0.035 | 0.859 | 0.859 | 0.859 | 0.929 | 0.948 | 0.907 ± 0.039 |
| cls_T1_rf_all_none_loso | T1 | classical | rf | all | none | loso | 48 | 0.844 ± 0.140 | 0.851 ± 0.128 | 0.851 | 0.851 | 0.851 | 0.926 | 0.938 | — |
| cls_T1_rf_all_rest_baseline_loso | T1 | classical | rf | all | rest_baseline | loso | 48 | 0.922 ± 0.109 | 0.926 ± 0.092 | 0.926 | 0.927 | 0.926 | 0.977 | 0.979 | — |
| cls_T1_svm_all_none_kfold5 | T1 | classical | svm | all | none | kfold5 | 5 | 0.854 ± 0.046 | 0.854 ± 0.046 | 0.855 | 0.858 | 0.855 | 0.932 | 0.896 | 0.897 ± 0.048 |
| cls_T1_svm_all_none_loso | T1 | classical | svm | all | none | loso | 48 | 0.850 ± 0.148 | 0.860 ± 0.130 | 0.859 | 0.862 | 0.860 | 0.938 | 0.906 | — |
| cls_T1_svm_all_rest_baseline_loso | T1 | classical | svm | all | rest_baseline | loso | 48 | 0.948 ± 0.101 | 0.951 ± 0.083 | 0.951 | 0.953 | 0.951 | 0.990 | 0.990 | — |
| cls_T1_xgb_A1_spectral_none_loso | T1 | classical | xgb | A1_spectral | none | loso | 48 | 0.802 ± 0.137 | 0.810 ± 0.124 | 0.810 | 0.811 | 0.810 | 0.890 | 0.895 | — |
| cls_T1_xgb_A1_spectral_rest_baseline_loso | T1 | classical | xgb | A1_spectral | rest_baseline | loso | 48 | 0.885 ± 0.133 | 0.893 ± 0.114 | 0.892 | 0.901 | 0.893 | 0.969 | 0.958 | — |
| cls_T1_xgb_A2_spectral_entropy_none_loso | T1 | classical | xgb | A2_spectral_entropy | none | loso | 48 | 0.819 ± 0.128 | 0.826 ± 0.116 | 0.826 | 0.828 | 0.826 | 0.904 | 0.927 | — |
| cls_T1_xgb_A2_spectral_entropy_rest_baseline_loso | T1 | classical | xgb | A2_spectral_entropy | rest_baseline | loso | 48 | 0.887 ± 0.127 | 0.893 ± 0.110 | 0.893 | 0.903 | 0.893 | 0.971 | 0.948 | — |
| cls_T1_xgb_A3_spectral_wavelet_none_loso | T1 | classical | xgb | A3_spectral_wavelet | none | loso | 48 | 0.812 ± 0.147 | 0.822 ± 0.130 | 0.822 | 0.823 | 0.822 | 0.897 | 0.895 | — |
| cls_T1_xgb_A3_spectral_wavelet_rest_baseline_loso | T1 | classical | xgb | A3_spectral_wavelet | rest_baseline | loso | 48 | 0.879 ± 0.155 | 0.889 ± 0.132 | 0.889 | 0.902 | 0.889 | 0.973 | 0.927 | — |
| cls_T1_xgb_A4_spectral_connectivity_none_loso | T1 | classical | xgb | A4_spectral_connectivity | none | loso | 48 | 0.860 ± 0.132 | 0.866 ± 0.121 | 0.866 | 0.867 | 0.866 | 0.938 | 0.927 | — |
| cls_T1_xgb_A4_spectral_connectivity_rest_baseline_loso | T1 | classical | xgb | A4_spectral_connectivity | rest_baseline | loso | 48 | 0.924 ± 0.114 | 0.929 ± 0.096 | 0.929 | 0.934 | 0.929 | 0.987 | 0.969 | — |
| cls_T1_xgb_A5_spectral_connectivity_graph_none_loso | T1 | classical | xgb | A5_spectral_connectivity_graph | none | loso | 48 | 0.865 ± 0.123 | 0.870 ± 0.113 | 0.870 | 0.871 | 0.870 | 0.940 | 0.937 | — |
| cls_T1_xgb_A5_spectral_connectivity_graph_rest_baseline_loso | T1 | classical | xgb | A5_spectral_connectivity_graph | rest_baseline | loso | 48 | 0.918 ± 0.127 | 0.924 ± 0.107 | 0.923 | 0.930 | 0.924 | 0.987 | 0.958 | — |
| cls_T1_xgb_all_none_kfold5 | T1 | classical | xgb | all | none | kfold5 | 5 | 0.874 ± 0.035 | 0.874 ± 0.035 | 0.874 | 0.875 | 0.874 | 0.942 | 0.958 | 0.915 ± 0.035 |
| cls_T1_xgb_all_none_loso | T1 | classical | xgb | all | none | loso | 48 | 0.859 ± 0.139 | 0.865 ± 0.127 | 0.865 | 0.867 | 0.865 | 0.941 | 0.917 | — |
| cls_T1_xgb_all_rest_baseline_loso | T1 | classical | xgb | all | rest_baseline | loso | 48 | 0.918 ± 0.128 | 0.924 ± 0.108 | 0.924 | 0.931 | 0.924 | 0.986 | 0.958 | — |
| cls_T2_lr_all_none_loso | T2 | classical | lr | all | none | loso | 45 | 0.417 ± 0.126 | 0.588 ± 0.192 | 0.537 | 0.537 | 0.538 | 0.738 | 0.568 | — |
| cls_T2_rf_all_none_loso | T2 | classical | rf | all | none | loso | 45 | 0.427 ± 0.139 | 0.631 ± 0.203 | 0.555 | 0.562 | 0.563 | 0.768 | 0.629 | — |
| cls_T2_svm_all_none_loso | T2 | classical | svm | all | none | loso | 45 | 0.421 ± 0.136 | 0.601 ± 0.212 | 0.544 | 0.544 | 0.546 | 0.726 | 0.572 | — |
| cls_T2_xgb_all_none_loso | T2 | classical | xgb | all | none | loso | 45 | 0.436 ± 0.134 | 0.623 ± 0.209 | 0.565 | 0.564 | 0.567 | 0.770 | 0.618 | — |
| cls_T2_xgb_all_rest_baseline_loso | T2 | classical | xgb | all | rest_baseline | loso | 45 | 0.439 ± 0.152 | 0.650 ± 0.211 | 0.555 | 0.571 | 0.563 | 0.821 | 0.538 | — |
| cls_T3_lr_all_none_kfold5 | T3 | classical | lr | all | none | kfold5 | 5 | 0.480 ± 0.103 | 0.501 ± 0.085 | 0.487 | 0.487 | 0.487 | 0.483 | 0.440 | 0.467 ± 0.117 |
| cls_T3_lr_all_none_loso | T3 | classical | lr | all | none | loso | 45 | — | — | 0.501 | 0.501 | 0.501 | 0.509 | 0.472 | — |
| cls_T3_rf_all_none_kfold5 | T3 | classical | rf | all | none | kfold5 | 5 | 0.512 ± 0.119 | 0.527 ± 0.105 | 0.531 | 0.536 | 0.533 | 0.534 | 0.583 | 0.525 ± 0.133 |
| cls_T3_rf_all_none_loso | T3 | classical | rf | all | none | loso | 45 | — | — | 0.527 | 0.531 | 0.529 | 0.536 | 0.556 | — |
| cls_T3_svm_all_none_kfold5 | T3 | classical | svm | all | none | kfold5 | 5 | 0.481 ± 0.099 | 0.493 ± 0.093 | 0.499 | 0.499 | 0.499 | 0.494 | 0.499 | 0.489 ± 0.129 |
| cls_T3_svm_all_none_loso | T3 | classical | svm | all | none | loso | 45 | — | — | 0.481 | 0.481 | 0.482 | 0.485 | 0.434 | — |
| cls_T3_xgb_A1_spectral_none_loso | T3 | classical | xgb | A1_spectral | none | loso | 45 | — | — | 0.481 | 0.481 | 0.481 | 0.474 | 0.440 | — |
| cls_T3_xgb_A1_spectral_rest_baseline_loso | T3 | classical | xgb | A1_spectral | rest_baseline | loso | 45 | — | — | 0.518 | 0.518 | 0.519 | 0.541 | 0.530 | — |
| cls_T3_xgb_A2_spectral_entropy_none_loso | T3 | classical | xgb | A2_spectral_entropy | none | loso | 45 | — | — | 0.465 | 0.465 | 0.465 | 0.450 | 0.440 | — |
| cls_T3_xgb_A2_spectral_entropy_rest_baseline_loso | T3 | classical | xgb | A2_spectral_entropy | rest_baseline | loso | 45 | — | — | 0.512 | 0.512 | 0.512 | 0.530 | 0.460 | — |
| cls_T3_xgb_A3_spectral_wavelet_none_loso | T3 | classical | xgb | A3_spectral_wavelet | none | loso | 45 | — | — | 0.481 | 0.481 | 0.481 | 0.464 | 0.464 | — |
| cls_T3_xgb_A3_spectral_wavelet_rest_baseline_loso | T3 | classical | xgb | A3_spectral_wavelet | rest_baseline | loso | 45 | — | — | 0.512 | 0.512 | 0.512 | 0.528 | 0.460 | — |
| cls_T3_xgb_A4_spectral_connectivity_none_loso | T3 | classical | xgb | A4_spectral_connectivity | none | loso | 45 | — | — | 0.542 | 0.543 | 0.542 | 0.562 | 0.564 | — |
| cls_T3_xgb_A4_spectral_connectivity_rest_baseline_loso | T3 | classical | xgb | A4_spectral_connectivity | rest_baseline | loso | 45 | — | — | 0.513 | 0.513 | 0.513 | 0.512 | 0.453 | — |
| cls_T3_xgb_A5_spectral_connectivity_graph_none_loso | T3 | classical | xgb | A5_spectral_connectivity_graph | none | loso | 45 | — | — | 0.534 | 0.534 | 0.534 | 0.552 | 0.570 | — |
| cls_T3_xgb_A5_spectral_connectivity_graph_rest_baseline_loso | T3 | classical | xgb | A5_spectral_connectivity_graph | rest_baseline | loso | 45 | — | — | 0.384 | 0.383 | 0.387 | 0.350 | 0.415 | — |
| cls_T3_xgb_all_none_kfold5 | T3 | classical | xgb | all | none | kfold5 | 5 | 0.530 ± 0.109 | 0.545 ± 0.093 | 0.548 | 0.548 | 0.548 | 0.550 | 0.590 | 0.543 ± 0.135 |
| cls_T3_xgb_all_none_loso | T3 | classical | xgb | all | none | loso | 45 | — | — | 0.538 | 0.539 | 0.539 | 0.544 | 0.545 | — |
| cls_T3_xgb_all_rest_baseline_loso | T3 | classical | xgb | all | rest_baseline | loso | 45 | — | — | 0.390 | 0.389 | 0.392 | 0.356 | 0.380 | — |
| deep_T1_A10_hybrid_kfold5 | T1 | deep | A10_hybrid | raw_epochs+graph | train_fold_channel_scale | kfold5 | 5 | 0.758 ± 0.103 | 0.765 ± 0.093 | 0.762 | 0.764 | 0.762 | 0.853 | 0.802 | — |
| deep_T1_A6_cnn_kfold5 | T1 | deep | A6_cnn | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.701 ± 0.138 | 0.709 ± 0.127 | 0.705 | 0.710 | 0.706 | 0.782 | 0.768 | — |
| deep_T1_A7_cnn_lstm_kfold5 | T1 | deep | A7_cnn_lstm | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.730 ± 0.091 | 0.736 ± 0.085 | 0.733 | 0.734 | 0.733 | 0.807 | 0.771 | — |
| deep_T1_A8_cnn_lstm_trans_kfold5 | T1 | deep | A8_cnn_lstm_trans | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.773 ± 0.045 | 0.775 ± 0.045 | 0.774 | 0.778 | 0.775 | 0.835 | 0.833 | — |
| deep_T1_A9_cnn_lstm_gnn_kfold5 | T1 | deep | A9_cnn_lstm_gnn | raw_epochs+graph | train_fold_channel_scale | kfold5 | 5 | 0.751 ± 0.077 | 0.751 ± 0.077 | 0.749 | 0.749 | 0.749 | 0.827 | 0.812 | — |
| deep_T1_eegnet_kfold5 | T1 | deep | eegnet | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.772 ± 0.099 | 0.773 ± 0.098 | 0.771 | 0.771 | 0.771 | 0.843 | 0.812 | — |
| deep_T3_A10_hybrid_kfold5 | T3 | deep | A10_hybrid | raw_epochs+graph | train_fold_channel_scale | kfold5 | 5 | 0.389 ± 0.062 | 0.492 ± 0.023 | 0.484 | 0.489 | 0.489 | 0.501 | 0.466 | — |
| deep_T3_A6_cnn_kfold5 | T3 | deep | A6_cnn | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.458 ± 0.076 | 0.468 ± 0.084 | 0.469 | 0.470 | 0.470 | 0.447 | 0.434 | — |
| deep_T3_A7_cnn_lstm_kfold5 | T3 | deep | A7_cnn_lstm | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.433 ± 0.089 | 0.476 ± 0.067 | 0.487 | 0.491 | 0.491 | 0.485 | 0.440 | — |
| deep_T3_A8_cnn_lstm_trans_kfold5 | T3 | deep | A8_cnn_lstm_trans | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.412 ± 0.097 | 0.500 ± 0.043 | 0.501 | 0.505 | 0.504 | 0.504 | 0.527 | — |
| deep_T3_A9_cnn_lstm_gnn_kfold5 | T3 | deep | A9_cnn_lstm_gnn | raw_epochs+graph | train_fold_channel_scale | kfold5 | 5 | 0.487 ± 0.058 | 0.493 ± 0.058 | 0.495 | 0.495 | 0.495 | 0.484 | 0.398 | — |
| deep_T3_eegnet_kfold5 | T3 | deep | eegnet | raw_epochs | train_fold_channel_scale | kfold5 | 5 | 0.514 ± 0.076 | 0.536 ± 0.085 | 0.540 | 0.540 | 0.540 | 0.543 | 0.464 | — |
<!-- END TABLE:master -->

Headline comparison (same 5 folds, same 18-s windows, no rest baseline):

![Classical vs deep](figures/summary_classical_vs_deep.png)

*Figure 7.1 — Classical (blue) vs deep (orange) models on identical folds and windows (generated from the result files).*

<!-- BEGIN TABLE:cls_kfold5 -->
| Task | Model | Epoch-level fold macro-F1 | Epoch-level pooled AUROC | 18-s window fold macro-F1 | 18-s window fold bal. acc | 18-s window pooled AUROC | Source |
|---|---|---|---|---|---|---|---|
| T1 | Logistic Regression | 0.841 ± 0.037 | 0.912 | 0.896 ± 0.051 | 0.897 ± 0.050 | 0.955 | `cls_T1_lr_all_none_kfold5` |
| T1 | SVM (RBF) | 0.854 ± 0.046 | 0.932 | 0.897 ± 0.048 | 0.897 ± 0.047 | 0.966 | `cls_T1_svm_all_none_kfold5` |
| T1 | Random Forest | 0.860 ± 0.035 | 0.929 | 0.907 ± 0.039 | 0.907 ± 0.039 | 0.969 | `cls_T1_rf_all_none_kfold5` |
| T1 | XGBoost | 0.874 ± 0.035 | 0.942 | 0.915 ± 0.035 | 0.915 ± 0.035 | 0.977 | `cls_T1_xgb_all_none_kfold5` |
| T3 | Logistic Regression | 0.480 ± 0.103 | 0.483 | 0.467 ± 0.117 | 0.485 ± 0.109 | 0.482 | `cls_T3_lr_all_none_kfold5` |
| T3 | SVM (RBF) | 0.481 ± 0.099 | 0.494 | 0.489 ± 0.129 | 0.503 ± 0.127 | 0.492 | `cls_T3_svm_all_none_kfold5` |
| T3 | Random Forest | 0.512 ± 0.119 | 0.534 | 0.525 ± 0.133 | 0.539 ± 0.120 | 0.542 | `cls_T3_rf_all_none_kfold5` |
| T3 | XGBoost | 0.530 ± 0.109 | 0.550 | 0.543 ± 0.135 | 0.559 ± 0.123 | 0.565 | `cls_T3_xgb_all_none_kfold5` |
<!-- END TABLE:cls_kfold5 -->

## 7.3 Rest-baseline inflation
<!-- BEGIN TABLE:rest_inflation -->
| Task | Model | Pooled bal. acc — none | — rest baseline | Δ | Recording bal. acc — none | — rest baseline | Sources |
|---|---|---|---|---|---|---|---|
| T1 | Logistic Regression | 0.845 | 0.892 | +0.047 | 0.948 | 0.927 | `cls_T1_lr_all_none_loso` / `cls_T1_lr_all_rest_baseline_loso` |
| T1 | SVM (RBF) | 0.860 | 0.951 | +0.091 | 0.906 | 0.990 | `cls_T1_svm_all_none_loso` / `cls_T1_svm_all_rest_baseline_loso` |
| T1 | Random Forest | 0.851 | 0.926 | +0.075 | 0.938 | 0.979 | `cls_T1_rf_all_none_loso` / `cls_T1_rf_all_rest_baseline_loso` |
| T1 | XGBoost | 0.865 | 0.924 | +0.059 | 0.917 | 0.958 | `cls_T1_xgb_all_none_loso` / `cls_T1_xgb_all_rest_baseline_loso` |
| T2 | XGBoost | 0.567 | 0.563 | -0.003 | 0.619 | 0.552 | `cls_T2_xgb_all_none_loso` / `cls_T2_xgb_all_rest_baseline_loso` |
| T3 | XGBoost | 0.539 | 0.392 | -0.147 | 0.545 | 0.385 | `cls_T3_xgb_all_none_loso` / `cls_T3_xgb_all_rest_baseline_loso` |
<!-- END TABLE:rest_inflation -->

![Rest-baseline inflation](figures/summary_rest_baseline_inflation.png)

*Figure 7.2 — T1 balanced accuracy with vs without the subject rest baseline.*

With the rest baseline, every T1 model improves significantly (Wilcoxon on per-subject macro-F1:
SVM p = 2.6e-06 [`results/stats_classical.json` → `tests.12.p_value`], RF p = 3.6e-06 [`results/stats_classical.json` → `tests.14.p_value`],
XGBoost p = 0.0011 [`results/stats_classical.json` → `tests.16.p_value`], LR p = 0.0079 [`results/stats_classical.json` → `tests.10.p_value`]).
The reason is mechanical, not neurophysiological: z-scoring with the subject's own rest recording makes the
rest recording's average feature vector exactly 0 (§3.8), so recognising "the reference recording" becomes
easy. On T3 the rest baseline makes XGBoost *worse* (per-subject accuracy 0.404 [`results/stats_classical.json` → `tests.30.mean_a`]
vs 0.547 [`results/stats_classical.json` → `tests.30.mean_b`], p = 0.010 [`results/stats_classical.json` → `tests.30.p_value`]) — T3 contains no
rest/task contrast for it to exploit, and the extra subject-specific scaling adds variance. **No headline
result uses the rest baseline.**

## 7.4 Confusion matrices of the key models
<!-- BEGIN TABLE:confusions -->
**`cls_T1_xgb_all_none_loso` — pooled** (source: `results/cls_T1_xgb_all_none_loso/metrics.json` → `confusion_pooled`)

| true \ predicted | low | high |
|---|---|---|
| low | 2964 | 588 |
| high | 368 | 3184 |

**`cls_T1_xgb_all_none_loso` — per recording** (source: `results/cls_T1_xgb_all_none_loso/metrics.json` → `confusion_recording`)

| true \ predicted | low | high |
|---|---|---|
| low | 42 | 6 |
| high | 2 | 46 |

**`cls_T1_xgb_all_none_kfold5` — pooled** (source: `results/cls_T1_xgb_all_none_kfold5/metrics.json` → `confusion_pooled`)

| true \ predicted | low | high |
|---|---|---|
| low | 3023 | 529 |
| high | 367 | 3185 |

**`cls_T1_xgb_all_none_kfold5` — per recording** (source: `results/cls_T1_xgb_all_none_kfold5/metrics.json` → `confusion_recording`)

| true \ predicted | low | high |
|---|---|---|
| low | 46 | 2 |
| high | 2 | 46 |

**`deep_T1_A8_cnn_lstm_trans_kfold5` — pooled** (source: `results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `confusion_pooled`)

| true \ predicted | low | high |
|---|---|---|
| low | 1179 | 453 |
| high | 282 | 1350 |

**`deep_T1_A8_cnn_lstm_trans_kfold5` — per recording** (source: `results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `confusion_recording`)

| true \ predicted | low | high |
|---|---|---|
| low | 39 | 9 |
| high | 7 | 41 |

**`deep_T1_A10_hybrid_kfold5` — pooled** (source: `results/deep_T1_A10_hybrid_kfold5/metrics.json` → `confusion_pooled`)

| true \ predicted | low | high |
|---|---|---|
| low | 1311 | 321 |
| high | 456 | 1176 |

**`deep_T1_A10_hybrid_kfold5` — per recording** (source: `results/deep_T1_A10_hybrid_kfold5/metrics.json` → `confusion_recording`)

| true \ predicted | low | high |
|---|---|---|
| low | 41 | 7 |
| high | 12 | 36 |

**`cls_T3_xgb_all_none_loso` — pooled** (source: `results/cls_T3_xgb_all_none_loso/metrics.json` → `confusion_pooled`)

| true \ predicted | moderate | high |
|---|---|---|
| moderate | 686 | 794 |
| high | 715 | 1135 |

**`cls_T3_xgb_all_none_loso` — per recording** (source: `results/cls_T3_xgb_all_none_loso/metrics.json` → `confusion_recording`)

| true \ predicted | moderate | high |
|---|---|---|
| moderate | 9 | 11 |
| high | 9 | 16 |

**`deep_T3_A10_hybrid_kfold5` — pooled** (source: `results/deep_T3_A10_hybrid_kfold5/metrics.json` → `confusion_pooled`)

| true \ predicted | moderate | high |
|---|---|---|
| moderate | 367 | 313 |
| high | 477 | 373 |

**`deep_T3_A10_hybrid_kfold5` — per recording** (source: `results/deep_T3_A10_hybrid_kfold5/metrics.json` → `confusion_recording`)

| true \ predicted | moderate | high |
|---|---|---|
| moderate | 10 | 10 |
| high | 14 | 11 |

**`cls_T2_xgb_all_none_loso` — pooled** (source: `results/cls_T2_xgb_all_none_loso/metrics.json` → `confusion_pooled`)

| true \ predicted | low | medium | high |
|---|---|---|---|
| low | 2608 | 288 | 212 |
| medium | 427 | 561 | 714 |
| high | 172 | 695 | 983 |

**`cls_T2_xgb_all_none_loso` — per recording** (source: `results/cls_T2_xgb_all_none_loso/metrics.json` → `confusion_recording`)

| true \ predicted | low | medium | high |
|---|---|---|---|
| low | 38 | 2 | 2 |
| medium | 5 | 9 | 9 |
| high | 1 | 10 | 14 |

<!-- END TABLE:confusions -->

![CM XGBoost T1 LOSO](figures/cm_cls_T1_xgb_all_none_loso.png)

*Figure 7.3 — XGBoost T1 LOSO: more rest epochs are predicted as task (588) than task epochs as rest (368)
(`results/cls_T1_xgb_all_none_loso/metrics.json` → `confusion_pooled`); per recording only 8 of 96 are wrong.*

![CM hybrid T1](figures/cm_deep_T1_A10_hybrid_kfold5.png)

*Figure 7.4 — Full hybrid T1: more task windows are classified as rest than vice versa.*

![CM XGBoost T3](figures/cm_cls_T3_xgb_all_none_loso.png)

*Figure 7.5 — XGBoost T3 LOSO: near-random confusion.*

## 7.5 Leakage audit
Re-runnable with `scripts/audit_leakage.py`; written up in `docs/leakage_audit.md`.
<!-- BEGIN TABLE:leakage -->
| Check | Result | Verdict |
|---|---|---|
| Fold integrity | 58 experiments, problems: 0 | OK |
| Feature locality (5 epochs re-extracted alone) | max abs difference 0.0 | OK |
| Label-permutation test (3 runs, LR LOSO T1) | balanced accuracy 0.498, 0.508, 0.512; mean 0.506 | ≈ chance → OK |
| Subject-identity probe (random epoch split) | accuracy 0.995 (chance 0.021) | shows why subject-wise CV is required |
| Results with balanced accuracy > 0.97 | cls_T1_rf_all_rest_baseline_loso (recording_level, 0.979, rest_baseline); cls_T1_svm_all_rest_baseline_loso (recording_level, 0.990, rest_baseline) | audited (rest-baseline effect) |

Source: `results/leakage_audit.json`
<!-- END TABLE:leakage -->

* **Fold integrity:** for every saved prediction the fold id is compared with the subject — LOSO fold id must
  equal the test subject; 5-fold runs must match `results/folds_5fold.json`.
* **Shuffled-label control:** the full LOSO pipeline (StandardScaler + LR, all features, T1) with the lo/hi
  labels randomly swapped per subject and shuffled within subject, 3 times: balanced accuracy
  0.498 [`results/leakage_audit.json` → `permutation_test.runs.0.pooled_balanced_accuracy`],
  0.508 [`results/leakage_audit.json` → `permutation_test.runs.1.pooled_balanced_accuracy`],
  0.512 [`results/leakage_audit.json` → `permutation_test.runs.2.pooled_balanced_accuracy`] — chance. A leaking pipeline would
  stay above chance.
* **Feature locality:** features of 5 epochs re-extracted alone equal the batch values exactly.
* **Built-in guards:** `cv.py:assert_no_subject_overlap` on every split (also fit/val/test inside the deep
  trainer); scalers, selection and PCA inside the sklearn Pipeline; deep scalers fit on fit subjects; early
  stopping and temperature scaling on a subject-wise inner validation split; hyper-parameters fixed in advance.
* **The > 97 % cases:** only two, both **T1 + rest baseline, per-recording view**: SVM
  0.990 [`results/leakage_audit.json` → `results_above_97pct.1.balanced_accuracy`] and RF
  0.979 [`results/leakage_audit.json` → `results_above_97pct.0.balanced_accuracy`]. Without the baseline the same models give
  per-recording balanced accuracy 0.906 [`results/cls_T1_svm_all_none_loso/metrics.json` → `recording_level.balanced_accuracy`] and
  0.938 [`results/cls_T1_rf_all_none_loso/metrics.json` → `recording_level.balanced_accuracy`]. Explanation: the reference-recording
  effect of §3.8 (averaging a recording's 74 epochs makes "is this the zero-mean reference?" almost trivial). Not
  cross-subject leakage; reported separately and never used as a headline.
* **Answer to "is 0.95 real?":** the 0.951 [`results/cls_T1_svm_all_rest_baseline_loso/metrics.json` → `pooled.balanced_accuracy`] (SVM,
  T1, rest baseline) is real *for that protocol* but it is inflated by the rest baseline; the honest T1 number is
  0.865 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `pooled.balanced_accuracy`] (XGBoost, no baseline, LOSO).

## 7.6 Key numbers
| What | Value |
|---|---|
| T1 best classical, LOSO, pooled macro-F1 | 0.865 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `pooled.macro_f1`] |
| T1 best classical, LOSO, fold macro-F1 | 0.859 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `per_fold.macro_f1.mean`] ± 0.139 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `per_fold.macro_f1.std`] |
| T1 best classical, 5-fold, 18-s windows, fold macro-F1 | 0.915 [`results/cls_T1_xgb_all_none_kfold5/metrics.json` → `sequence_level.per_fold.macro_f1.mean`] ± 0.035 [`results/cls_T1_xgb_all_none_kfold5/metrics.json` → `sequence_level.per_fold.macro_f1.std`] |
| T1 best deep (A8), fold macro-F1 | 0.773 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `per_fold.macro_f1.mean`] ± 0.045 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `per_fold.macro_f1.std`] |
| T1 full hybrid (A10), fold macro-F1 | 0.758 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `per_fold.macro_f1.mean`] ± 0.103 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `per_fold.macro_f1.std`] |
| T3 best classical, LOSO, pooled macro-F1 / AUROC | 0.538 [`results/cls_T3_xgb_all_none_loso/metrics.json` → `pooled.macro_f1`] / 0.544 [`results/cls_T3_xgb_all_none_loso/metrics.json` → `pooled.auroc`] |
| T3 best deep (EEGNet), fold macro-F1 | 0.514 [`results/deep_T3_eegnet_kfold5/metrics.json` → `per_fold.macro_f1.mean`] |
| T3 full hybrid, fold macro-F1 / pooled AUROC | 0.389 [`results/deep_T3_A10_hybrid_kfold5/metrics.json` → `per_fold.macro_f1.mean`] / 0.501 [`results/deep_T3_A10_hybrid_kfold5/metrics.json` → `pooled.auroc`] |

## 7.7 Interpretation
1. **T1 works across subjects.** Rest and multitasking are separated for unseen people with balanced accuracy
   ≈ 0.87 per epoch and ≈ 0.92 over 18-s windows. Averaging over a longer window helps because single 4-s
   epochs are noisy.
2. **Connectivity helps.** Adding PLV/coherence is the only significant improvement over band power on T1 —
   the network-level coupling between regions changes between rest and task beyond what local power shows.
   Graph metrics add nothing beyond the PLV values they are computed from.
3. **T3 ≈ chance — the central scientific finding.** When condition is held fixed, neither 1,499 handcrafted
   features nor any deep model separates moderate from high self-rated workload. Plausible reasons: (a) only 45
   labelled recordings, one per subject, so between-subject differences in EEG dwarf the effect; (b) self-ratings
   are subjective and compressed (task ratings 4–9, most 5–8), so the "moderate" vs "high" boundary at 6/7 is
   noisy; (c) all subjects did the same task, so the true workload difference between groups may be small;
   (d) consumer-grade EEG with frequent artefacts. Consequence: high T1 scores mainly reflect the rest-vs-task
   difference (which also includes eye/hand movement and arousal), not a calibrated workload scale.
4. **The hybrid does not beat XGBoost on 48 subjects.** Reasons, in order of likely importance: (a) **data size**
   — ≈ 38 training subjects per fold is very small for a 141 k-parameter model that must learn its own filters,
   whereas XGBoost starts from 1,499 engineered features that already encode decades of EEG knowledge;
   (b) **inductive bias** — band power and PLV are exactly the information that separates T1, and the trees use
   it directly; the network must discover it from raw signals under strong between-subject variability;
   (c) **training budget/tuning** — CPU-only training, ≤ 25 epochs, untuned hyper-parameters, kernel 32, and
   early stopping on ≈ 6 validation subjects (best epoch 0 in T1 fold 1 for every deep model, §6.5); (d) the
   deep models never see the rest baseline or engineered connectivity features except through the GAT's mean-PLV
   graph. EEGNet (1.6 k parameters) doing as well as the 141 k-parameter hybrid supports the small-data explanation.
5. **Calibration** is partially successful; with tiny validation sets it can also hurt.

## 7.8 Comparison with published STEW results
The repository does **not** cite any published STEW classification results, so no comparison numbers are
given here (NOT AVAILABLE). What should be looked up before the final report:
* the original STEW paper (Lim, Sourina & Wang, 2018, IEEE TNSRE) — its reported rest-vs-task and
  rating-based accuracies and **its evaluation protocol**;
* later papers that report STEW accuracies (often > 90 %) — check whether they use **random epoch splits or
  within-subject splits** (which our subject-identity probe shows to be heavily optimistic) vs subject-independent
  evaluation like ours;
* any subject-independent (LOSO) STEW results for a like-for-like comparison.
