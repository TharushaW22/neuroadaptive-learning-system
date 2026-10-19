# 5. Classical models

## 5.1 The four models
All four are scikit-learn `Pipeline`s built by `src/eegengine/models/classical.py:make_pipeline`:
`StandardScaler → (optional SelectKBest(f_classif) or PCA) → classifier`. Selection/PCA is implemented
but **not used** in any reported run (`configs/experiments/classical.yaml` → `selection: null`).
Hyper-parameters are fixed in advance — no nested tuning (`docs/decisions.md` D12).

| Model | WHAT | WHY chosen | Exact configuration (`classical.py`) | Class imbalance |
|---|---|---|---|---|
| Logistic Regression (LR) | linear model `p(y=1|x) = σ(wᵀx + b)`, L2 penalty | simplest interpretable baseline; tests whether a linear boundary suffices | `C=0.1`, `max_iter=3000`, lbfgs | `class_weight="balanced"` |
| SVM, RBF kernel | max-margin classifier with kernel `K(x,x′) = exp(−γ‖x−x′‖²)` | strong non-linear baseline, standard in EEG workload papers | `C=1.0`, `gamma="scale"`, **no Platt scaling** (D13) | `class_weight="balanced"` |
| Random Forest (RF) | bagged ensemble of decision trees with random feature subsets | robust to many noisy/correlated features, little tuning | 200 trees, `max_features="sqrt"`, `min_samples_leaf=2`, `max_samples=0.5` | `class_weight="balanced_subsample"` |
| XGBoost | gradient-boosted trees: each tree fits the gradient of the log-loss of the current ensemble | typically the strongest model on tabular features | 200 trees, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.3`, `max_bin=64`, `tree_method="hist"` | balanced `sample_weight` (`fit_pipeline`) |

* **Why StandardScaler inside the pipeline:** LR/SVM need features on comparable scales; because the
  scaler is a pipeline step it is fit on the training subjects of each fold only (tested:
  `tests/test_models_cv.py:test_scaler_is_fit_inside_pipeline`).
* **Scores:** `classical.py:predict_scores` returns `predict_proba` for LR/RF/XGB; the SVM's decision
  value is squashed with a sigmoid/softmax — valid for ranking (AUROC, averaging) but **not calibrated
  probabilities** (D13).
* **Missing classes in a fold:** `training/classical_runner.py:FoldModel` maps a fold model trained on only
  some classes back to all classes; XGBoost requires contiguous labels, so labels are re-encoded per fold.
* **DEVIATION FROM PROPOSAL (D12):** RF and XGBoost were reduced from 300 to 200 trees (RF bootstrap 50 %;
  XGB 64 bins, 30 % columns) because 48-fold LOSO on the shared 4-core laptop was too slow. On 6 held-out
  subjects T1 AUROC was unchanged (RF 0.932 vs 0.935, XGB 0.951 vs 0.949 — measured during development and
  recorded in `docs/decisions.md` D12; no result file holds these two checks).

## 5.2 Cross-validation design
Implemented in `src/eegengine/training/cv.py`; every split passes `assert_no_subject_overlap`.

* **LOSO (leave-one-subject-out)** — `cv.py:loso_splits`: 48 folds for T1 (45 for T2/T3), each fold tests
  one unseen person and trains on all others. **Why:** the engine is for *new* learners. EEG is highly
  person-specific: with a *random* epoch split, logistic regression identifies **which subject** an epoch
  comes from with accuracy 0.995 [`results/leakage_audit.json` → `subject_identity_probe.random_split_subject_id_accuracy`]
  (chance 0.021 [`results/leakage_audit.json` → `subject_identity_probe.chance`]). With overlapping 4-s epochs, a random split
  would also put almost identical neighbouring windows into train and test. Both make random splits wildly
  optimistic; LOSO avoids both.
* **Fixed 5-fold by subject** — `cv.py:subject_fold_assignment` shuffles the 48 subjects once (seed 42) and
  deals them round-robin into 5 folds (`results/folds_5fold.json`); `cv.py:kfold_splits` reuses this exact
  assignment for **every** deep and classical model and every task (a task skips subjects it has no labels
  for). **Why:** deep training ×48 folds is unaffordable on a CPU; using the same folds for all models makes
  the classical-vs-deep comparison paired. This is the "GroupKFold" of the brief, implemented with a fixed,
  saved assignment.
* **Inner validation split** (deep models only) — `cv.py:inner_validation_split`: 15 % of the *training*
  subjects, re-drawn deterministically until it contains every class.
* **18-s window aggregation** — for the 5-fold classical runs (`classical_runner.py:run_classical`,
  `cv == "kfold5"`), the epoch scores are averaged over the **same windows of 8 consecutive epochs (stride 2
  epochs, 18 s) that the deep sequence models are scored on** (`training/sequences.py:strided_items`).
  **Why:** the sequence models see 18 s of EEG per decision; scoring the classical models per 4-s epoch would
  compare unequal amounts of information. The `sequence_level` view makes the final comparison fair.
* **Three metric views** (`evaluation/metrics.py`): *per fold* (mean ± std over folds where the metric is
  defined), *pooled* (all out-of-fold predictions at once) and *per recording* (each recording's epoch
  probabilities averaged into one decision; for T3 this is one decision per subject).
* **T3 special case:** each subject has exactly one T3 class, so LOSO test folds contain one class and
  per-fold macro-F1/AUROC are undefined; T3 LOSO is read from the pooled and per-recording views.

## 5.3 Baseline results (LOSO, all 1,499 features)
<!-- BEGIN TABLE:cls_baselines -->
| Task | Model | Norm | Fold macro-F1 (mean ± std) | Fold bal. acc | Pooled macro-F1 | Pooled bal. acc | Pooled AUROC | Recording macro-F1 | Source: results/<id>/metrics.json |
|---|---|---|---|---|---|---|---|---|---|
| T1 | Logistic Regression | none | 0.836 ± 0.134 | 0.845 ± 0.118 | 0.845 | 0.845 | 0.918 | 0.948 | `cls_T1_lr_all_none_loso` |
| T1 | SVM (RBF) | none | 0.850 ± 0.148 | 0.860 ± 0.130 | 0.859 | 0.860 | 0.938 | 0.906 | `cls_T1_svm_all_none_loso` |
| T1 | Random Forest | none | 0.844 ± 0.140 | 0.851 ± 0.128 | 0.851 | 0.851 | 0.926 | 0.938 | `cls_T1_rf_all_none_loso` |
| T1 | XGBoost | none | 0.859 ± 0.139 | 0.865 ± 0.127 | 0.865 | 0.865 | 0.941 | 0.917 | `cls_T1_xgb_all_none_loso` |
| T1 | Logistic Regression | rest_baseline | 0.882 ± 0.142 | 0.892 ± 0.121 | 0.891 | 0.892 | 0.959 | 0.927 | `cls_T1_lr_all_rest_baseline_loso` |
| T1 | SVM (RBF) | rest_baseline | 0.948 ± 0.101 | 0.951 ± 0.083 | 0.951 | 0.951 | 0.990 | 0.990 | `cls_T1_svm_all_rest_baseline_loso` |
| T1 | Random Forest | rest_baseline | 0.922 ± 0.109 | 0.926 ± 0.092 | 0.926 | 0.926 | 0.977 | 0.979 | `cls_T1_rf_all_rest_baseline_loso` |
| T1 | XGBoost | rest_baseline | 0.918 ± 0.128 | 0.924 ± 0.108 | 0.924 | 0.924 | 0.986 | 0.958 | `cls_T1_xgb_all_rest_baseline_loso` |
| T2 | Logistic Regression | none | 0.417 ± 0.126 | 0.588 ± 0.192 | 0.537 | 0.538 | 0.738 | 0.568 | `cls_T2_lr_all_none_loso` |
| T2 | SVM (RBF) | none | 0.421 ± 0.136 | 0.601 ± 0.212 | 0.544 | 0.546 | 0.726 | 0.572 | `cls_T2_svm_all_none_loso` |
| T2 | Random Forest | none | 0.427 ± 0.139 | 0.631 ± 0.203 | 0.555 | 0.563 | 0.768 | 0.629 | `cls_T2_rf_all_none_loso` |
| T2 | XGBoost | none | 0.436 ± 0.134 | 0.623 ± 0.209 | 0.565 | 0.567 | 0.770 | 0.618 | `cls_T2_xgb_all_none_loso` |
| T2 | XGBoost | rest_baseline | 0.439 ± 0.152 | 0.650 ± 0.211 | 0.555 | 0.563 | 0.821 | 0.538 | `cls_T2_xgb_all_rest_baseline_loso` |
| T3 | Logistic Regression | none | — | — | 0.501 | 0.501 | 0.509 | 0.472 | `cls_T3_lr_all_none_loso` |
| T3 | SVM (RBF) | none | — | — | 0.481 | 0.482 | 0.485 | 0.434 | `cls_T3_svm_all_none_loso` |
| T3 | Random Forest | none | — | — | 0.527 | 0.529 | 0.536 | 0.556 | `cls_T3_rf_all_none_loso` |
| T3 | XGBoost | none | — | — | 0.538 | 0.539 | 0.544 | 0.545 | `cls_T3_xgb_all_none_loso` |
| T3 | XGBoost | rest_baseline | — | — | 0.390 | 0.392 | 0.356 | 0.380 | `cls_T3_xgb_all_rest_baseline_loso` |
<!-- END TABLE:cls_baselines -->

Reading the table: on **T1** all four models are close (pooled macro-F1 0.845–0.865); XGBoost is best and
was chosen as the "best classical model" by a rule fixed in advance — lowest mean rank of pooled macro-F1
over T1 and T3, without rest baseline (`scripts/run_ablation.py:pick_best_model`; ranks in
`results/stats_classical.json` → `selection.mean_rank`: XGBoost 1.0 [`results/stats_classical.json` → `selection.mean_rank.xgb`]).
On **T3** all models are at chance. On **T2** fold macro-F1 is ≈ 0.42–0.44 while AUROC is 0.73–0.77,
because most of what is learnable is the rest-vs-task difference (rest ≈ "low").

## 5.4 Feature-family ablation (A1–A5, XGBoost, LOSO)
Design (`training/classical_runner.py:FEATURE_SETS`, `scripts/run_ablation.py`): A1 spectral; A2 spectral +
entropy; A3 spectral + wavelet; A4 spectral + connectivity (PLV + coherence); A5 spectral + connectivity +
graph; plus "all". Each run for both normalisations. Scope (D15): T1 and T3 fully; T2 only "all" with rest baseline.

**T1**
<!-- BEGIN TABLE:ablation_T1 -->
| Feature set | Norm | #features | Fold macro-F1 | Pooled macro-F1 | Pooled AUROC | Recording macro-F1 | Source |
|---|---|---|---|---|---|---|---|
| A1_spectral | none | 185 | 0.802 ± 0.137 | 0.810 | 0.890 | 0.895 | `cls_T1_xgb_A1_spectral_none_loso` |
| A2_spectral_entropy | none | 269 | 0.819 ± 0.128 | 0.826 | 0.904 | 0.927 | `cls_T1_xgb_A2_spectral_entropy_none_loso` |
| A3_spectral_wavelet | none | 451 | 0.812 ± 0.147 | 0.822 | 0.897 | 0.895 | `cls_T1_xgb_A3_spectral_wavelet_none_loso` |
| A4_spectral_connectivity | none | 913 | 0.860 ± 0.132 | 0.866 | 0.938 | 0.927 | `cls_T1_xgb_A4_spectral_connectivity_none_loso` |
| A5_spectral_connectivity_graph | none | 1149 | 0.865 ± 0.123 | 0.870 | 0.940 | 0.937 | `cls_T1_xgb_A5_spectral_connectivity_graph_none_loso` |
| all | none | 1499 | 0.859 ± 0.139 | 0.865 | 0.941 | 0.917 | `cls_T1_xgb_all_none_loso` |
| A1_spectral | rest_baseline | 185 | 0.885 ± 0.133 | 0.892 | 0.969 | 0.958 | `cls_T1_xgb_A1_spectral_rest_baseline_loso` |
| A2_spectral_entropy | rest_baseline | 269 | 0.887 ± 0.127 | 0.893 | 0.971 | 0.948 | `cls_T1_xgb_A2_spectral_entropy_rest_baseline_loso` |
| A3_spectral_wavelet | rest_baseline | 451 | 0.879 ± 0.155 | 0.889 | 0.973 | 0.927 | `cls_T1_xgb_A3_spectral_wavelet_rest_baseline_loso` |
| A4_spectral_connectivity | rest_baseline | 913 | 0.924 ± 0.114 | 0.929 | 0.987 | 0.969 | `cls_T1_xgb_A4_spectral_connectivity_rest_baseline_loso` |
| A5_spectral_connectivity_graph | rest_baseline | 1149 | 0.918 ± 0.127 | 0.923 | 0.987 | 0.958 | `cls_T1_xgb_A5_spectral_connectivity_graph_rest_baseline_loso` |
| all | rest_baseline | 1499 | 0.918 ± 0.128 | 0.924 | 0.986 | 0.958 | `cls_T1_xgb_all_rest_baseline_loso` |
<!-- END TABLE:ablation_T1 -->

**T3**
<!-- BEGIN TABLE:ablation_T3 -->
| Feature set | Norm | #features | Fold macro-F1 | Pooled macro-F1 | Pooled AUROC | Recording macro-F1 | Source |
|---|---|---|---|---|---|---|---|
| A1_spectral | none | 185 | — | 0.481 | 0.474 | 0.440 | `cls_T3_xgb_A1_spectral_none_loso` |
| A2_spectral_entropy | none | 269 | — | 0.465 | 0.450 | 0.440 | `cls_T3_xgb_A2_spectral_entropy_none_loso` |
| A3_spectral_wavelet | none | 451 | — | 0.481 | 0.464 | 0.464 | `cls_T3_xgb_A3_spectral_wavelet_none_loso` |
| A4_spectral_connectivity | none | 913 | — | 0.542 | 0.562 | 0.564 | `cls_T3_xgb_A4_spectral_connectivity_none_loso` |
| A5_spectral_connectivity_graph | none | 1149 | — | 0.534 | 0.552 | 0.570 | `cls_T3_xgb_A5_spectral_connectivity_graph_none_loso` |
| all | none | 1499 | — | 0.538 | 0.544 | 0.545 | `cls_T3_xgb_all_none_loso` |
| A1_spectral | rest_baseline | 185 | — | 0.518 | 0.541 | 0.530 | `cls_T3_xgb_A1_spectral_rest_baseline_loso` |
| A2_spectral_entropy | rest_baseline | 269 | — | 0.512 | 0.530 | 0.460 | `cls_T3_xgb_A2_spectral_entropy_rest_baseline_loso` |
| A3_spectral_wavelet | rest_baseline | 451 | — | 0.512 | 0.528 | 0.460 | `cls_T3_xgb_A3_spectral_wavelet_rest_baseline_loso` |
| A4_spectral_connectivity | rest_baseline | 913 | — | 0.513 | 0.512 | 0.453 | `cls_T3_xgb_A4_spectral_connectivity_rest_baseline_loso` |
| A5_spectral_connectivity_graph | rest_baseline | 1149 | — | 0.384 | 0.350 | 0.415 | `cls_T3_xgb_A5_spectral_connectivity_graph_rest_baseline_loso` |
| all | rest_baseline | 1499 | — | 0.390 | 0.356 | 0.380 | `cls_T3_xgb_all_rest_baseline_loso` |
<!-- END TABLE:ablation_T3 -->

**T2**
<!-- BEGIN TABLE:ablation_T2 -->
| Feature set | Norm | #features | Fold macro-F1 | Pooled macro-F1 | Pooled AUROC | Recording macro-F1 | Source |
|---|---|---|---|---|---|---|---|
| all | none | 1499 | 0.436 ± 0.134 | 0.565 | 0.770 | 0.618 | `cls_T2_xgb_all_none_loso` |
| all | rest_baseline | 1499 | 0.439 ± 0.152 | 0.555 | 0.821 | 0.538 | `cls_T2_xgb_all_rest_baseline_loso` |
<!-- END TABLE:ablation_T2 -->

![Feature ablation](figures/summary_feature_ablation.png)

*Figure 5.1 — Feature ablation (generated from the result files by `scripts/build_report.py`).*

Findings: connectivity is the only family that significantly improves T1 over spectral alone
(A4 p = 5.4e-05 [`results/stats_classical.json` → `tests.2.p_value`]); entropy gives a smaller gain
(p = 0.0028 [`results/stats_classical.json` → `tests.0.p_value`]); wavelets add nothing significant
(p = 0.107 [`results/stats_classical.json` → `tests.1.p_value`]); graph metrics add nothing beyond PLV.
On T3 no feature set differs significantly from A1 without the rest baseline (all Holm p ≥ 0.76).

## 5.5 Statistical tests
**Wilcoxon signed-rank test** (`src/eegengine/evaluation/ablation.py:paired_wilcoxon`, scipy): compares two
configurations on the **same test subjects**, using one score per subject (macro-F1 for T1/T2; accuracy for
T3 because each subject has one class). **Why this test:** the scores are paired (same subjects), there are
only 45–48 of them, and they are bounded and often skewed, so a non-parametric paired test is appropriate;
a paired t-test would assume normal differences. **Multiple comparisons:** Holm–Bonferroni correction within
each group of related tests (`ablation.py:holm`).

<!-- BEGIN TABLE:stats_classical -->
| # | Group | A | B | Per-subject metric | n | mean A | mean B | A better / B better / ties | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ablation_vs_A1/T1/none | cls_T1_xgb_A2_spectral_entropy_none_loso | cls_T1_xgb_A1_spectral_none_loso | macro_f1 | 48 | 0.8192 | 0.8017 | 30/14/4 | 0.003 | 0.006 |
| 1 | ablation_vs_A1/T1/none | cls_T1_xgb_A3_spectral_wavelet_none_loso | cls_T1_xgb_A1_spectral_none_loso | macro_f1 | 48 | 0.8123 | 0.8017 | 29/19/0 | 0.107 | 0.107 |
| 2 | ablation_vs_A1/T1/none | cls_T1_xgb_A4_spectral_connectivity_none_loso | cls_T1_xgb_A1_spectral_none_loso | macro_f1 | 48 | 0.8601 | 0.8017 | 35/13/0 | 5.43e-05 | 1.68e-04 |
| 3 | ablation_vs_A1/T1/none | cls_T1_xgb_A5_spectral_connectivity_graph_none_loso | cls_T1_xgb_A1_spectral_none_loso | macro_f1 | 48 | 0.8653 | 0.8017 | 38/9/1 | 3.35e-05 | 1.68e-04 |
| 4 | ablation_vs_A1/T1/none | cls_T1_xgb_all_none_loso | cls_T1_xgb_A1_spectral_none_loso | macro_f1 | 48 | 0.8586 | 0.8017 | 38/9/1 | 3.85e-05 | 1.68e-04 |
| 5 | ablation_vs_A1/T1/rest_baseline | cls_T1_xgb_A2_spectral_entropy_rest_baseline_loso | cls_T1_xgb_A1_spectral_rest_baseline_loso | macro_f1 | 48 | 0.8868 | 0.8854 | 22/22/4 | 0.875 | 1.000 |
| 6 | ablation_vs_A1/T1/rest_baseline | cls_T1_xgb_A3_spectral_wavelet_rest_baseline_loso | cls_T1_xgb_A1_spectral_rest_baseline_loso | macro_f1 | 48 | 0.8791 | 0.8854 | 29/17/2 | 0.840 | 1.000 |
| 7 | ablation_vs_A1/T1/rest_baseline | cls_T1_xgb_A4_spectral_connectivity_rest_baseline_loso | cls_T1_xgb_A1_spectral_rest_baseline_loso | macro_f1 | 48 | 0.9245 | 0.8854 | 37/10/1 | 4.42e-04 | 0.002 |
| 8 | ablation_vs_A1/T1/rest_baseline | cls_T1_xgb_A5_spectral_connectivity_graph_rest_baseline_loso | cls_T1_xgb_A1_spectral_rest_baseline_loso | macro_f1 | 48 | 0.9175 | 0.8854 | 37/10/1 | 0.005 | 0.015 |
| 9 | ablation_vs_A1/T1/rest_baseline | cls_T1_xgb_all_rest_baseline_loso | cls_T1_xgb_A1_spectral_rest_baseline_loso | macro_f1 | 48 | 0.9181 | 0.8854 | 36/10/2 | 0.002 | 0.008 |
| 10 | rest_baseline_vs_none/T1 | cls_T1_lr_all_rest_baseline_loso | cls_T1_lr_all_none_loso | macro_f1 | 48 | 0.8824 | 0.8359 | 34/14/0 | 0.008 | 0.008 |
| 11 | best_vs_other_model/T1/none | cls_T1_xgb_all_none_loso | cls_T1_lr_all_none_loso | macro_f1 | 48 | 0.8586 | 0.8359 | 34/12/2 | 0.006 | 0.017 |
| 12 | rest_baseline_vs_none/T1 | cls_T1_svm_all_rest_baseline_loso | cls_T1_svm_all_none_loso | macro_f1 | 48 | 0.9475 | 0.8502 | 40/6/2 | 2.56e-06 | 1.02e-05 |
| 13 | best_vs_other_model/T1/none | cls_T1_xgb_all_none_loso | cls_T1_svm_all_none_loso | macro_f1 | 48 | 0.8586 | 0.8502 | 26/20/2 | 0.287 | 0.287 |
| 14 | rest_baseline_vs_none/T1 | cls_T1_rf_all_rest_baseline_loso | cls_T1_rf_all_none_loso | macro_f1 | 48 | 0.9221 | 0.8444 | 39/8/1 | 3.57e-06 | 1.07e-05 |
| 15 | best_vs_other_model/T1/none | cls_T1_xgb_all_none_loso | cls_T1_rf_all_none_loso | macro_f1 | 48 | 0.8586 | 0.8444 | 30/15/3 | 0.016 | 0.033 |
| 16 | rest_baseline_vs_none/T1 | cls_T1_xgb_all_rest_baseline_loso | cls_T1_xgb_all_none_loso | macro_f1 | 48 | 0.9181 | 0.8586 | 35/11/2 | 0.001 | 0.002 |
| 17 | ablation_vs_A1/T3/none | cls_T3_xgb_A2_spectral_entropy_none_loso | cls_T3_xgb_A1_spectral_none_loso | accuracy | 45 | 0.4712 | 0.4844 | 20/21/4 | 0.546 | 1.000 |
| 18 | ablation_vs_A1/T3/none | cls_T3_xgb_A3_spectral_wavelet_none_loso | cls_T3_xgb_A1_spectral_none_loso | accuracy | 45 | 0.4847 | 0.4844 | 15/21/9 | 1.000 | 1.000 |
| 19 | ablation_vs_A1/T3/none | cls_T3_xgb_A4_spectral_connectivity_none_loso | cls_T3_xgb_A1_spectral_none_loso | accuracy | 45 | 0.5508 | 0.4844 | 24/20/1 | 0.229 | 0.917 |
| 20 | ablation_vs_A1/T3/none | cls_T3_xgb_A5_spectral_connectivity_graph_none_loso | cls_T3_xgb_A1_spectral_none_loso | accuracy | 45 | 0.5420 | 0.4844 | 21/23/1 | 0.378 | 1.000 |
| 21 | ablation_vs_A1/T3/none | cls_T3_xgb_all_none_loso | cls_T3_xgb_A1_spectral_none_loso | accuracy | 45 | 0.5468 | 0.4844 | 23/20/2 | 0.152 | 0.762 |
| 22 | ablation_vs_A1/T3/rest_baseline | cls_T3_xgb_A2_spectral_entropy_rest_baseline_loso | cls_T3_xgb_A1_spectral_rest_baseline_loso | accuracy | 45 | 0.5165 | 0.5225 | 21/19/5 | 0.867 | 1.000 |
| 23 | ablation_vs_A1/T3/rest_baseline | cls_T3_xgb_A3_spectral_wavelet_rest_baseline_loso | cls_T3_xgb_A1_spectral_rest_baseline_loso | accuracy | 45 | 0.5165 | 0.5225 | 21/23/1 | 0.591 | 1.000 |
| 24 | ablation_vs_A1/T3/rest_baseline | cls_T3_xgb_A4_spectral_connectivity_rest_baseline_loso | cls_T3_xgb_A1_spectral_rest_baseline_loso | accuracy | 45 | 0.5195 | 0.5225 | 16/24/5 | 0.432 | 1.000 |
| 25 | ablation_vs_A1/T3/rest_baseline | cls_T3_xgb_A5_spectral_connectivity_graph_rest_baseline_loso | cls_T3_xgb_A1_spectral_rest_baseline_loso | accuracy | 45 | 0.4015 | 0.5225 | 13/29/3 | 0.013 | 0.051 |
| 26 | ablation_vs_A1/T3/rest_baseline | cls_T3_xgb_all_rest_baseline_loso | cls_T3_xgb_A1_spectral_rest_baseline_loso | accuracy | 45 | 0.4039 | 0.5225 | 13/30/2 | 0.007 | 0.035 |
| 27 | best_vs_other_model/T3/none | cls_T3_xgb_all_none_loso | cls_T3_lr_all_none_loso | accuracy | 45 | 0.5468 | 0.5075 | 21/22/2 | 0.340 | 0.680 |
| 28 | best_vs_other_model/T3/none | cls_T3_xgb_all_none_loso | cls_T3_svm_all_none_loso | accuracy | 45 | 0.5468 | 0.4928 | 26/14/5 | 0.025 | 0.074 |
| 29 | best_vs_other_model/T3/none | cls_T3_xgb_all_none_loso | cls_T3_rf_all_none_loso | accuracy | 45 | 0.5468 | 0.5438 | 15/23/7 | 0.862 | 0.862 |
| 30 | rest_baseline_vs_none/T3 | cls_T3_xgb_all_rest_baseline_loso | cls_T3_xgb_all_none_loso | accuracy | 45 | 0.4039 | 0.5468 | 13/32/0 | 0.010 | 0.010 |
| 31 | best_vs_other_model/T2/none | cls_T2_xgb_all_none_loso | cls_T2_lr_all_none_loso | macro_f1 | 45 | 0.4624 | 0.4298 | 26/19/0 | 0.120 | 0.240 |
| 32 | best_vs_other_model/T2/none | cls_T2_xgb_all_none_loso | cls_T2_svm_all_none_loso | macro_f1 | 45 | 0.4624 | 0.4358 | 31/13/1 | 0.009 | 0.026 |
| 33 | best_vs_other_model/T2/none | cls_T2_xgb_all_none_loso | cls_T2_rf_all_none_loso | macro_f1 | 45 | 0.4624 | 0.4508 | 25/20/0 | 0.414 | 0.414 |
| 34 | rest_baseline_vs_none/T2 | cls_T2_xgb_all_rest_baseline_loso | cls_T2_xgb_all_none_loso | macro_f1 | 45 | 0.4492 | 0.4624 | 18/26/1 | 0.608 | 0.608 |

Source: `results/stats_classical.json` (row # = index in `tests`)
<!-- END TABLE:stats_classical -->

## 5.6 Additional analyses requested in the brief
| Analysis | Status |
|---|---|
| Label-permutation test | **Done** for the pipeline (LR, all features, T1, 3 permutations): balanced accuracy 0.506 [`results/leakage_audit.json` → `permutation_test.mean_balanced_accuracy`] (§7.5) |
| Subject-level (per-recording) T3 predictions | **Done** — `recording_level` view; e.g. XGBoost T3 recording macro-F1 0.545 [`results/cls_T3_xgb_all_none_loso/metrics.json` → `recording_level.macro_f1`] |
| Bootstrap confidence intervals | **Not done** — NOT AVAILABLE |
| Spearman correlation of predictions with the 1–9 rating | **Not done** — NOT AVAILABLE |
| Nested hyper-parameter tuning | **Not done** (D12) |

## 5.7 Limitations
Fixed hyper-parameters; SVM scores uncalibrated; LOSO estimates have high variance per subject
(fold std ≈ 0.13–0.15 on T1); the reduced grid (D14, D15) means T2/T3 rest-baseline comparisons exist only
for XGBoost; feature selection/PCA was not explored.
