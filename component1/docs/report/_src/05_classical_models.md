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
<!-- TABLE:cls_baselines -->

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
<!-- TABLE:ablation_T1 -->

**T3**
<!-- TABLE:ablation_T3 -->

**T2**
<!-- TABLE:ablation_T2 -->

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

<!-- TABLE:stats_classical -->

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
