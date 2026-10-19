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
<!-- TABLE:master -->

Headline comparison (same 5 folds, same 18-s windows, no rest baseline):

![Classical vs deep](figures/summary_classical_vs_deep.png)

*Figure 7.1 — Classical (blue) vs deep (orange) models on identical folds and windows (generated from the result files).*

<!-- TABLE:cls_kfold5 -->

## 7.3 Rest-baseline inflation
<!-- TABLE:rest_inflation -->

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
<!-- TABLE:confusions -->

![CM XGBoost T1 LOSO](figures/cm_cls_T1_xgb_all_none_loso.png)

*Figure 7.3 — XGBoost T1 LOSO: more rest epochs are predicted as task (588) than task epochs as rest (368)
(`results/cls_T1_xgb_all_none_loso/metrics.json` → `confusion_pooled`); per recording only 8 of 96 are wrong.*

![CM hybrid T1](figures/cm_deep_T1_A10_hybrid_kfold5.png)

*Figure 7.4 — Full hybrid T1: more task windows are classified as rest than vice versa.*

![CM XGBoost T3](figures/cm_cls_T3_xgb_all_none_loso.png)

*Figure 7.5 — XGBoost T3 LOSO: near-random confusion.*

## 7.5 Leakage audit
Re-runnable with `scripts/audit_leakage.py`; written up in `docs/leakage_audit.md`.
<!-- TABLE:leakage -->

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
