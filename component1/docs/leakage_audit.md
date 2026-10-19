# Leakage audit

What was checked, how, and the outcome. The automated checks are re-runnable with
`python scripts/audit_leakage.py` (output: `results/leakage_audit.json`); numbers below are
copied from that file and from the experiments' `metrics.json`.

## 1. Automated checks

| Check | How | Result |
|---|---|---|
| Fold integrity | For all 58 saved experiments, every out-of-fold prediction's fold is compared with its subject: LOSO fold id must equal the test subject; 5-fold runs must match `results/folds_5fold.json` | **58 / 58 OK**, 0 problem rows |
| Label-permutation test | The full LOSO pipeline (scaler + logistic regression, all 1,499 features, T1) re-run 3× with labels shuffled within each subject | balanced accuracy **0.498, 0.508, 0.512** (mean 0.506), AUROC 0.500–0.507 → chance, as it must be |
| Feature locality | 5 epochs re-extracted one at a time vs. the batch feature table | max abs difference **0.0** → features use only the epoch itself |
| Subject fingerprint (why splits must be by subject) | Logistic regression predicting *which subject* an epoch is from, with a random (not subject-wise) 5-fold split | **99.5 %** accuracy (chance 2.1 %). Random epoch splits would let a model recognise the person; every reported result uses subject-wise splits |
| Results above 97 % | All experiments scanned (pooled, per-recording and window views) | 2, both T1 + rest baseline, per recording — audited in section 3 |

## 2. Checks built into the code (run on every experiment)
* `training/cv.py::assert_no_subject_overlap` on every outer split, and in the deep trainer also
  between fit / validation / test subjects. Tested in `tests/test_models_cv.py`.
* Scaler, SelectKBest and PCA are steps of the sklearn `Pipeline`, so they are fit on the training
  fold only (`tests/test_models_cv.py::test_scaler_is_fit_inside_pipeline`).
* Deep models: channel-scale and node-feature scalers are fit on the fit subjects of each fold;
  early stopping and temperature scaling use a subject-wise inner validation split, never the test fold.
* Hyper-parameters were fixed before evaluation (D12); the best classical model (for the ablation)
  is chosen by a rule fixed in advance (mean rank over T1 and T3), not by peeking at one fold.
* Overlapping epochs (50 %) cannot leak because whole subjects are held out.

## 3. The two scores above 97 % — audited
| Experiment | View | Balanced accuracy |
|---|---|---|
| `cls_T1_svm_all_rest_baseline_loso` | per recording | 0.990 |
| `cls_T1_rf_all_rest_baseline_loso` | per recording | 0.979 |

Both use the **subject rest baseline** and both are *recording-level* views (the 74 epoch scores of a
recording averaged into one decision). Diagnosis:
* Not cross-subject leakage: the same pipelines without the rest baseline score 0.906 / 0.938
  recording macro-F1, the permutation test is at chance, and fold integrity holds.
* The rest baseline z-scores every feature with the statistics of the subject's own rest
  recording. By construction the rest recording's *average* feature vector is then exactly 0, while
  the task recording's average is not. Averaging a recording's epochs therefore turns T1 into
  "is this the subject's reference recording?", which is nearly trivial. This is a within-subject
  reference effect of the normalisation — the inflation anticipated in the methodology — not a bug.
* Consequence: rest-baseline numbers are reported separately (`results/summary_classical.md`,
  "Norm = rest_baseline") and are never used as headline results. All headline and comparison
  tables use `normalisation = none`. In a deployment the baseline would be recorded *before* the
  session, so its value would come from a separate recording, not from one of the two classes.

## 4. Other things considered
* **T2 confound** (labels ≈ rest/task) — documented in `docs/labels.md`; T3 used as the
  confound-free task. Not leakage, but a validity threat.
* **Below-chance T3 under LOSO** (D16) — the expected LOSO "anti-learning" effect with one class per
  subject, consistent with no learnable signal; not leakage.
* **Interpolation and quality flags** are computed per recording from that recording only.
* **ICA** (optional, off) would be fit per recording, unsupervised.
