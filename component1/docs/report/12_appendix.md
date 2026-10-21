# 12. Appendix

## 12.1 All configuration files (verbatim)
<!-- BEGIN TABLE:configs -->
### `configs/experiments/classical.yaml`

```yaml
# Phase 4 classical baselines: every (task x model) on the full feature set, LOSO, no rest baseline.
# The rest-baseline variants are in classical_rest.yaml (T1, all models) and in the ablation (best
# model, all tasks) — see docs/decisions.md D14.
# One results/<experiment_id>/ folder per run, id = cls_<T>_<model>_<feature_set>_<norm>_<cv>.
features_config: features.yaml
seed: 42
cv: loso
tasks: [workload_binary, workload_within_task, workload_level]
models: [lr, svm, rf, xgb]
feature_sets: [all]
normalisations: [none]
selection: null              # or {method: kbest, k: 200} / {method: pca, n_components: 0.95}
exclude_bad_windows: false
n_jobs: 4                    # = physical cores; 8 caused memory thrashing on a 16 GB laptop
```

### `configs/experiments/classical_kfold5.yaml`

```yaml
# Classical models on the SAME 5 subject folds as the deep models (results/folds_5fold.json), so the
# final comparison table is paired. Also scored on the same 8-epoch sequences (sequence_level).
features_config: features.yaml
seed: 42
cv: kfold5
tasks: [workload_binary, workload_within_task]
models: [lr, svm, rf, xgb]
feature_sets: [all]
normalisations: [none]
selection: null
exclude_bad_windows: false
n_jobs: 4
```

### `configs/experiments/classical_rest.yaml`

```yaml
# Phase 4: T1 with the subject rest baseline for every model (the case where it can inflate scores).
# The rest-baseline variants are in classical_rest.yaml (T1, all models) and in the ablation (best
# model, all tasks) — see docs/decisions.md D14.
# One results/<experiment_id>/ folder per run, id = cls_<T>_<model>_<feature_set>_<norm>_<cv>.
features_config: features.yaml
seed: 42
cv: loso
tasks: [workload_binary]
models: [lr, svm, rf, xgb]
feature_sets: [all]
normalisations: [rest_baseline]
selection: null              # or {method: kbest, k: 200} / {method: pca, n_components: 0.95}
exclude_bad_windows: false
n_jobs: 4                    # = physical cores; 8 caused memory thrashing on a 16 GB laptop
```

### `configs/experiments/deep.yaml`

```yaml
# Phase 5 deep models. Same fixed 5 subject folds (results/folds_5fold.json) for every model.
features_config: features.yaml
seed: 42
tasks: [workload_binary, workload_within_task]
folds_file: results/folds_5fold.json

# A6-A10 = architecture ablation of the hybrid; eegnet = published baseline.
models:
  eegnet:          {type: eegnet}
  A6_cnn:          {type: hybrid, branches: [cnn], ablation_id: A6}
  A7_cnn_lstm:     {type: hybrid, branches: [cnn, lstm], ablation_id: A7}
  A8_cnn_lstm_trans: {type: hybrid, branches: [cnn, lstm, transformer], ablation_id: A8}
  A9_cnn_lstm_gnn: {type: hybrid, branches: [cnn, lstm, gnn], ablation_id: A9}
  A10_hybrid:      {type: hybrid, branches: [cnn, lstm, transformer, gnn], ablation_id: A10}

input:
  edge_source: plv          # plv | coh — edge weights of the GAT graph
  edge_band: mean           # mean of theta/alpha/beta/gamma, or a band index 0-3

model:
  emb_dim: 64
  cnn: {F1: 8, D: 2, F2: 16, kernel: 32, dropout: 0.25}
  lstm_hidden: 64
  trans_heads: 4
  trans_layers: 2
  trans_ff: 128
  gnn: {hidden: 16, out_dim: 32, heads: 4, dropout: 0.1}
  fusion_dim: 64
  dropout: 0.3

train:
  seq_len: 8                       # 8 epochs = 18 s
  test_stride: 2                   # evaluation sequences (shared with the classical 5-fold runs)
  val_stride: 4
  epoch_items_per_recording: 8     # fresh random sequence starts per recording per epoch
  epoch_model_multiplier: 4        # EEGNet sees 4x as many (single-epoch) items per epoch
  batch_size: 16
  max_epochs: 25
  patience: 5
  lr: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
  val_frac: 0.15                   # inner validation subjects (early stopping + temperature)
  clip: 20.0
  threads: 4                       # physical cores of the dev laptop

augment:
  noise_std: 0.1                   # in units of the channel scale
  channel_dropout: 0.1
  max_shift: 16                    # samples (125 ms)
```

### `configs/features.yaml`

```yaml
# Phase 3 feature extraction. Output columns are prefixed by family so experiments can select
# families (spectral_, entropy_, wavelet_, plv_, coh_, graph_).
preprocess_config: preprocess.yaml     # which epochs to use (found via its config hash)

families: [spectral, entropy, wavelet, plv, coh, graph]

welch:
  nperseg: 256          # 2 s segments -> 0.5 Hz resolution on 4 s epochs
  noverlap: 128

wavelet:
  name: db4
  level: 5

connectivity:
  bands: [theta, alpha, beta, gamma]
  coh_nperseg: 128      # 1 s segments, 50 % overlap -> 7 segments per 4 s epoch
  filter_order: 4       # Butterworth band filter before the Hilbert transform (PLV)

graph:
  source: plv
  threshold: proportional   # proportional | absolute | none
  proportion: 0.3           # keep the strongest 30 % of edges
  absolute: 0.5             # used when threshold = absolute

n_jobs: 6
out_dir: data/processed
results_dir: results/features
```

### `configs/preprocess.yaml`

```yaml
# Phase 2 preprocessing + epoching. Changing anything below (except paths/plots) changes the
# config hash, so a new epochs_<hash>.npz is written next to the old one.
seed: 42

data:
  stew_dir: data/raw/STEW
  cache: data/processed/stew_raw_cache.npz
  out_dir: data/processed

filter:
  l_freq: 0.5            # Hz, zero-phase FIR (firwin, Hamming)
  h_freq: 45.0
  notch_enabled: true    # Emotiv already notches 50/60 Hz on board and 45 Hz LP makes this
  notch_freq: 50.0       # mostly redundant, but it is cheap and protects other headsets

quality:                 # applied to the FILTERED signal; window/step are taken from `epoching`
  flat_std_uv: 0.5
  max_abs_uv: 150.0      # chosen from the threshold sweep in results/preprocessing (see D7)
  var_z_thresh: 3.5
  hf_band_hz: [30.0, 45.0]
  hf_ratio_thresh: 0.25
  line_freq_hz: 50.0
  line_ratio_thresh: 10.0
  max_bad_channels_warn: 2
  blink_channels: [AF3, AF4, F7, F8]
  blink_max_uv: 300.0    # frontal amp/variance-only failures below this are blinks -> "warn"

bad_channels:
  interpolate: true
  min_bad_fraction: 0.6  # bad in >= 60 % of windows -> persistently bad
  max_std_ratio: 4.0     # or median window std > 4x that of the median channel
  max_interpolated: 3    # never rebuild more than 3 of 14 channels

ica:
  enabled: false         # optional; AF3/AF4 used as EOG proxies
  method: fastica
  n_components: 13
  eog_proxies: [AF3, AF4]
  eog_z_thresh: 3.0
  max_removed: 2

epoching:
  window_s: 4.0          # also supported: 2.0 (use configs/preprocess_2s.yaml)
  step_s: 2.0            # 50 % overlap

normalisation: none      # none | epoch_zscore | rest_baseline (signal level). Feature-level
                         # rest baseline is applied in Phase 4 experiments instead (see D9).

exclude_bad_windows_from_training: false   # flags are kept either way; experiments can opt in

plots:
  psd_subjects: [1, 38]
  out_dir: results/preprocessing
```

### `configs/preprocess_2s.yaml`

```yaml
# 2-second epoch variant of preprocess.yaml (everything else identical).
# config hash, so a new epochs_<hash>.npz is written next to the old one.
seed: 42

data:
  stew_dir: data/raw/STEW
  cache: data/processed/stew_raw_cache.npz
  out_dir: data/processed

filter:
  l_freq: 0.5            # Hz, zero-phase FIR (firwin, Hamming)
  h_freq: 45.0
  notch_enabled: true    # Emotiv already notches 50/60 Hz on board and 45 Hz LP makes this
  notch_freq: 50.0       # mostly redundant, but it is cheap and protects other headsets

quality:                 # applied to the FILTERED signal; window/step are taken from `epoching`
  flat_std_uv: 0.5
  max_abs_uv: 150.0      # chosen from the threshold sweep in results/preprocessing (see D7)
  var_z_thresh: 3.5
  hf_band_hz: [30.0, 45.0]
  hf_ratio_thresh: 0.25
  line_freq_hz: 50.0
  line_ratio_thresh: 10.0
  max_bad_channels_warn: 2
  blink_channels: [AF3, AF4, F7, F8]
  blink_max_uv: 300.0    # frontal amp/variance-only failures below this are blinks -> "warn"

bad_channels:
  interpolate: true
  min_bad_fraction: 0.6  # bad in >= 60 % of windows -> persistently bad
  max_std_ratio: 4.0     # or median window std > 4x that of the median channel
  max_interpolated: 3    # never rebuild more than 3 of 14 channels

ica:
  enabled: false         # optional; AF3/AF4 used as EOG proxies
  method: fastica
  n_components: 13
  eog_proxies: [AF3, AF4]
  eog_z_thresh: 3.0
  max_removed: 2

epoching:
  window_s: 2.0
  step_s: 1.0            # 50 % overlap

normalisation: none      # none | epoch_zscore | rest_baseline (signal level). Feature-level
                         # rest baseline is applied in Phase 4 experiments instead (see D9).

exclude_bad_windows_from_training: false   # flags are kept either way; experiments can opt in

plots:
  psd_subjects: [1, 38]
  out_dir: results/preprocessing_2s
```

### `configs/quality.yaml`

```yaml
# Signal quality assessment (Phase 1). Thresholds are in microvolts after DC removal.
data:
  stew_dir: data/raw/STEW
  cache: data/processed/stew_raw_cache.npz

dc_removal:
  highpass_hz: 0.5          # zero-phase Butterworth high-pass after mean removal

quality:
  window_s: 4.0             # same window/step as the Phase 2 epochs
  step_s: 2.0
  flat_std_uv: 0.5          # std below this -> flatline (Emotiv LSB is ~0.51 uV)
  max_abs_uv: 150.0         # |x| above this -> excessive amplitude (blink / movement)
  var_z_thresh: 3.5         # |robust z| of log-variance across channels
  hf_band_hz: [30.0, 45.0]  # Emotiv low-passes at ~43 Hz -> measure EMG inside the passband
  hf_ratio_thresh: 0.25     # P(30-45 Hz) / P(1-45 Hz) -> muscle / EMG (median on STEW ~0.03)
  line_freq_hz: 50.0        # Sri Lanka mains = 50 Hz
  line_ratio_thresh: 10.0   # PSD at line freq / flanking PSD
  max_bad_channels_warn: 2  # <=2 bad channels -> "warn", more -> "bad"

output_dir: results/quality
```

<!-- END TABLE:configs -->

## 12.2 Decisions log (summary of `docs/decisions.md`)
<!-- BEGIN TABLE:decisions -->
| ID | Decision | Summary (first lines of docs/decisions.md) |
|---|---|---|
| D1 | Proposal document not available | `docs/proposal.docx` is not in the repository. All methods follow the phase specifications given by the author (which summarise proposal sections 4.6–4.15). If the proposal differs on a detail, the code is config-driven and can be adjusted. |
| D2 | Python 3.13 instead of 3.11 | Only Python 3.13.7 is installed on the development machine; all dependencies support it. `requires-python >= 3.11`, so 3.11 also works. |
| D3 | component1/ is the eeg-engine root | The STEW data was already placed in `component1/data/raw/STEW/`, and the git repository is shared with teammates, so `component1/` plays the role of `eeg-engine/`. |
| D4 | HF/EMG quality check uses 30–45 Hz, not > 45 Hz | The Emotiv EPOC low-passes on board at ~43 Hz, so power above 45 Hz is ~0 even in noisy data (measured ratio ~0.001). The EMG check therefore uses P(30–45 Hz) / P(1–45 Hz). |
| D5 | Montage: colin27_1020 (= standard_1020) | MNE 1.13 deprecates the name `standard_1020` in favour of `colin27_1020`. For our 14 Emotiv electrodes the positions of `colin27_1020`, `standard_1020` and `standard_1005` are identical (max difference 0.0 mm, checked in Phase 2). The code uses `colin27_1020` … |
| D6 | Quality checks on detrended windows | The amplitude/variance checks linearly detrend each window first. Without this, slow drift inside a 4 s window counted as an "artifact", and the result depended on how much 0.25–0.5 Hz drift the high-pass let through (MNE's FIR has a 0.5 Hz-wide transition ban… |
| D7 | Amplitude threshold 150 µV + frontal blink tolerance | Threshold sweep on the filtered data (results/preprocessing/quality_raw_vs_filtered.json): 100 µV marks ~26 % of windows bad, 150 µV ~14 %, 200 µV ~10 %. 100 µV is too strict for a consumer headset with dry-ish saline sensors; 150 µV is the common upper bound … |
| D8 | Persistent bad channels: detect + spherical-spline interpolation | Rule (bad_channels.py): flat or hard-failing in >= 60 % of windows, or a median per-window std > 4x that of the median channel. The median (not the whole-recording std) is used so a few huge spikes do not condemn a whole channel — those windows are flagged ins… |
| D9 | Normalisation location | Epochs are saved un-normalised (`normalisation: none`). Signal-level options (epoch z-score, subject rest baseline) are implemented for the deep models; the classical models use the feature-level rest baseline (z-score each feature with the subject's own rest … |
| D10 | Usable epochs | "Usable" = quality flag != "bad". Overall 87.8 % usable; sub27_hi and sub37_lo are below 50 % but are kept. Experiments train on all epochs by default (the model sees realistic noise); `exclude_bad_windows` is an experiment option. |
| D11 | FIR filter edges | MNE's default transition bands are 0.5 Hz at the 0.5 Hz edge and 11.25 Hz at the 45 Hz edge (–6 dB at ~50.6 Hz). Combined with the 50 Hz notch, power at 57–64 Hz drops by roughly 50–60 dB (results/preprocessing/psd_before_after_*.png). |
| D12 | Classical hyper-parameters fixed (no nested tuning) | Nested LOSO tuning (48 outer x k inner folds x grid) is not affordable on a CPU. Each model uses one fixed, conventional configuration (models/classical.py). RF and XGBoost were shrunk (200 trees; RF bootstrap 50 %; XGB 64 bins, 30 % columns) after checking on… |
| D13 | SVM scores are not probabilities | The RBF-SVM is trained without Platt scaling (which would cost 5 extra fits per fold). Its decision values are squashed with a sigmoid/softmax so AUROC and averaging work, but they are not calibrated probabilities. Calibration is studied on the deep models (te… |
| D14 | Classical grid reduced to fit the machine | The development laptop (4 cores, 16 GB, shared with VS Code, Chrome, WSL and SQL Server — commit charge ~31 of 35 GB) paged heavily when 8 LOSO workers ran; RF folds took ~650 s instead of ~17 s. The grid was therefore run with 4 workers and split: |
| D15 | Feature ablation scope | A1–A5 + all are run with the best classical model for T1 and T3 (the two primary tasks), each with and without the rest baseline. T2 gets only the full feature set with the rest baseline (it is nearly the rest/task split, D: docs/labels.md), which keeps the ru… |
| D16 | Below-chance T3 scores under LOSO are an artifact, not a signal | Some T3 runs score clearly below chance (e.g. XGB all features + rest baseline: pooled AUROC 0.36). In T3 every subject contributes exactly one class (20 moderate, 25 high). Leaving one subject out shifts the training class balance *against* that subject's cla… |
| D17 | Deep models: small, early-stopped, 5-fold, random-start sequences | On the 4-core laptop each deep model–task pair had to fit in ~15 min. Choices: EEGNet-style encoder with a 32-sample (0.25 s) temporal kernel (a 64-sample kernel hit a very slow CPU convolution path), 8 random sequence starts per training recording per epoch (… |
| D18 | Deployed models | API/engine models (models/): T1_xgb_all and T3_xgb_all (best classical, trained on all subjects, no rest baseline), T1_A10_hybrid and T3_A10_hybrid (full hybrid), T1_A8_cnn_lstm_trans (best deep on T1). The T3 hybrid's final training stopped at epoch 0 and tem… |
<!-- END TABLE:decisions -->

## 12.3 Figures in this report
All PNGs are copied to `docs/report/figures/` by `scripts/build_report.py`. The `summary_*` figures are
generated by that script directly from the result files (originals in `results/figures/`).
<!-- BEGIN TABLE:figures -->
| File (docs/report/figures/) | Original location | Caption |
|---|---|---|
| `eda_rating_distribution.png` | `results/eda/rating_distribution.png` | Self-rated workload (1–9) for rest vs SIMKAP task; dashed lines = T2 level boundaries. |
| `eda_raw_sub01.png` | `results/eda/raw_sub01.png` | First 10 s of subject 01 (rest vs task), DC removed + 0.5 Hz HP; shows the broken T7 electrode at rest. |
| `eda_raw_sub24.png` | `results/eda/raw_sub24.png` | First 10 s of subject 24 (an unrated subject), rest vs task. |
| `eda_psd_grand_average.png` | `results/eda/psd_grand_average.png` | Grand-average PSD (mean of 14 channels) rest vs task, band = ±1 SD across subjects. |
| `eda_psd_subjects.png` | `results/eda/psd_subjects.png` | Per-subject PSD rest vs task for subjects 01, 12, 24, 36. |
| `prep_psd_before_after_sub01.png` | `results/preprocessing/psd_before_after_sub01.png` | PSD before vs after preprocessing, subject 01. |
| `prep_psd_before_after_sub38.png` | `results/preprocessing/psd_before_after_sub38.png` | PSD before vs after preprocessing, subject 38 (F3 interpolated in the task recording). |
| `cm_cls_T1_xgb_all_none_loso.png` | `results/cls_T1_xgb_all_none_loso/confusion_matrix.png` | Confusion matrices, XGBoost T1 LOSO (pooled epochs and per recording). |
| `cm_cls_T1_xgb_all_none_kfold5.png` | `results/cls_T1_xgb_all_none_kfold5/confusion_matrix.png` | Confusion matrices, XGBoost T1 5-fold. |
| `cm_cls_T1_svm_all_rest_baseline_loso.png` | `results/cls_T1_svm_all_rest_baseline_loso/confusion_matrix.png` | Confusion matrices, SVM T1 LOSO WITH rest baseline (the 0.990 per-recording case). |
| `cm_cls_T3_xgb_all_none_loso.png` | `results/cls_T3_xgb_all_none_loso/confusion_matrix.png` | Confusion matrices, XGBoost T3 LOSO (≈ chance). |
| `cm_cls_T2_xgb_all_none_loso.png` | `results/cls_T2_xgb_all_none_loso/confusion_matrix.png` | Confusion matrices, XGBoost T2 LOSO (rest ≈ low). |
| `cm_deep_T1_eegnet_kfold5.png` | `results/deep_T1_eegnet_kfold5/confusion_matrix.png` | Confusion matrices, EEGNet T1 5-fold (18-s windows). |
| `cm_deep_T1_A8_cnn_lstm_trans_kfold5.png` | `results/deep_T1_A8_cnn_lstm_trans_kfold5/confusion_matrix.png` | Confusion matrices, A8 CNN-LSTM-Transformer T1. |
| `cm_deep_T1_A10_hybrid_kfold5.png` | `results/deep_T1_A10_hybrid_kfold5/confusion_matrix.png` | Confusion matrices, A10 full hybrid T1. |
| `cm_deep_T3_A10_hybrid_kfold5.png` | `results/deep_T3_A10_hybrid_kfold5/confusion_matrix.png` | Confusion matrices, A10 full hybrid T3. |
| `reliability_deep_T1_A10_hybrid_kfold5.png` | `results/deep_T1_A10_hybrid_kfold5/reliability.png` | Reliability diagram before/after temperature scaling, deep_T1_A10_hybrid_kfold5. |
| `reliability_deep_T1_A8_cnn_lstm_trans_kfold5.png` | `results/deep_T1_A8_cnn_lstm_trans_kfold5/reliability.png` | Reliability diagram before/after temperature scaling, deep_T1_A8_cnn_lstm_trans_kfold5. |
| `reliability_deep_T1_eegnet_kfold5.png` | `results/deep_T1_eegnet_kfold5/reliability.png` | Reliability diagram before/after temperature scaling, deep_T1_eegnet_kfold5. |
| `reliability_deep_T3_A10_hybrid_kfold5.png` | `results/deep_T3_A10_hybrid_kfold5/reliability.png` | Reliability diagram before/after temperature scaling, deep_T3_A10_hybrid_kfold5. |
| `summary_feature_ablation.png` | `results/figures/feature_ablation.png` | Feature-family ablation A1–A5 + all (XGBoost, LOSO), pooled macro-F1, with/without rest baseline; dashed = chance. |
| `summary_classical_vs_deep.png` | `results/figures/classical_vs_deep.png` | All models on the same 5 subject folds and the same 18-s windows (fold macro-F1 ± std). |
| `summary_rest_baseline_inflation.png` | `results/figures/rest_baseline_inflation.png` | T1 pooled balanced accuracy with vs without the subject rest baseline (LOSO, all features). |
| `summary_latency.png` | `results/figures/latency.png` | CPU latency per window (T1 models). |
<!-- END TABLE:figures -->

Further figures not reproduced here: a `confusion_matrix.png` in every `results/<experiment>/` folder and
a `reliability.png` in every `results/deep_*/` folder; `results/preprocessing_2s/psd_before_after_*.png`.

## 12.4 File-by-file code index
<!-- BEGIN TABLE:code_index -->
| File | Purpose (first docstring / comment line) | Lines | Public functions / classes / exports |
|---|---|---|---|
| `api/main.py` | FastAPI backend of the EEG Neuro-Cognitive Understanding Engine. | 272 | default_model, check_model, epoch_meta, interpolation_log, read_json, health, subject_ratings, subjects, models, results_summary, result, result_file |
| `scripts/audit_leakage.py` | Leakage audit: checks that can be re-run at any time. Writes results/leakage_audit.json. | 122 | fold_integrity, permutation_test, subject_identity_probe, feature_locality, main |
| `scripts/build_report.py` | Build the technical source report docs/report/ from hand-written section templates + result files. | 729 | jload, metrics, all_results, f3, ms, pfmt, table, src, t_labels, t_ratings, t_quality_rates, t_sweep |
| `scripts/deep_stats.py` | Paired Wilcoxon tests for the deep models (same 5 folds, same 18-s windows). | 61 | seq_view, score, main |
| `scripts/explore_dataset.py` | Exploratory summary of the STEW dataset. | 209 | main, plot_ratings, plot_raw, grand_psd, plot_grand_psd, plot_subject_psd |
| `scripts/export_contract.py` | Export the output contract for teammates: JSON Schema + one REAL example produced by the engine. | 40 | main |
| `scripts/export_demo.py` | Export an offline snapshot for the dashboard's demo mode (frontend/public/demo/). | 86 | dump, stream_messages, main |
| `scripts/export_report.py` | Build every summary table FROM the result files (no number is typed by hand). | 257 | row, sort_key, md_table, pooled, main, write_classical_md, write_deep_md, write_all_md, key_results, key_table_md, update_readme |
| `scripts/run_ablation.py` | Phase 4 feature-family ablation (A1-A5 + all) with the best classical model, plus Wilcoxon tests. | 113 | pick_best_model, main |
| `scripts/run_all.py` | Run the whole pipeline from the raw STEW files, in order, with one command. | 70 | main |
| `scripts/run_baselines.py` | Phase 4: run a suite of classical experiments from a YAML config. | 66 | main |
| `scripts/run_deep.py` | Phase 5: deep models over the fixed 5 subject folds, and final deployable checkpoints. | 131 | ensure_folds, main, train_final |
| `scripts/run_features.py` | Phase 3: extract all feature families from the saved epochs (cached by config hash). | 86 | main |
| `scripts/run_preprocess.py` | Phase 2: preprocess + epoch all STEW recordings. | 261 | flag_rates, main, plot_psd_before_after, write_labels_doc |
| `scripts/run_quality.py` | Run the signal-quality assessment on all STEW recordings. | 47 | main |
| `scripts/train_final_classical.py` | Train deployable classical models on ALL labelled subjects (for the inference engine / API). | 69 | main |
| `src/eegengine/config.py` | Small helpers for YAML configs, project paths and seeding. | 42 | config_path, load_config, resolve, set_seed |
| `src/eegengine/data.py` | Locate and load the generated epochs / features from their config hashes. | 58 | feature_hash, epochs_paths, load_epochs, features_paths, load_features |
| `src/eegengine/evaluation/ablation.py` | Paired statistical comparison of two experiments (Wilcoxon signed-rank on per-subject scores). | 46 | paired_wilcoxon, holm |
| `src/eegengine/evaluation/metrics.py` | Evaluation metrics: macro-F1, balanced accuracy, precision, recall, AUROC, confusion, ECE. | 104 | classification_metrics, auroc, confusion, mean_std, recording_level, per_subject_scores, expected_calibration_error |
| `src/eegengine/evaluation/reports.py` | Write experiment outputs and build summary tables FROM the saved result files. | 100 | save_experiment, plot_confusions, load_results, fmt, num |
| `src/eegengine/features/connectivity.py` | Functional connectivity per frequency band: PLV and magnitude-squared coherence. | 74 | band_filter, plv_matrix, coherence_matrix, connectivity_matrices, upper_triangle_columns |
| `src/eegengine/features/entropy.py` | Complexity / time-domain features (family prefix `entropy_`). | 46 | spectral_entropy, hjorth, entropy_features |
| `src/eegengine/features/extract.py` | Feature extraction entry point: epochs -> feature table + connectivity matrices. | 103 | extract_features, extract_features_parallel, family_columns |
| `src/eegengine/features/graph.py` | Graph-theory features of the PLV brain network (family prefix `graph_`), using networkx. | 96 | threshold_matrix, graph_metrics, graph_features |
| `src/eegengine/features/spectral.py` | Spectral features from the Welch power spectral density (family prefix `spectral_`). | 72 | welch_psd, band_power, spectral_features |
| `src/eegengine/features/wavelet.py` | Discrete wavelet transform features (family prefix `wavelet_`). | 30 | wavelet_features |
| `src/eegengine/inference/engine.py` | Real-time Cognitive Engine: raw 14-channel EEG in, CognitiveStateOutput per window out. | 171 | CognitiveEngine |
| `src/eegengine/inference/heuristics.py` | Heuristic EEG indices for attention, fatigue and readiness (NOT trained predictions). | 41 | heuristic_indices |
| `src/eegengine/inference/predictors.py` | Load deployable models (classical .joblib or deep .pt) behind one small interface. | 97 | ClassicalPredictor, DeepPredictor, load_predictor, list_models |
| `src/eegengine/inference/schema.py` | Output contract of the EEG engine (consumed by the fusion, XAI and cueing components). | 72 | SignalQuality, WorkloadState, HeuristicState, States, Connectivity, CognitiveStateOutput |
| `src/eegengine/io/stew.py` | STEW (Simultaneous Task EEG Workload) dataset loader and label builder. | 257 | Recording, rating_to_level, load_ratings, recording_path, read_signal_file, load_recording, load_stew, recordings_to_frame, to_mne_raw |
| `src/eegengine/labels.py` | Task label builder. STEW only provides WORKLOAD information. | 71 | within_task_label, add_labels, task_view, label_counts |
| `src/eegengine/models/classical.py` | Classical baselines as scikit-learn Pipelines: StandardScaler -> optional SelectKBest/PCA -> model. | 82 | make_pipeline, fit_pipeline, predict_scores |
| `src/eegengine/models/deep/cnn_lstm.py` | CNN-LSTM baseline: per-epoch EEGNet-style CNN encoder -> LSTM over the epoch sequence. | 13 | build_cnn_lstm |
| `src/eegengine/models/deep/eegnet.py` | EEGNet (Lawhern et al., 2018, J. Neural Eng.) and the EEGNet-style epoch encoder. | 64 | EEGNetEncoder, EEGNet |
| `src/eegengine/models/deep/gat.py` | Graph Attention Network layer in plain PyTorch (no torch_geometric needed). | 58 | GATLayer, GATEncoder |
| `src/eegengine/models/deep/hybrid.py` | Hybrid CNN + LSTM + Transformer + GAT multi-task model. | 115 | PositionalEncoding, HybridModel, masked_multitask_loss |
| `src/eegengine/preprocess/bad_channels.py` | Detect persistently bad channels in a recording and repair them by spherical-spline interpolation. | 91 | detect_bad_channels, interpolate_channels, repair_bad_channels |
| `src/eegengine/preprocess/dc.py` | DC-offset removal (the first step for every STEW signal). | 25 | remove_dc |
| `src/eegengine/preprocess/epoching.py` | Cut a continuous (channels x samples) recording into fixed-length, overlapping epochs. | 20 | make_epochs |
| `src/eegengine/preprocess/filters.py` | Zero-phase FIR band-pass and notch filtering (MNE implementations, applied to plain arrays). | 38 | bandpass, notch, filter_recording |
| `src/eegengine/preprocess/ica.py` | Optional ICA ocular-artifact removal (default OFF in preprocess.yaml). | 43 | run_ica |
| `src/eegengine/preprocess/normalize.py` | Normalisation options: none | epoch_zscore | rest_baseline. | 64 | epoch_zscore, rest_baseline_signal, normalize_epochs, rest_baseline_features |
| `src/eegengine/preprocess/pipeline.py` | Full preprocessing pipeline for one recording, driven by configs/preprocess.yaml. | 80 | config_hash, epoch_quality_config, preprocess_recording |
| `src/eegengine/quality/checks.py` | Per-channel, per-window signal quality checks. | 188 | QualityConfig, window_starts, robust_z, channel_metrics, assess_window, assess_recording |
| `src/eegengine/quality/report.py` | Build the dataset-level quality report (CSV + JSON) from per-recording checks. | 83 | build_quality_report, save_quality_report |
| `src/eegengine/training/calibration.py` | Temperature scaling (Guo et al., 2017): divide logits by one scalar T > 0. | 35 | fit_temperature, softmax |
| `src/eegengine/training/classical_runner.py` | Run one classical experiment: (task, model, feature set, normalisation, CV) -> results folder. | 200 | experiment_id, FoldModel, run_classical |
| `src/eegengine/training/cv.py` | Subject-wise cross-validation splitters with a built-in leakage guard. | 74 | assert_no_subject_overlap, loso_splits, subject_fold_assignment, kfold_splits, inner_validation_split, save_assignment, load_assignment |
| `src/eegengine/training/deep_runner.py` | Run one deep experiment (model spec x task) over the fixed 5 subject folds and save results. | 157 | deep_experiment_id, run_deep, plot_reliability, save_checkpoint |
| `src/eegengine/training/sequences.py` | Sequences of consecutive epochs from the SAME recording (input unit of the temporal models). | 61 | make_sequences, sequence_scores_from_epochs, recording_groups, sample_items, strided_items |
| `src/eegengine/training/trainer.py` | Deep model training for one CV fold (and for the final deployable model). | 271 | set_torch_seed, Scalers, DeepInputs, augment_batch, edge_matrix, build_model, is_epoch_model, forward, predict_logits, train_model, run_fold, measure_latency |
| `tests/test_engine_api.py` | Phase 6 tests: inference engine, output contract, heuristics, and the FastAPI app. | 216 | tiny_model, test_heuristic_formulas, test_engine_full_recording, test_engine_streaming_matches_batch, test_engine_band_power_close_to_offline, client, bare_client, test_health, test_subjects, test_models, test_schema, test_quality |
| `tests/test_features.py` | Phase 3 tests: connectivity, band powers, graph metrics, feature table structure. | 160 | test_plv_identical_signals_is_one, test_plv_constant_phase_lag_is_one, test_plv_random_phase_is_near_zero, test_plv_symmetric_unit_diagonal, test_coherence_matches_scipy, test_coherence_bounds, test_relative_band_powers_sum_to_one, test_band_powers_sum_to_total_1_45, test_alpha_sine_lands_in_alpha, test_spectral_entropy_white_vs_sine, test_hjorth_mobility_of_sine, test_wavelet_relative_energy_sums_to_one |
| `tests/test_models_cv.py` | Tests for the deep models, the masked multi-task loss, calibration, metrics and the CV leakage guard. | 150 | test_loso_never_mixes_subjects, test_kfold_assignment_fixed_and_disjoint, test_guard_raises_on_overlap, test_inner_validation_split_by_subject, test_scaler_is_fit_inside_pipeline, test_sequences_stay_within_recording, test_eegnet_shapes, test_hybrid_branches, test_hybrid_accepts_shorter_sequences, test_masked_loss_ignores_unlabelled_heads, test_gat_respects_adjacency, test_temperature_scaling_reduces_nll_on_overconfident_logits |
| `tests/test_preprocess.py` | Phase 2 tests: filters, epoching, bad-channel repair, normalisation, labels. | 186 | sine, gain, test_bandpass_passes_in_band, test_bandpass_attenuates_out_of_band, test_bandpass_is_zero_phase, test_notch_removes_50hz, test_filter_recording_removes_dc, test_epoch_count, test_frontal_blink_is_warn_not_bad, test_non_frontal_artifact_is_bad, test_bad_channel_detected_and_interpolated, test_epoch_zscore |
| `tests/test_quality.py` | Tests for the signal quality checks, using synthetic signals with known defects. | 112 | clean_window, cfg, test_clean_window_is_good, test_flatline_detected, test_excessive_amplitude_detected, test_abnormal_variance_detected, test_line_noise_detected, test_high_frequency_noise_detected, test_many_bad_channels_flag_bad, test_window_starts, test_assess_recording_marks_not_deletes, test_robust_z_constant |
| `tests/test_stew_loader.py` | Tests for the STEW loader and label builder. | 139 | recs, test_rating_to_level, test_rating_out_of_range, test_ratings_missing_subjects, test_all_recordings_loaded, test_shapes_and_metadata, test_binary_labels, test_missing_ratings_are_masked_not_dropped, test_each_recording_uses_its_own_rating, test_file_orientation_matches_raw_text, test_dc_offset_present_in_raw, test_cache_roundtrip |
| `frontend/src/App.tsx` | HashRouter: the built dashboard works from any static folder, including offline demo mode. | 32 | App |
| `frontend/src/components/EEGTrace.tsx` | Scrolling 14-channel EEG trace on a canvas. The display runs 2 s behind the newest data so the | 109 | EEGTrace |
| `frontend/src/components/HeadMap.tsx` | 14-electrode head map (10-20 positions, nose up) coloured by band power (log scale, per window). | 46 | HeadMap |
| `frontend/src/components/Layout.tsx` |  | 62 | Layout |
| `frontend/src/components/NetworkGraph.tsx` | Brain-network graph: electrodes as nodes, PLV above the threshold as edges (width/opacity = PLV). | 49 | NetworkGraph |
| `frontend/src/components/QualityIndicator.tsx` |  | 27 | QualityIndicator |
| `frontend/src/components/ResultCharts.tsx` |  | 88 | MetricBars, Reliability, Confusion |
| `frontend/src/components/StateCards.tsx` |  | 72 | StateCards |
| `frontend/src/components/ui.tsx` |  | 130 | Card, HeuristicBadge, ModelBadge, FlagPill, Select, Button, Segmented, Notice, Stat |
| `frontend/src/components/WorkloadTimeline.tsx` | Probability of the higher-workload class over time, with a confidence band. | 48 | WorkloadTimeline |
| `frontend/src/lib/api.ts` | Data access. Live mode talks to the FastAPI backend; demo mode reads precomputed JSON from | 64 | API_URL, isDemo, setDemo, wsUrl, resultFileUrl |
| `frontend/src/lib/eeg.ts` | EEG constants shared by the views. | 29 | CHANNELS, BANDS, BAND_LABEL, POS, SEQ, seqColor, TASK_LABEL, fmt |
| `frontend/src/lib/settings.tsx` | App-wide settings: theme (light/dark), projector mode (larger type) and demo mode. | 81 | SettingsProvider, useSettings |
| `frontend/src/lib/stream.tsx` | One stream for the whole app (Live Monitor and Brain View share it, and it survives page changes). | 136 | DISPLAY_SECONDS, RawBuffer, StreamProvider, useStream |
| `frontend/src/lib/types.ts` | Types mirror src/eegengine/inference/schema.py (output contract v1.0.0) and the API responses. | 168 | Flag, HeuristicState, CognitiveStateOutput, StreamMeta, StreamMessage, ModelInfo, Subject, QualityRow, QualityWindow, MeanStd, ResultRow, StatTest |
| `frontend/src/lib/useApi.ts` |  | 25 | useApi |
| `frontend/src/main.tsx` |  | 10 |  |
| `frontend/src/pages/About.tsx` |  | 78 | About |
| `frontend/src/pages/BrainView.tsx` |  | 54 | BrainView |
| `frontend/src/pages/Integration.tsx` |  | 90 | Integration |
| `frontend/src/pages/LiveMonitor.tsx` |  | 101 | StreamControls, LiveMonitor |
| `frontend/src/pages/Results.tsx` | Every number on this page comes from /results/summary (built by scripts/export_report.py from results/). | 136 | Results |
| `frontend/src/pages/SignalQuality.tsx` |  | 108 | SignalQuality |
| `frontend/src/vite-env.d.ts` | / <reference types="vite/client" /> | 1 |  |
<!-- END TABLE:code_index -->

## 12.5 Dependencies
<!-- BEGIN TABLE:versions -->
| Package | Required (requirements.txt / package.json) | Installed in the dev environment |
|---|---|---|
| Python | ≥ 3.11 (pyproject) | 3.13.7 |
| numpy | numpy>=1.26 | 2.5.4 |
| scipy | scipy>=1.11 | 1.18.1 |
| pandas | pandas>=2.1 | 3.0.6 |
| pyarrow | pyarrow>=14 | 26.0.0 |
| matplotlib | matplotlib>=3.8 | 3.11.2 |
| PyYAML | PyYAML>=6.0 | 6.0.3 |
| mne | mne>=1.6 | 1.13.2 |
| PyWavelets | PyWavelets>=1.5 | 1.10.0 |
| networkx | networkx>=3.2 | 3.7 |
| scikit-learn | scikit-learn>=1.4 | 1.9.1 |
| xgboost | xgboost>=2.0 | 3.4.1 |
| torch | torch>=2.2 | 2.14.1 |
| joblib | (dependency) | 1.6.0 |
| fastapi | fastapi>=0.110 | 0.146.0 |
| uvicorn | uvicorn[standard]>=0.27 | 0.54.0 |
| pydantic | pydantic>=2.5 | 2.14.1 |
| python-multipart | python-multipart>=0.0.9 | 0.0.32 |
| httpx | httpx>=0.27 | 0.28.1 |
| websockets | (dependency) | 17.2 |
| pytest | pytest>=8.0 | 9.1.1 |
| react | ^19.1.0 | 19.3.1 |
| react-dom | ^19.1.0 | 19.3.1 |
| react-router-dom | ^7.6.2 | 7.18.4 |
| recharts | ^2.15.3 | 2.15.3 |
| @tailwindcss/vite | ^4.1.10 | 4.1.10 |
| @vitejs/plugin-react | ^4.4.1 | 4.7.0 |
| tailwindcss | ^4.1.10 | 4.1.10 |
| typescript | ~5.8.3 | 5.8.3 |
| vite | ^6.3.5 | 6.4.4 |
<!-- END TABLE:versions -->

## 12.6 Generated result files (where every number lives)
| File | Content | Written by |
|---|---|---|
| `results/eda/dataset_summary.json`, `recordings.csv`, `*.png` | dataset facts, ratings, EDA plots | `scripts/explore_dataset.py` |
| `results/quality/quality_summary.json`, `quality_windows.csv`, `quality_channels.csv` | raw-signal quality | `scripts/run_quality.py` |
| `results/preprocessing/preprocess_summary.json`, `quality_raw_vs_filtered.json`, `interpolation_log.{csv,json}`, `usable_epochs_per_subject.csv`, `label_counts.json`, `ica_demo_log.json`, `psd_before_after_*.png` | preprocessing results | `scripts/run_preprocess.py` |
| `results/preprocessing_2s/…` | the 2-s epoch variant | same, `--config preprocess_2s.yaml` |
| `results/features/feature_summary.json`, `feature_columns.txt` | feature counts, timings | `scripts/run_features.py` |
| `results/cls_<T>_<model>_<features>_<norm>_<cv>/` | one classical experiment (metrics.json, folds.csv, predictions.csv, confusion_matrix.png, config.yaml) | `scripts/run_baselines.py`, `run_ablation.py` |
| `results/deep_<T>_<model>_kfold5/` | one deep experiment (+ history.csv, reliability.png) | `scripts/run_deep.py` |
| `results/folds_5fold.json` | fixed subject → fold assignment | `scripts/run_deep.py` |
| `results/stats_classical.json`, `stats_deep.json` | Wilcoxon tests | `run_ablation.py`, `deep_stats.py` |
| `results/leakage_audit.json` | audit checks | `scripts/audit_leakage.py` |
| `results/engine_latency.json` | end-to-end engine latency | `scripts/build_report.py` (from the demo streams) |
| `results/summary.json`, `summary_classical.md`, `summary_deep.md`, `summary_all.md` | generated summary tables | `scripts/export_report.py` |
| `models/*.json` | deployable model metadata | `train_final_classical.py`, `run_deep.py --final` |

## 12.7 References cited in the code and documentation
* Lim, W. L., Sourina, O., & Wang, L. (2018). STEW: Simultaneous Task EEG Workload data set. *IEEE Transactions on
  Neural Systems and Rehabilitation Engineering*. — `src/eegengine/io/stew.py` ("Lim et al., 2018").
* Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer
  interfaces. *Journal of Neural Engineering*. — `src/eegengine/models/deep/eegnet.py`.
* Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*. — `src/eegengine/models/deep/gat.py`.
* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*. —
  `src/eegengine/training/calibration.py`.
* Pope, A. T., Bogart, E. H., & Bartolome, D. S. (1995). Biocybernetic system evaluates indices of operator
  engagement in automated task. *Biological Psychology*. — `src/eegengine/inference/heuristics.py`.
* Jap, B. T., Lal, S., Fischer, P., & Bekiaris, E. (2009). Using EEG spectral components to assess algorithms for
  detecting fatigue. *Expert Systems with Applications*. — `src/eegengine/inference/heuristics.py`.
* Onnela, J.-P., et al. (2005), weighted clustering coefficient — named in `src/eegengine/features/graph.py`
  ("weighted, Onnela"; implemented by networkx).
* Spherical-spline interpolation (Perrin et al., 1989) — the method behind MNE `interpolate_bads`; not cited in the code.

Bibliographic details beyond author/year/venue (volume, pages) are not stored in the repository and should be
completed from the publishers before submission.

## 12.8 Verification of this report
`python scripts/build_report.py` (re)builds every section from `docs/report/_src/` + the result files, copies the
figures, writes `FULL_REPORT.md`, checks every prose citation (`value [`file` → `key`]`) against the cited JSON/YAML
value (rounded to the written precision; 5 % tolerance for numbers written in scientific notation) and checks
that every generated table equals a fresh generation. The outcome is written to `docs/report/verification.json`,
then everything is zipped into `docs/report_bundle.zip`.
