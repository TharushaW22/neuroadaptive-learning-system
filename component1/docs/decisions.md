# Design decisions

Judgment calls made while building the engine, with the reason for each. Numbered for reference
in the viva notes and code comments.

## D1 — Proposal document not available
`docs/proposal.docx` is not in the repository. All methods follow the phase specifications
given by the author (which summarise proposal sections 4.6–4.15). If the proposal differs on a
detail, the code is config-driven and can be adjusted.

## D2 — Python 3.13 instead of 3.11
Only Python 3.13.7 is installed on the development machine; all dependencies support it.
`requires-python >= 3.11`, so 3.11 also works.

## D3 — component1/ is the eeg-engine root
The STEW data was already placed in `component1/data/raw/STEW/`, and the git repository is
shared with teammates, so `component1/` plays the role of `eeg-engine/`.

## D4 — HF/EMG quality check uses 30–45 Hz, not > 45 Hz
The Emotiv EPOC low-passes on board at ~43 Hz, so power above 45 Hz is ~0 even in noisy data
(measured ratio ~0.001). The EMG check therefore uses P(30–45 Hz) / P(1–45 Hz).

## D5 — Montage: colin27_1020 (= standard_1020)
MNE 1.13 deprecates the name `standard_1020` in favour of `colin27_1020`. For our 14 Emotiv
electrodes the positions of `colin27_1020`, `standard_1020` and `standard_1005` are identical
(max difference 0.0 mm, checked in Phase 2). The code uses `colin27_1020` when available and
falls back to `standard_1020` on older MNE, so results do not depend on the MNE version.

## D6 — Quality checks on detrended windows
The amplitude/variance checks linearly detrend each window first. Without this, slow drift
inside a 4 s window counted as an "artifact", and the result depended on how much 0.25–0.5 Hz
drift the high-pass let through (MNE's FIR has a 0.5 Hz-wide transition band, so it passes more
drift than the 4th-order Butterworth used in Phase 1).

## D7 — Amplitude threshold 150 µV + frontal blink tolerance
Threshold sweep on the filtered data (results/preprocessing/quality_raw_vs_filtered.json):
100 µV marks ~26 % of windows bad, 150 µV ~14 %, 200 µV ~10 %. 100 µV is too strict for a
consumer headset with dry-ish saline sensors; 150 µV is the common upper bound for "clean" EEG.
Frontal channels (AF3/AF4/F7/F8) that fail only amplitude/variance below 300 µV are treated as
blinks: the window becomes "warn", never "bad", because blinks are expected and the model must
cope with them (they are also present at inference time).

## D8 — Persistent bad channels: detect + spherical-spline interpolation
Rule (bad_channels.py): flat or hard-failing in >= 60 % of windows, or a median per-window std
> 4x that of the median channel. The median (not the whole-recording std) is used so a few
huge spikes do not condemn a whole channel — those windows are flagged instead. At most 3 of
14 channels are interpolated per recording. Result: 30 channels in 26 recordings, every one
logged with its reason in results/preprocessing/interpolation_log.csv. No subject is dropped.

## D9 — Normalisation location
Epochs are saved un-normalised (`normalisation: none`). Signal-level options (epoch z-score,
subject rest baseline) are implemented for the deep models; the classical models use the
feature-level rest baseline (z-score each feature with the subject's own rest epochs), which is
the stronger and more common form. Every T1 result is reported with and without it.

## D10 — Usable epochs
"Usable" = quality flag != "bad". Overall 87.8 % usable; sub27_hi and sub37_lo are below 50 %
but are kept. Experiments train on all epochs by default (the model sees realistic noise);
`exclude_bad_windows` is an experiment option.

## D11 — FIR filter edges
MNE's default transition bands are 0.5 Hz at the 0.5 Hz edge and 11.25 Hz at the 45 Hz edge
(–6 dB at ~50.6 Hz). Combined with the 50 Hz notch, power at 57–64 Hz drops by roughly 50–60 dB
(results/preprocessing/psd_before_after_*.png).

## D12 — Classical hyper-parameters fixed (no nested tuning)
Nested LOSO tuning (48 outer x k inner folds x grid) is not affordable on a CPU. Each model uses
one fixed, conventional configuration (models/classical.py). RF and XGBoost were shrunk
(200 trees; RF bootstrap 50 %; XGB 64 bins, 30 % columns) after checking on 6 held-out subjects
that T1 AUROC is unchanged (RF 0.932 vs 0.935, XGB 0.951 vs 0.949) while training is 3-4x
faster. Because nothing is tuned on the test subjects, the reported scores are not optimistic
from model selection — but they may be slightly below what a tuned model could reach.

## D13 — SVM scores are not probabilities
The RBF-SVM is trained without Platt scaling (which would cost 5 extra fits per fold). Its
decision values are squashed with a sigmoid/softmax so AUROC and averaging work, but they are
not calibrated probabilities. Calibration is studied on the deep models (temperature scaling).

## D14 — Classical grid reduced to fit the machine
The development laptop (4 cores, 16 GB, shared with VS Code, Chrome, WSL and SQL Server — commit
charge ~31 of 35 GB) paged heavily when 8 LOSO workers ran; RF folds took ~650 s instead of ~17 s.
The grid was therefore run with 4 workers and split:
* all 4 models x T1, T2, T3 without rest baseline (configs/experiments/classical.yaml);
* all 4 models x T1 WITH rest baseline (classical_rest.yaml) — T1 is where the rest baseline can
  inflate results, so this is where every model is compared;
* for T2 and T3 the rest-baseline comparison uses the best model only (inside the feature ablation,
  which runs both normalisations for every task).
No hyper-parameter changed, so all runs remain comparable.

## D15 — Feature ablation scope
A1–A5 + all are run with the best classical model for T1 and T3 (the two primary tasks), each
with and without the rest baseline. T2 gets only the full feature set with the rest baseline
(it is nearly the rest/task split, D: docs/labels.md), which keeps the run time within budget.

## D16 — Below-chance T3 scores under LOSO are an artifact, not a signal
Some T3 runs score clearly below chance (e.g. XGB all features + rest baseline: pooled AUROC 0.36).
In T3 every subject contributes exactly one class (20 moderate, 25 high). Leaving one subject out
shifts the training class balance *against* that subject's class (e.g. 20/24 when a "high" subject
is held out), and a model that has learned little beyond the prior then leans towards the other
class — so held-out predictions anti-correlate with the truth. This "LOSO anti-learning" effect is
well known for small datasets with one label per subject. It means: T3 contains little or no
learnable signal with these features; it must not be read as "the model learned the opposite".
The leakage audit's label-permutation test shows what pure chance looks like for comparison.

## D17 — Deep models: small, early-stopped, 5-fold, random-start sequences
On the 4-core laptop each deep model–task pair had to fit in ~15 min. Choices: EEGNet-style encoder
with a 32-sample (0.25 s) temporal kernel (a 64-sample kernel hit a very slow CPU convolution
path), 8 random sequence starts per training recording per epoch (fresh each epoch), batch 16,
AdamW 1e-3, at most 25 epochs with patience 5 on validation loss, 4 threads. Every model uses the
same folds, seeds and windows. These are reasonable but untuned settings — the deep models may be
under-optimised relative to XGBoost, which is stated as a limitation.

## D18 — Deployed models
API/engine models (models/): T1_xgb_all and T3_xgb_all (best classical, trained on all subjects,
no rest baseline), T1_A10_hybrid and T3_A10_hybrid (full hybrid), T1_A8_cnn_lstm_trans (best deep
on T1). The T3 hybrid's final training stopped at epoch 0 and temperature scaling reached its
upper bound (T = 20), which flattens its probabilities towards 0.5 — consistent with T3 having no
learnable signal. The dashboard's default is the first listed model; XGBoost T1 is the most
accurate choice for demonstrations.
