# Viva notes — EEG Neuro-Cognitive Understanding Engine (J26-DS-333, Component 1)

Plain-language explanations of every module, the design decisions and how to defend them.
Numbers quoted here are copied from `results/summary_*.md` (generated from the result files);
section 9 lists where each one comes from.

---

## 1. The pipeline in one paragraph
Raw Emotiv EEG (14 channels, 128 Hz) is checked for quality, cleaned (DC offset removed, 0.5–45 Hz
band-pass, 50 Hz notch, broken channels interpolated), cut into 4-second windows that overlap by
half, and described by ~1,500 numbers per window (how much power is in each frequency band, how
complex the signal is, how strongly electrodes are synchronised, and the shape of the resulting
brain network). Classical classifiers learn from those numbers; deep networks learn directly from
the cleaned signal plus the connectivity graph. The deployed engine repeats exactly the same steps
on live data every 2 seconds and sends a JSON message to the other components and the dashboard.

## 2. Module by module

| Module | What it does | Key idea to explain |
|---|---|---|
| `io/stew.py` | Loads the 96 STEW text files → arrays (14 × 19,200) + ratings; builds labels | Files are samples × channels, so we transpose; missing ratings (5, 24, 42) become NaN and are **masked**, never dropped |
| `quality/checks.py` | Five checks per window and channel: flat, amplitude, variance (robust z across channels), EMG (30–45 Hz share), line noise | Windows are only *marked* good/warn/bad. Frontal blinks only make a window "warn" (D7) |
| `preprocess/filters.py` | Mean removal, zero-phase FIR band-pass, notch | Zero-phase = no time shift of peaks; FIR = stable, linear phase |
| `preprocess/bad_channels.py` | Finds persistently bad channels and rebuilds them with spherical-spline interpolation | Uses the *median* window amplitude so a few spikes don't condemn a channel (D8); max 3 of 14 |
| `preprocess/ica.py` | Optional ICA eye-artifact removal (off by default) | No EOG electrode → AF3/AF4 act as EOG proxies |
| `preprocess/epoching.py` | 4-s windows, 2-s step → 74 per recording | Overlap gives more training samples and a smoother live output |
| `preprocess/normalize.py` | none / per-epoch z-score / subject rest baseline | Rest baseline is a calibration step that can inflate T1 → always reported both ways |
| `labels.py` | T1, T2, T3 label columns with -1 = masked | T3 is the confound-free task |
| `features/spectral.py` | Welch PSD → absolute/relative band power, ratios, frontal alpha asymmetry | Bands tile 1–45 Hz so relative powers sum to 1 (tested) |
| `features/entropy.py` | Spectral entropy, Hjorth mobility/complexity, variance, skewness, kurtosis | Hjorth activity = variance, stored once |
| `features/wavelet.py` | db4 DWT, 5 levels → energy, relative energy, std, wavelet entropy | At 128 Hz the levels map to 32–64, 16–32, 8–16, 4–8, 2–4, 0–2 Hz |
| `features/connectivity.py` | PLV (Hilbert phase) and magnitude-squared coherence, 4 bands → 14 × 14 matrices | PLV(identical) = 1, PLV(random) ≈ 0 (tested); coherence matches SciPy (tested) |
| `features/graph.py` | networkx graph metrics from PLV: strength, clustering, betweenness, eigenvector centrality, efficiency, density, entropy | Proportional threshold (top 30 % edges) keeps graphs of different subjects comparable |
| `models/classical.py` | Scaler → (optional SelectKBest/PCA) → LR / SVM / RF / XGBoost in one sklearn Pipeline | The whole pipeline is fit inside each fold → scaler never sees the test subject |
| `models/deep/eegnet.py` | EEGNet (Lawhern 2018) and the shared per-epoch CNN encoder | temporal conv = learned band-pass filters; depthwise spatial conv = learned spatial filters |
| `models/deep/gat.py` | Graph attention layer in plain PyTorch | Attention over neighbours + log(PLV) prior; dense 14 × 14 adjacency, no extra library |
| `models/deep/hybrid.py` | CNN + LSTM + Transformer + GAT branches → fusion MLP → 4 heads | Each branch switchable (ablation A6–A10); masked loss makes unlabelled heads inert |
| `training/cv.py` | LOSO and fixed 5-fold subject splits, inner validation split | `assert_no_subject_overlap` runs on every split |
| `training/trainer.py` | AdamW, class weights, augmentation, early stopping, temperature scaling | Scalers fit on the training subjects of the fold only |
| `training/calibration.py` | Temperature scaling | Divides logits by T: changes confidence, never the predicted class |
| `evaluation/metrics.py` | macro-F1, balanced accuracy, precision, recall, AUROC, confusion, ECE | Three views: per fold, pooled, per recording |
| `evaluation/ablation.py` | Paired Wilcoxon signed-rank + Holm correction | Paired because both configurations are tested on the same subjects |
| `inference/engine.py` | Live engine: quality → preprocessing → features → model → EMA → JSON | Same functions as training, same 2-s grid |
| `inference/schema.py` | pydantic output contract | `source` field separates model predictions from heuristic indices |
| `api/main.py` | FastAPI REST + WebSocket replay | Engine runs in a worker thread so the event loop stays responsive |
| `frontend/` | React dashboard (6 pages), offline demo mode | All result numbers come from `/results/summary` |

## 3. Why subject-wise cross-validation (LOSO / GroupKFold)
EEG from one person is highly self-similar (skull thickness, electrode placement, alpha frequency).
If windows of the same person are in train and test, a model can recognise the *person* instead of
the *state* and scores look excellent but do not transfer to a new learner. The engine is meant
for new users, so every split keeps each subject entirely in train or entirely in test.
* LOSO (48 folds) for classical models: the closest simulation of "a new student puts on the headset".
* 5 fixed subject folds for deep models (LOSO × deep training is too slow on a CPU) — and the same 5
  folds are re-used for the classical models in the final comparison, so the comparison is paired.

## 4. Leakage precautions (see also docs/leakage_audit.md)
1. Splits by subject, checked by an assertion on every fold (and by tests).
2. Scaler / SelectKBest / PCA inside the sklearn Pipeline → fit on training folds only.
3. Deep input scalers and node-feature scalers computed from the fit subjects of each fold.
4. Early stopping and temperature scaling use an inner **validation split by subject**, never the test fold.
5. Hyper-parameters fixed in advance (no tuning on test subjects) — D12.
6. Features are computed per epoch from that epoch only (no statistics across epochs or subjects).
7. The rest baseline uses only the subject's own rest recording, and every T1 result is shown with and without it.
8. Overlapping epochs never cross the train/test boundary because whole subjects are held out.

## 5. The T2 confound and why T3 matters
T2 (low/medium/high from the rating) looks like graded workload, but rest recordings are rated low
(42 low, 3 medium, 0 high) and task recordings medium or high (0/20/25). A classifier can reach a
good T2 score just by detecting rest vs task. T1 itself is confounded too: rest vs SIMKAP differs in
eye movements, hand movements and arousal, not only mental workload.
T3 keeps the condition fixed — only task recordings — and asks whether EEG separates *moderate*
(rating 4–6) from *high* (7–9) self-reported workload. It is the honest test of workload sensitivity.
Its weakness: only 45 labelled recordings (one per subject), so it is noisy.

## 6. Design decisions (full list in docs/decisions.md)
D4 EMG band 30–45 Hz (Emotiv low-passes at 43 Hz) · D6 detrended quality windows · D7 150 µV + blink
tolerance · D8 median-based bad-channel rule · D9 feature-level rest baseline · D12 fixed classical
hyper-parameters · D13 SVM scores ≠ probabilities · D14 reduced grid for the shared laptop ·
D15+ deep-model choices.

## 7. Limitations
* Workload is the only labelled state; attention / fatigue / readiness are heuristic indices.
* Rest-vs-task confound (T1, T2); T3 is small.
* Consumer headset: frequent blinks and EMG; 30 channels in 26 recordings needed interpolation.
* CPU-only, lightly tuned deep models; 5-fold instead of LOSO for deep models.
* Online filtering uses 20 s of context instead of the whole recording (band powers agree within 5 %, tested).

## 8. Next steps with the team-collected Emotiv data
1. Record sessions with labels for attention, fatigue and readiness (format in README → "Plugging in team data").
2. Switch the masked heads on (label masks), retrain the multi-task hybrid, keep subject-wise CV.
3. Calibrate per user with a short rest baseline (already supported) and compare with/without.
4. Replace heuristic indices in the contract by model outputs (`source: "model"`).

## 9. Results to quote
Filled in from `results/summary_classical.md`, `results/summary_deep.md`, `results/summary_all.md`
(see the "Results" section of the README, which is generated from the same files).
