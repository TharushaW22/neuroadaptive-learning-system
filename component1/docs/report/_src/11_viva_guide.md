# 11. Viva guide

## 11.1 Two-minute explanation (spoken)
"My component is the EEG engine of our neuroadaptive learning system. It takes raw 14-channel EEG from an
Emotiv EPOC headset and, every two seconds, outputs a timestamped JSON message with the learner's estimated
workload, its confidence, a signal-quality flag, and the brain evidence behind it, for the fusion, XAI and
cueing components and for our dashboard.

I developed and validated it on STEW, a public dataset recorded with the same headset: 48 people, each at
rest and during a multitasking test, with a 1-to-9 workload rating. The pipeline checks signal quality,
filters 0.5 to 45 Hz with a zero-phase FIR filter, removes 50 Hz mains noise, repairs broken electrodes by
spherical-spline interpolation, cuts 4-second windows and extracts 1,499 features — band powers, entropy,
wavelets, phase-locking and coherence between electrodes, and brain-network graph metrics. I compared
logistic regression, SVM, random forest and XGBoost with leave-one-subject-out validation, and EEGNet,
a CNN-LSTM and a hybrid CNN-LSTM-Transformer-graph-attention network on fixed subject folds.

Rest versus task is detected well for unseen people — XGBoost reaches about 0.87 macro-F1 per 4-second
window and about 0.92 over 18-second windows — and connectivity features help most. But when I remove the
rest-versus-task difference and only compare moderate against high self-rated workload during the task,
every model is at chance. So the honest conclusion is: the engine detects 'doing a demanding task versus
resting' reliably, but STEW does not support graded workload detection. The system, API and dashboard are
complete, and the multi-task heads for attention, fatigue and readiness are ready for our own labelled data."

## 11.2 Five-minute explanation (spoken, with structure)
1. **Problem and role (30 s).** Adaptive learning needs a continuous, objective measure of the learner's
   cognitive state. EEG gives that directly. My component converts raw EEG into calibrated state estimates in
   a stable JSON contract for the other three components.
2. **Data and labels (45 s).** STEW: 48 subjects, Emotiv EPOC, 128 Hz, 2.5 min rest + 2.5 min SIMKAP
   multitasking, 1–9 workload ratings, three subjects unrated. Three tasks: T1 rest vs task; T2 rating levels;
   T3 moderate vs high *within* the task. I showed that T2 is almost identical to T1 (rest is rated low, task
   medium/high), so T3 is the only confound-free workload test.
3. **Signal processing (60 s).** Quality checks per window and channel (flat, amplitude, variance z-score,
   EMG at 30–45 Hz because the Emotiv low-passes at 43 Hz, 50 Hz line noise); windows are marked, never deleted,
   and frontal blinks only produce 'warn'. Mean removal, zero-phase FIR 0.5–45 Hz (linear phase preserves the
   phase relations PLV needs), 50 Hz notch (Sri Lanka mains), spherical-spline interpolation of 30 persistently
   broken channels, 4-s windows with 50 % overlap.
4. **Features and models (60 s).** 1,499 features in six families; four classical models in sklearn pipelines;
   leave-one-subject-out because a random split lets the model recognise the person — I measured 99.5 % subject
   identification with random splits. Deep models: EEGNet, CNN-LSTM, and a hybrid where a CNN encodes each
   epoch, an LSTM and a Transformer model the sequence of 8 epochs, and a graph-attention network uses band power
   as node features and phase-locking as edges; four task heads with a masked loss so unlabelled states don't
   train. Temperature scaling for calibration.
5. **Results (60 s).** T1: XGBoost best (pooled macro-F1 0.865 LOSO; 0.915 on 18-s windows); connectivity is the
   family that adds significantly; deep models are significantly worse on the same folds (best 0.773). The
   rest-baseline normalisation inflates T1 to 0.95 for the SVM, which I explain and exclude from headlines.
   T3 is at chance for all models. A label-permutation test gives 0.506, so there is no leakage.
6. **System (30 s).** Real-time engine with EMA smoothing, ~100 ms per window; FastAPI with REST and WebSocket
   streaming; React dashboard with live EEG, head map, brain network, quality, results and an offline demo mode.
7. **Limitations and next steps (15 s).** Workload-only labels, small N, CPU training. Next: our own labelled
   Emotiv data, a within-task workload manipulation, live LSL streaming, per-user calibration.

## 11.3 Demo script (evaluation day)
Preparation (before the panel arrives): close other heavy applications; open two terminals in `component1/`.
1. Terminal 1: `.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` — wait for
   "Application startup complete". (If "Errno 10048": `netstat -ano | findstr :8000`, `taskkill /PID <pid> /F`.)
2. Terminal 2: `cd frontend` → `npm run dev` → open **http://127.0.0.1:5173**. Header must show "API online"
   and "Live API" (if the API fails, switch on **Demo (offline)** — subjects 1, 2, 16, 38 are bundled).
3. **About page (30 s):** show the pipeline diagram and the limitations box — sets honest expectations.
4. **Live Monitor:** model `T1_xgb_all`, Subject 01, recording **hi — SIMKAP task**, speed **2×**, Start.
   Point out: scrolling 14-channel EEG; red rows = channels failing quality now; the workload card ("model
   prediction" badge) with probabilities and confidence; attention/fatigue/readiness with the **"heuristic index
   — not a trained prediction"** badge; the probability timeline with the confidence band; the quality squares.
   Then Stop, switch to **lo — rest**, Start: the workload moves towards "low".
5. **Brain View:** while streaming, change the band (theta → alpha → beta) on the head map; move the PLV
   threshold slider and show edges appearing/disappearing; mention volume conduction.
6. **Signal Quality:** sort by "usable ↑": show sub27_hi / sub37_lo below 50 % (kept, not dropped); click
   sub01_lo → the channel × window grid shows T7 bad everywhere before interpolation → interpolated.
7. **Results:** T1 tab — baseline table, rest-baseline rows in amber, feature ablation bars (connectivity), the
   5-fold comparison (XGBoost vs deep), reliability diagram; then **T3 tab** — everything at chance; explain why
   this is the key finding.
8. **Integration / API:** the contract table and the live JSON; open `http://127.0.0.1:8000/docs` to show the
   Swagger UI.
9. Optional: `.venv\Scripts\python -m pytest` — 105 tests pass (~40 s).

## 11.4 Likely examiner questions with model answers

### A. Dataset and labels
1. **Why STEW?** — Same Emotiv EPOC headset (14 channels, 128 Hz) as our planned system, public, and it has
   workload labels, so I could build and validate the full pipeline before our own data exists.
2. **What exactly are the conditions?** — 2.5 min resting and 2.5 min SIMKAP multitasking per subject; each
   followed by a 1–9 workload self-rating.
3. **What about the three subjects without ratings?** — Subjects 5, 24 and 42 keep their rest/task labels (T1)
   but are masked (label −1) for T2 and T3; they are never dropped.
4. **What is T3 and why did you create it?** — Moderate (4–6) vs high (7–9) workload inside the task only. T1 and
   T2 are confounded with the condition; T3 keeps the condition fixed, so only workload differs.
5. **Why is T2 confounded?** — Rest recordings are rated low/medium/high 42/3/0 and task recordings 0/20/25, so
   predicting the level is almost the same as predicting rest vs task.
6. **Is the rating difference real?** — Yes: all 45 rated subjects rated the task higher; Wilcoxon p ≈ 4.3e-9.
7. **How many samples do you have?** — 96 recordings × 74 four-second epochs = 7,104 epochs; for T3, 45
   recordings = 3,330 epochs, but only 45 independent labels.
8. **Is class balance an issue?** — T1 is balanced; T2/T3 are imbalanced, so I use macro-F1, balanced accuracy and
   class weights.

### B. Preprocessing
9. **Why 0.5–45 Hz?** — 0.5 Hz removes drift and sweat artefacts but keeps delta; 45 Hz keeps gamma for the features,
   stays below 50 Hz mains and matches the Emotiv's ~43 Hz on-board low-pass.
10. **Why a zero-phase FIR filter?** — Linear phase does not distort the waveform or the phase relations between
    channels, which PLV depends on; zero-phase compensates the delay.
11. **Why a 50 Hz notch?** — Sri Lanka's mains frequency is 50 Hz; it is mostly redundant after the 45 Hz low-pass but
    protects against other headsets/rooms.
12. **Why is ICA off?** — No EOG channel, so AF3/AF4 must act as proxies, and they contain frontal brain signal —
    removing those components risks deleting frontal theta, the key workload rhythm. With 14 channels ICA is also
    limited. It is implemented and logged as an option.
13. **How do you detect bad channels?** — Persistently bad = flat or hard-failing in ≥ 60 % of windows, or a median
    window amplitude > 4× the median channel. At most 3 per recording are rebuilt by spherical-spline interpolation;
    30 channels in 26 recordings, each logged with its reason.
14. **Why the median, not the overall std?** — A few huge spikes inflate the overall std and would condemn a good
    channel; those windows are flagged instead.
15. **Why do blinks only produce "warn"?** — Blinks are normal and frequent, and they also occur in live use; the
    models must cope with them. Only non-blink failures make a window "bad".
16. **Why 30–45 Hz for EMG, not above 45 Hz?** — The Emotiv low-passes on board at ~43 Hz; above 45 Hz there was no
    power even in noisy data (ratio ≈ 0.001), so the check had to move inside the passband.
17. **Why 4-s windows with 50 % overlap?** — 4 s gives stable spectra and phase estimates down to delta; the overlap
    doubles training windows and gives an output every 2 s.
18. **Do you delete bad windows?** — No. They are flagged; experiments could exclude them from training, but all
    reported results train on all windows so the models see realistic noise.

### C. Features
19. **What does theta/alpha/beta mean for workload?** — Frontal theta rises with working-memory load; alpha drops
    with engagement/visual attention (it reflects idling/inhibition); beta reflects active processing.
20. **Why relative and absolute power?** — Absolute power depends on impedance and skull thickness; relative power
    normalises within each channel and epoch.
21. **What is Welch's method?** — Averaging periodograms of overlapping windowed segments (here 2-s Hann segments,
    50 % overlap) to reduce variance.
22. **What are Hjorth parameters?** — Activity = variance; mobility = √(var(x′)/var(x)) ≈ mean frequency;
    complexity = mobility of the derivative / mobility ≈ bandwidth.
23. **Why db4 wavelets?** — Compact, EEG-like shape, the most common choice; 5 levels at 128 Hz map onto the
    classical bands.
24. **What is PLV?** — |mean over time of exp(i·phase difference)| after band filtering and the Hilbert transform;
    1 = phase-locked, 0 = random.
25. **PLV vs coherence?** — Coherence depends on amplitude and phase consistency across segments; PLV on phase
    only. They give partly independent views.
26. **What is volume conduction?** — One source reaches several nearby electrodes instantly, inflating zero-lag
    coupling between neighbours. Measures like wPLI would avoid it; I list it as a limitation.
27. **Why proportional thresholding for graphs?** — It fixes the density (top 30 % edges), so graph metrics compare
    structure rather than overall PLV level, which differs between people.
28. **Which feature family mattered?** — Connectivity: spectral 0.810 → spectral + connectivity 0.866 on T1,
    p = 5.4e-05; entropy helps a little; wavelets and graph metrics add nothing significant.
29. **Isn't 1,499 features too many for 48 subjects?** — Tree ensembles and L2-regularised models handle it, and
    everything is fit inside the folds; the ablation shows a few families carry the signal. Feature selection is
    implemented but was not needed.

### D. Classical models
30. **Why these four models?** — A linear baseline (LR), the classic EEG kernel method (SVM), and two tree
    ensembles that are strong on tabular features (RF, XGBoost).
31. **Why XGBoost as the best?** — A pre-defined rule: lowest mean rank of pooled macro-F1 over T1 and T3 without
    the baseline; XGBoost ranked first on both.
32. **Did you tune hyper-parameters?** — No; fixed conventional settings, chosen before evaluation, so there is no
    optimism from tuning on test subjects (D12).
33. **How do you handle class imbalance?** — Balanced class weights (LR, SVM, RF) and balanced sample weights
    (XGBoost); macro metrics.
34. **Are your SVM probabilities calibrated?** — No, I skipped Platt scaling for speed; the scores are used for
    ranking (AUROC) only (D13).

### E. Deep models
35. **Describe the hybrid.** — CNN encoder per 4-s epoch (EEGNet blocks → 64-d), LSTM and a 2-layer Transformer
    over the 8-epoch sequence, a 2-layer GAT on the electrode graph (band-power nodes, PLV edges); the embeddings are
    concatenated, passed through an MLP and four heads.
36. **How does the GAT attention work?** — For each node, attention scores over its neighbours from a learned
    vector on the projected features, plus log(PLV) as a prior, softmax over existing edges, weighted sum of
    neighbours; 4 heads.
37. **Why plain PyTorch for the GAT?** — The graph has only 14 nodes, so a dense implementation is simple and
    avoids installing torch_geometric.
38. **What is the masked multi-task loss?** — Sum of cross-entropies over the heads that have labels in the batch;
    a head without labels contributes zero and gets no gradient, so attention/fatigue/readiness are inert until
    labels exist.
39. **Why EEGNet as a baseline?** — It is the standard compact EEG CNN (Lawhern et al., 2018) with interpretable
    temporal and spatial filters.
40. **Why sequences of 8 epochs?** — 18 s of context lets the temporal branches model how the state evolves;
    the classical models are scored on the same 18-s windows for fairness.
41. **How did you prevent overfitting?** — Small models, dropout, weight decay, augmentation (noise, channel
    dropout, time shift), early stopping on validation subjects, class weights.
42. **What does the architecture ablation show?** — Adding the Transformer helped most (0.773), the GAT did not add,
    and none of the differences to the full hybrid is significant.
43. **What is temperature scaling?** — Dividing logits by one scalar T fitted on validation data to minimise NLL;
    it changes confidence but not the predicted class.
44. **Did calibration work?** — Partly: ECE improved for the hybrid (0.098 → 0.055) and A8, but worsened for EEGNet
    and A6, because T is estimated from only ~6 validation subjects.

### F. Evaluation, CV and leakage
45. **Why leave-one-subject-out?** — The engine must work for new people. With a random epoch split, a classifier
    can identify the person from EEG with 99.5 % accuracy, and overlapping windows would leak; LOSO avoids both.
46. **Why 5 folds for the deep models?** — 48 deep trainings per model were unaffordable on a CPU; the same fixed
    folds are reused for the classical models, so that comparison is paired.
47. **How do you know there is no leakage?** — A subject-overlap assertion on every split, scalers inside the folds,
    fold-integrity check of 58 experiments, feature locality check, and a label-permutation control that gives 0.506.
48. **Why macro-F1?** — It weights classes equally, so the majority class cannot dominate.
49. **Why Wilcoxon?** — Paired, non-parametric, suited to 45–48 bounded, possibly skewed per-subject scores; Holm
    correction for multiple tests.
50. **Why do you report pooled, per-fold and per-recording numbers?** — Per-fold shows variability between people;
    pooled is defined even when a fold has one class (T3); per-recording shows what a longer decision window achieves.

### G. Results interpretation
51. **Why does your hybrid lose to XGBoost?** — Small data (≈ 38 training subjects per fold for 141 k parameters),
    XGBoost starts from features that already encode the relevant physiology, untuned CPU training with noisy early
    stopping, and EEGNet with 1.6 k parameters doing equally well confirms data is the bottleneck.
52. **Why is T3 at chance?** — Only 45 labels (one per subject), subjective and compressed ratings, a small true
    difference between people doing the same task, and large between-subject EEG variability.
53. **Is 0.95 real?** — The SVM's 0.951 uses the subject rest baseline, which mechanically makes the rest recording
    the zero-mean reference; it is real for that protocol but inflated. My honest T1 number is 0.865 (XGBoost, LOSO).
54. **Then what does T1 actually detect?** — The difference between resting and doing a demanding visual-motor task,
    which includes workload but also eye movements, motor activity and arousal.
55. **Why are some T3 scores below 0.5?** — LOSO anti-learning: holding out a subject shifts the training class
    balance against that subject's class; with no real signal, predictions anti-correlate with the truth (D16).
56. **What is the latency?** — ~100 ms per window end-to-end (median 98.4 ms), well within the 2-s update interval.

### H. System and integration
57. **What do the other components get?** — A JSON per window with workload probabilities, confidence, quality
    flag, heuristic indices (clearly marked), band power and the connectivity matrix; schema in docs.
58. **How do you stream?** — A WebSocket replays recordings in real time (speed adjustable) and sends raw display
    chunks and the state; the engine's `push()` accepts any chunk size, so a live LSL inlet can feed it.
59. **What was the CORS bug?** — A missing data file caused a 500 error, which carries no CORS header, so the browser
    reported CORS; I added a 503 handler with a hint and a Vite proxy so the browser never needs CORS.
60. **How is the dashboard kept honest?** — All numbers come from `/results/summary`, generated from result files;
    heuristic states carry a visible badge.
61. **How reproducible is it?** — Configs + seeds + hash-named data; a clean rebuild gave byte-identical features;
    105 tests.

### I. Novelty / contribution
62. **What is new?** — An end-to-end, subject-independent, leakage-audited EEG engine for the Emotiv EPOC with a
    multi-task hybrid model ready for unlabelled states, a documented integration contract, and an explicit
    demonstration that STEW's within-task workload ratings are not decodable — an important negative result
    that many papers miss because they use within-subject splits.

### J. Ethics
63. **Ethical issues?** — EEG is sensitive biometric data (subject fingerprint 99.5 %!); consent, anonymisation,
    secure storage and purpose limitation are needed; outputs must not be used to judge or grade learners; heuristic
    indices must not be presented as diagnoses; the system must allow opting out. STEW is public, de-identified data.
64. **Bias?** — 48 subjects from one study; performance may differ across ages, hair types (electrode contact)
    and cultures; the team data should be diverse.

### K. Future work
65. **What would you do with more time?** — Collect labelled team data, a graded within-task workload paradigm,
    live LSL integration, per-user calibration, GPU-trained deep models with LOSO, bootstrap CIs, SHAP explanations.

## 11.5 Tricky questions — suggested honest answers
* **"Your deep model is the centrepiece of the proposal but it loses. Did the project fail?"** — No: the system is
  complete and the hybrid is implemented, tested and ablated exactly as planned; the evaluation shows, with paired
  statistics, that on 48 subjects engineered features are better. That is a valid, reported finding, and the hybrid's
  multi-task design is what we need once our labelled data arrives.
* **"Isn't T3 at chance just a bug?"** — The same pipeline gets 0.865 on T1, the permutation control is at 0.506,
  and every model independently lands at chance on T3. A bug would not produce exactly this pattern.
* **"Why didn't you tune the deep models until they win?"** — Tuning on the same 48 subjects without a separate test
  set would leak; with a CPU budget I kept them untuned and said so.
* **"Your attention/fatigue numbers — are they predictions?"** — No. They are published band-power ratios marked
  `heuristic_index` everywhere, including the API and dashboard.
* **"Why only 3 permutations?"** — Each permutation is a full 48-fold LOSO run; 3 runs all at chance are sufficient to
  show the pipeline does not leak; more would tighten the null distribution (future work).
* **"Where is your proposal comparison?"** — The proposal document was not in the repository when the report was
  generated; the alignment table maps every requirement of the brief, and deviations are listed with reasons.
* **"Your latency was measured on a busy laptop."** — Yes; even there it is ~100 ms per window against a 2-s budget.

## 11.6 Glossary
| Term | Meaning |
|---|---|
| AUROC | Area under the ROC curve; probability a random positive is ranked above a random negative; 0.5 = chance |
| Balanced accuracy | mean recall over classes |
| Band power | power of the EEG within a frequency band (µV²) |
| Blink tolerance | frontal amplitude/variance-only failures < 300 µV treated as blinks ("warn") |
| Calibration / ECE | agreement between predicted confidence and actual accuracy / its error |
| Coherence (MSC) | frequency-domain correlation of two signals (amplitude + phase) |
| Confound | a variable that differs between classes besides the one of interest |
| Cross-validation | repeatedly training on part of the data and testing on the rest |
| DC offset | constant voltage added by the amplifier (~4333 µV on the Emotiv) |
| DWT | discrete wavelet transform; splits a signal into octave sub-bands |
| EEGNet | compact CNN for EEG (Lawhern et al., 2018) |
| EMA | exponential moving average smoothing |
| EMG | muscle electrical activity, an artefact in EEG |
| Epoch | a fixed-length window of EEG (here 4 s) |
| EOG | eye electrical activity (blinks, eye movements) |
| FIR filter | finite impulse response filter; linear phase possible |
| GAT | graph attention network |
| Hilbert transform | gives the analytic signal → instantaneous phase/amplitude |
| Hjorth parameters | activity, mobility, complexity of a signal |
| Holm correction | step-down multiple-comparison correction |
| ICA | independent component analysis |
| Leakage | information from the test set influencing training |
| LOSO | leave-one-subject-out cross-validation |
| LSL | Lab Streaming Layer, a protocol for real-time biosignal streams |
| Macro-F1 | mean F1 over classes |
| Masked loss | loss computed only on samples that have a label for that head |
| Montage | electrode positions (10-20 system) |
| Notch filter | removes a narrow frequency band (50 Hz) |
| Permutation test | re-running with shuffled labels to get the chance distribution |
| PLV | phase-locking value |
| Proportional threshold | keep the strongest fraction of graph edges |
| PSD / Welch | power spectral density / averaged-periodogram estimator |
| Rest baseline | normalising a subject's features with their own rest recording |
| SIMKAP | multitasking test used for STEW's high-workload condition |
| Spherical-spline interpolation | rebuilding a channel from neighbours on a sphere model of the head |
| STEW | Simultaneous Task EEG Workload dataset |
| Temperature scaling | dividing logits by T to calibrate probabilities |
| Transformer | self-attention sequence model |
| Volume conduction | spread of one source to many electrodes |
| Wilcoxon signed-rank | non-parametric paired test |
| XGBoost | gradient-boosted decision trees |
| Zero-phase filtering | filtering without time shift |

## 11.7 Cheat sheet — 15 numbers to memorise
| # | Number | Meaning | Source |
|---|---|---|---|
| 1 | 48 [`results/eda/dataset_summary.json` → `n_subjects`] | subjects (96 recordings) | EDA |
| 2 | 7104 [`results/preprocessing/preprocess_summary.json` → `X_shape.0`] | 4-s epochs (74 per recording) | preprocessing |
| 3 | 1499 [`results/features/feature_summary.json` → `n_features`] | features in 6 families | features |
| 4 | 30 [`results/preprocessing/preprocess_summary.json` → `n_channels_interpolated`] | channels interpolated (in 26 recordings) | preprocessing |
| 5 | 0.1225 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered_interpolated_final.bad`] | fraction of windows flagged bad | quality |
| 6 | 0.865 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `pooled.macro_f1`] | T1 XGBoost LOSO pooled macro-F1 | classical |
| 7 | 0.941 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `pooled.auroc`] | T1 XGBoost LOSO AUROC | classical |
| 8 | 0.915 [`results/cls_T1_xgb_all_none_kfold5/metrics.json` → `sequence_level.per_fold.macro_f1.mean`] | T1 XGBoost 5-fold, 18-s windows | classical |
| 9 | 0.773 [`results/deep_T1_A8_cnn_lstm_trans_kfold5/metrics.json` → `per_fold.macro_f1.mean`] | best deep (A8) on T1 | deep |
| 10 | 0.758 [`results/deep_T1_A10_hybrid_kfold5/metrics.json` → `per_fold.macro_f1.mean`] | full hybrid on T1 | deep |
| 11 | 0.538 [`results/cls_T3_xgb_all_none_loso/metrics.json` → `pooled.macro_f1`] | T3 XGBoost LOSO (≈ chance) | classical |
| 12 | 0.951 [`results/cls_T1_svm_all_rest_baseline_loso/metrics.json` → `pooled.balanced_accuracy`] | SVM T1 with rest baseline (inflated) | classical |
| 13 | 0.506 [`results/leakage_audit.json` → `permutation_test.mean_balanced_accuracy`] | permutation control (chance) | audit |
| 14 | 0.995 [`results/leakage_audit.json` → `subject_identity_probe.random_split_subject_id_accuracy`] | subject identification with random splits | audit |
| 15 | 98.4 [`results/engine_latency.json` → `median_ms`] | ms per window end-to-end (median) | engine |
