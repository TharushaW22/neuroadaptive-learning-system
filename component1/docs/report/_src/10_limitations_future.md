# 10. Limitations and future work

## 10.1 Limitations (honest list)
1. **Workload-only labels.** STEW labels only workload (condition + self-rating). Attention, fatigue and
   learning readiness are heuristic band-power indices, not trained or validated predictions.
2. **Rest-vs-task confound.** T1 compares sitting still with a demanding visual-motor task; eye movements,
   hand movements and arousal differ, so T1 accuracy overstates what is learned about mental workload.
3. **T2 ≈ T1.** Rest is rated low (42/3/0) and task medium/high (0/20/25); T2 mostly re-measures T1.
4. **T3 is small and at chance.** 45 labelled recordings, one per subject; no model beats chance. This limits
   any claim of graded workload detection.
5. **Small N for deep learning.** 48 subjects (≈ 38 training subjects per fold) is small for a 141 k-parameter
   hybrid; EEGNet with 1.6 k parameters performs as well.
6. **CPU-only, lightly tuned deep models.** ≤ 25 epochs, fixed hyper-parameters, kernel 32, early stopping on
   ≈ 6 validation subjects (best epoch 0 in T1 fold 1 for every model).
7. **Deep models on 5 folds, not LOSO**, so classical-LOSO and deep numbers are not directly comparable (the
   paired 5-fold comparison is).
8. **Calibration is fragile** — temperature scaling from small validation sets improved some models and hurt
   others; T hits its bounds on T3.
9. **Fixed classical hyper-parameters** (no nested tuning) and a reduced experiment grid (D14, D15).
10. **ICA off; no EOG channel.** Blinks are flagged, not removed; frontal channels carry eye artefacts.
11. **Consumer headset artefacts.** Final flags: bad 0.1225 [`results/preprocessing/quality_raw_vs_filtered.json` → `filtered_interpolated_final.bad`]
    of windows; 30 channels interpolated in 26 recordings; two recordings < 50 % usable.
12. **Volume conduction** inflates PLV/coherence between neighbouring electrodes; no lag-insensitive measure.
13. **2-s epochs not evaluated**, bootstrap CIs and rating correlation (Spearman) not computed.
14. **Online vs offline filtering differ slightly** (20-s context; band power within 5 %).
15. **Heuristic indices are ratios** with person-specific scales; readiness is an ad-hoc combination.
16. **No live headset integration** (LSL/Emotiv) and Docker images untested.
17. **Single dataset**; no external validation; the proposal document was not available to check alignment.
18. **Shared development machine** — runtime/latency measurements vary with background load.

## 10.2 Future work (prioritised)
| Priority | Work | Why | What is needed |
|---|---|---|---|
| 1 | Collect the team Emotiv dataset with attention/fatigue/readiness labels | the core goal of the component; replaces heuristic indices | protocol + ethics approval, labelling scheme (README format), ≥ 20–30 participants |
| 2 | Design tasks with graded workload *within* one condition (e.g. n-back levels) | T3 showed STEW cannot test graded workload; a within-task manipulation gives clean labels | task software, counterbalancing |
| 3 | Live acquisition (LSL inlet → `CognitiveEngine.push`) | real-time use in the learning system | `pylsl`, Emotiv Cortex access, device contact-quality |
| 4 | Per-user calibration protocol (short rest + easy-task baseline before a session) and evaluate with/without | between-subject variability dominates; calibration is realistic in deployment | protocol + evaluation on team data (baseline from a *separate* recording) |
| 5 | Deep-model improvements: GPU training, hyper-parameter search, LOSO, larger validation sets, pre-training on larger public EEG corpora, band-specific graphs | close the gap to XGBoost | GPU access, time |
| 6 | Statistics: bootstrap CIs, Spearman correlation of P(high) with ratings, more permutations | stronger evidence | compute time only |
| 7 | Artefact handling: evaluate ICA on/off, lag-insensitive connectivity (wPLI, imaginary coherence) | reduce eye/EMG and volume-conduction effects | compute time |
| 8 | Evaluate 2-s epochs (already preprocessed) | faster response, more windows | compute time |
| 9 | Explainability for the XAI component: feature importances / SHAP for XGBoost, attention maps for the GAT | trust and interpretability | integration with the XAI component |
| 10 | Build and test the Docker images; CI pipeline running the tests | deployment | a machine with Docker |
