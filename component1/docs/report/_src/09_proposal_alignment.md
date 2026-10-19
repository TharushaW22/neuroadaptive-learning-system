# 9. Proposal alignment

**The proposal document (`docs/proposal.docx`) is NOT AVAILABLE in the repository**, so proposal section
numbers (§1.4, §3, §4.6–4.15) could not be checked against the text. The table maps every requirement of
the author's brief — which summarises those sections — to its implementation. Where the brief named a
proposal section, it is shown; otherwise "brief".

| Requirement (source) | Implemented? | Where | Evidence | Deviation + reason |
|---|---|---|---|---|
| Reproducible Python engine, config-driven, seeds fixed (brief) | Yes | `configs/`, `src/eegengine/config.py`, `scripts/run_all.py` | byte-identical rebuild (§8.7) | — |
| STEW loader, 96 recordings, labels, MNE RawArray + 10-20 montage (brief) | Yes | `io/stew.py` | 19 loader tests | `colin27_1020` = `standard_1020` positions (D5) |
| Missing ratings masked, never dropped (brief) | Yes | `labels.py` | `docs/labels.md` | — |
| Signal quality: flatline, amplitude, robust variance z, HF/line-noise; mark not delete; CSV + JSON (brief) | Yes | `quality/` | `results/quality/`, `results/preprocessing/` | EMG band 30–45 Hz instead of > 45 Hz (D4); detrended windows (D6); blink tolerance (D7) |
| DC removal, 0.5–45 Hz zero-phase FIR, 50 Hz notch (brief, proposal preprocessing section) | Yes | `preprocess/filters.py` | filter tests, PSD figures | — |
| Optional ICA with AF3/AF4 as EOG proxies, default OFF, log components (brief) | Yes (OFF) | `preprocess/ica.py` | `results/preprocessing/ica_demo_log.json` | effect on accuracy not evaluated |
| Persistent bad channels interpolated (spherical spline), logged (brief) | Yes | `preprocess/bad_channels.py` | `results/preprocessing/interpolation_log.csv` | rule refined to median-window std (D8) |
| Epoching 4 s / 50 % overlap; also test 2 s (brief) | Partly | `preprocess/epoching.py`, `configs/preprocess_2s.yaml` | 7104 epochs | **2-s epochs generated but not evaluated** (time, D14) |
| Normalisation none / epoch z / rest baseline; report with and without baseline (brief) | Yes | `preprocess/normalize.py` | §7.3 | T2/T3 baseline comparison only for XGBoost (D14) |
| T1, T2, T3 labels; T3 as confound-free primary result (brief) | Yes | `labels.py` | `docs/labels.md` | — |
| Features: Welch abs/rel band power, ratios, frontal asymmetry (brief, proposal features section) | Yes | `features/spectral.py` | 185 features | — |
| Spectral entropy, Hjorth, variance, skewness, kurtosis (brief) | Yes | `features/entropy.py` | 84 features | Hjorth activity stored once as log variance |
| DWT db4 5 levels: energy, relative energy, std, wavelet entropy (brief) | Yes | `features/wavelet.py` | 266 features | — |
| PLV (Hilbert) + magnitude-squared coherence per band; matrices saved + upper triangles (brief) | Yes | `features/connectivity.py` | 2 × 364 features, `features_*_arrays.npz` | — |
| Graph metrics from PLV with proportional/absolute/weighted thresholding (brief) | Yes | `features/graph.py` | 236 features | proportional 30 % used |
| Tests: PLV identical = 1, random ≈ 0, band sums (brief) | Yes | `tests/test_features.py` | 18 tests | — |
| Classical pipelines LR/SVM/RF/XGB, fit inside folds (brief, proposal models section) | Yes | `models/classical.py` | §5 | fixed hyper-parameters, smaller RF/XGB (D12) |
| LOSO for T1, T2, T3; per-subject aggregation for T3 (brief) | Yes | `training/cv.py`, `classical_runner.py` | `results/cls_*_loso` | 4 workers / reduced grid on the shared laptop (D14) |
| Feature ablation A1–A5 + all with best model (brief / proposal ablation A1–A5) | Yes | `scripts/run_ablation.py` | §5.4 | T2 only "all" with baseline (D15) |
| Wilcoxon signed-rank on per-subject scores (brief) | Yes | `evaluation/ablation.py` | 35 + 14 tests | Holm correction added |
| Metrics macro-F1, bal. acc., precision, recall, AUROC, confusion, latency; mean ± std (brief) | Yes | `evaluation/metrics.py` | §7 | — |
| Deep inputs: raw epochs, connectivity matrices, sequences of 8 (brief) | Yes | `training/trainer.py`, `sequences.py` | §6.1 | — |
| EEGNet (brief) | Yes | `models/deep/eegnet.py` | §6.2 | temporal kernel 32 instead of 64 (D17) |
| CNN-LSTM (brief) | Yes | `models/deep/cnn_lstm.py` | §6.3 | — |
| Hybrid per proposal §4.10: CNN, LSTM, Transformer (2 layers + PE), GAT in plain PyTorch, fusion, 4 heads, masked multi-task loss, switchable branches (brief) | Yes | `models/deep/hybrid.py`, `gat.py` | §6.4 | GAT uses one mean-PLV graph |
| AdamW, early stopping on subject-wise inner validation, class weights, augmentation, seeds (brief) | Yes | `training/trainer.py` | §6.5 | small validation sets make early stopping noisy |
| 5-fold by subject, same folds for all deep models; T1 and T3 (T2 optional) (brief) | Yes (T2 not run) | `results/folds_5fold.json` | 12 deep experiments | T2 deep not run (optional in the brief) |
| Architecture ablation A6–A10 (brief / proposal A6–A10) | Yes | `configs/experiments/deep.yaml` | §6.6 | — |
| Temperature scaling, ECE before/after, reliability diagram (brief) | Yes | `training/calibration.py` | §6.7 | does not improve every model |
| CPU latency per window (brief) | Yes | `trainer.py:measure_latency`, `results/engine_latency.json` | §6.8 | — |
| Best checkpoints in `models/`; summary_deep.md; summary_all.md (brief) | Yes | `models/`, `results/summary_*.md` | — | — |
| CognitiveEngine: quality → preprocess → features → model → EMA → pydantic output (brief) | Yes | `inference/engine.py` | §8.2 | — |
| Heuristic attention/fatigue/readiness with documented formulas, marked heuristic (brief) | Yes | `inference/heuristics.py` | — | fatigue uses posterior channels (else = 1/engagement) |
| Output contract doc + JSON Schema for teammates (brief) | Yes | `docs/output_contract.md`, `.schema.json` | — | — |
| FastAPI: health, subjects, models, results, schema, predict, WS stream; CORS; API tests (brief) | Yes | `api/main.py` | 15 engine/API tests | added `/quality`, `/contract/sample`, 503 handler, Vite proxy |
| React dashboard: 6 pages, dark/light, projector, demo mode, no hard-coded numbers (brief) | Yes | `frontend/` | `npm run build` clean | browser console not inspected by the author's tooling |
| run_all, leakage guard tests, leakage audit, README, viva notes, Docker (brief) | Yes (Docker untested) | `scripts/run_all.py`, `docs/` | 105 test cases | Docker not built (NOT AVAILABLE) |
| Proposal §1.4 scope items | NOT AVAILABLE | — | — | proposal document missing |
| Proposal §3 objectives | mapped via the brief in `01_overview.md` §1.3 | — | — | proposal document missing |
| Proposal §2.14 STEW as pilot dataset | consistent | — | — | proposal document missing |

## 9.1 Still planned for the team-collected data
| Item | Status | What is ready | What is missing |
|---|---|---|---|
| Attention / fatigue / readiness labels | Not started (no data) | masked heads in the hybrid; label CSV format in README; heuristic indices as placeholders | labelled recordings; label columns + masks; retraining; validation |
| Live acquisition via LSL (Lab Streaming Layer) | **Not implemented** | `CognitiveEngine.push(chunk)` accepts any chunk size; streaming = batch is tested | an LSL inlet (e.g. `pylsl`) that reads the Emotiv stream and calls `push` |
| Emotiv integration (EmotivPRO / Cortex API) | **Not implemented** | channel order, 128 Hz, µV with DC offset are already the expected input | API credentials/SDK, mapping of the Emotiv stream to `push`, device contact-quality |
| Per-user calibration session | Partly | rest-baseline normalisation implemented and evaluated | protocol for a short baseline before each session; with/without comparison on team data |
| Multimodal fusion with the computer-vision component | Interface ready | timestamped JSON + quality flag + calibrated probabilities | fusion model (other component) |
