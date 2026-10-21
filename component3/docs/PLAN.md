# Component 3 — Comparative XAI evaluation: final plan

Project J26-DS-333 (SLIIT IT4010), Component 3. This is the approved plan (`PLAN_DRAFT.md`) with the
student's decisions D1–D9 folded in and the deviations found while building it. The status of each
step is tracked in `PROGRESS.md`.

## 1. Research question

Which explanation (XAI) method should be trusted for cognitive-state predictions, judged on
explanation quality (fidelity, stability, robustness, sparsity, cost), on metrics that are
themselves shown to be reliable, with subject-level statistics and a ranking that does not depend
on one person's choice of weights? The answer is delivered to Component 4 per prediction context.

## 2. Data and model (from Component 1, read-only)

* STEW: 48 subjects × rest (`lo`) / SIMKAP task (`hi`), 14 Emotiv channels, 128 Hz, 150 s each.
  Placed by the student in `component3/data/raw/STEW/`. Cleaned and epoched with **Component 1's
  library functions** (not its scripts), output only in `component3/data/processed/`.
  7,104 epochs of 4 s (2-s step); label counts identical to C1's `docs/labels.md`.
* Primary model: `T1_A8_cnn_lstm_trans` (EEGNet encoder + LSTM + Transformer; 130,616 parameters,
  temperature 1.101; pooled CV macro-F1 0.774). Input: 8 consecutive epochs = 18-s window.
* Component 1 is imported through `PYTHONPATH=../component1/src` (never `pip install -e`), and all
  bytecode caches are redirected to `component3/.pycache` so nothing is written into `component1/`.

## 3. Decisions (D1–D9)

| | Decision |
|---|---|
| D1 | Retrain per-fold models inside component3/ with C1's **unchanged** `run_fold`, same `folds_5fold.json`, seeds and hyper-parameters → `artifacts/fold_models/` (git-ignored). Done-check: pooled out-of-fold macro-F1 within ±0.03 of C1's 0.774. Until they exist, the deployed checkpoint is used and every output carries `in_sample=True` and the label "IN-SAMPLE — deployed model trained on all subjects; not for final results". |
| D2 | Conditional SHAP = Gaussian conditional on the vector of log band powers (Ledoit-Wolf covariance fitted on the instance's fold-training subjects only); donor waveform rescaled to the sampled power; check that the composed signal's band power matches the target (reported); limitation: phase/connectivity not conditioned. Empirical-conditional alternative (nearest donors on present features) selectable in `configs/explainers.yaml`. |
| D3 | Integrated Gradients with both baselines: stochastic donor baselines (main variant, used for stability and self-agreement) and the zero baseline (reported separately, flagged deterministic, never rewarded, never ranked). |
| D4 | Contexts: primary `eeg/T1/A8_cnn_lstm_trans`; negative control = same model with **re-initialised weights** (Adebayo et al., 2018); T3 and A10 skipped; optional `eeg/T1/xgb_all` used for the applicability screening only (IG screened out). |
| D5 | Regions (pending C1 confirmation): frontal AF3 F7 F3 FC5 FC6 F4 F8 AF4; temporal T7 T8; parietal P7 P8; occipital O1 O2. In the context config. |
| D6 | All budgets in `configs/run.yaml`; `--pilot` mode < 30 min; measured costs and the recommended full budget in `PROGRESS.md`. |
| D7 | Component 4 export inside component3/ only: JSON schema + Python functions (recommended explainer, P(rank 1), practical equivalence to the runner-up via Bayesian ROPE, per-window top-k region/band/channel×band with a trust indicator). |
| D8 | ModelAdapter interface + config ready for vision/fusion adapters; no fake Component 2 models or data. |
| D9 | Streamlit dashboard for non-specialists: plain language, glossary, Okabe–Ito colours, blue↔orange diverging attributions, status colours only for status, run-type banner, research-order navigation. |

## 4. Deviations found during the build (documented, config-switchable)

1. **84 features instead of 70 — a sixth "slow" band (< 1 Hz).** After C1's 0.5–45 Hz FIR, about
   two thirds of the cleaned window power lies below 1 Hz (wide transition band; drifts, eye
   movements). Leaving it in the never-explained residual would hide input the model uses
   (removing it flips ~21 % of A8's labels). The slow band is therefore an explained feature;
   deleting one line in the context YAML restores the 70-feature space. The Gaussian conditional
   (D2) is then on the 84-d log band-power vector.
2. **Parallel single-threaded workers, small batches.** One model evaluation costs ~17 ms in a
   single process; batches above ~32 windows thrash the 8.6 GB machine's memory (128 windows:
   ~227 ms each), so batches are 16. Windows run in parallel single-threaded processes (3 workers:
   ~30 ms each, ~1.7× the throughput of one process; the M1 has 4 performance cores).
3. **LIME's colour is Okabe–Ito sky blue, not green**, because green is reserved for "pass" status (D9).

## 5. Pipeline

```
STEW ─(C1 library)→ cleaned continuous recordings + epochs            scripts/build_data.py
     → fold models (C1 run_fold)                                      scripts/train_fold_models.py
     → windows (18 s) → 14 ch × 6 bands components + residual          representation/eeg_bands.py
     → instance manifest + seeded stratified selection                 data/instances.py
     → applicability screening (8 checks)                              screening/applicability.py
     → KernelSHAP · Conditional SHAP · LIME · IG (donor) · IG (zero)   explainers/
     → fidelity (deletion/insertion; zero + donor removal; random/LeRF references), stability,
       robustness (σ = 0.1 × channel scale, label kept), sparsity, cost, agreement   metrics/
     → reliability: ICC test-retest, degradation monotonicity, redundancy, randomised-weights control
     → disagreement: noise-corrected agreement, MixedLM (+ value-function effect KS vs CS)
     → statistics at subject level: Friedman, Wilcoxon-Holm, rank-biserial, cluster bootstrap, baycomp ROPE
     → ranking: reliability-based criteria → TOPSIS → Monte-Carlo SMAA (several seeds)
     → Component 4 export (JSON schema + functions) → Streamlit dashboard (reads results/ only)
```

## 6. Steps

| # | Step | Done when |
|---|---|---|
| 0 | venv, requirements, C1 importable | predictor loads (130,616 params, T 1.101, [low, high]); git clean outside component3/ |
| 1 | data | 7,104 epochs; counts = C1 labels.md |
| 2 | fold models (D1) | tested on 1 fold, tiny budget; full command given to the student |
| 3 | adapter + bands + torch module | predict = C1 within 1e-5; reconstruction reported; gradients finite |
| 4 | manifest + screening | manifest per window; 8-check table; DiCE/Anchors excluded with reasons |
| 5 | four explainers | 5 real windows: SHAP efficiency, IG completeness small; common format; cost logged |
| 6 | fidelity | random order between best and worst |
| 7 | other metrics | stability, robustness, sparsity, cost, agreement |
| 8 | orchestrator + pilot | resumable; pilot < 30 min |
| 9 | metric reliability | ICC, degradation, redundancy, negative control |
| 10 | disagreement | corrected agreement (flag > 1); MixedLM; KS vs CS |
| 11 | statistics | subject level; Friedman/Wilcoxon-Holm/effect sizes; bootstrap; baycomp ROPE |
| 12 | TOPSIS + SMAA | P(rank 1) per method, stable across MC seeds |
| 13 | C4 export | JSON validates |
| 14 | dashboard | reads results/ only; D9 |
| 15 | C2 adapters | waits for Component 2 |

## 7. Known limitations

* Conditional SHAP conditions on band power only (not phase or connectivity).
* On-manifold fidelity uses donor replacement, the same removal as KernelSHAP/LIME/IG-donor —
  it may favour them; off-manifold (zero) fidelity favours IG-zero. Both are reported, and the
  reliability analysis decides whether each enters the ranking.
* 48 subjects; T1 is rest-vs-task, so "workload" is confounded with task engagement (C1 docs).
* Until the fold models are trained, all results are IN-SAMPLE.
