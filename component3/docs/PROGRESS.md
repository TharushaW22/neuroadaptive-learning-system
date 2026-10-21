# Progress — Component 3

Updated continuously so work can resume after an interruption. Machine: Apple M1 (4 performance
+ 4 efficiency cores), 8.6 GB RAM, Python 3.13.7, CPU only. Date: 2026-10-20.

Git-status rule: after every step `git status --porcelain` (repo root) listed only `component3/`
paths, and `find component1 -newer component3/docs/PLAN_DRAFT.md` returned nothing.

## Step status

| Step | Status | Tests | Notes |
|---|---|---|---|
| 0 Setup | done | `test_step0_setup.py` | venv 3.13; predictor 130,616 params, T = 1.1011, [low, high]; pycache redirected to `.pycache`; `safe_path` guard; git-status test |
| 1 Data | done | `test_step1_data.py` | STEW present; 7,104 epochs (74/recording), T1 3552/3552, T3 1480/1850 = C1 `labels.md`; hash 6d7dd612; build 64 s |
| 2 Fold models (D1) | **implemented, tested on 1 fold × 1 epoch; full retraining NOT run (rule 10)** | `test_step2_fold_training.py` | 1 epoch ≈ 30 s here. C1's runs stopped after 15/9/24/10/20 epochs (78 in total) → ≈ 40–45 min for all folds. Command below. Until then every result is IN-SAMPLE. |
| 3 Adapter + bands | done | `test_step3_representation.py`, `test_step3_adapter.py` | max \|p_C3 − p_C1\| = 5.2e-8 (≤ 1e-5 required); reconstruction error 3e-5 µV (float32); residual 0.03 % of power with the slow band; gradients finite. **Deviation:** 6th "slow" band (< 1 Hz): 67 % of window power lies below 1 Hz after C1's 0.5–45 Hz FIR; dropping it flips 21 % of labels → 84 features |
| 4 Manifest + screening | done | `test_step4_manifest_screening.py` | 3,264 windows (= C1's T1 test sequences); 8-check screening; DiCE + Anchors excluded (no per-feature attribution; DiCE needs actionable features); IG excluded for the XGBoost context (not differentiable) |
| 5 Explainers | done | `test_step5_explainers.py` | 5 real windows checked (2 in the interactive check, 3 in the budget study): KernelSHAP/Conditional SHAP efficiency error 0.0; IG completeness error 3e-5 – 0.013 (≤ 3 % relative, 16 steps); LIME R² 0.79–0.83. D2 power check: mean \|error\| 0.013–0.018 log10 (≈ 0.15 dB) vs 0.005–0.016 for re-measuring the unmodified window → Gaussian mode kept, empirical mode available |
| 6 Fidelity | done | `test_step6_7_metrics.py` | toy model with known weights: MoRF ≤ random ≤ LeRF; real-data check in the pilot (below) |
| 7 Other metrics | done | `test_step6_7_metrics.py` | stability, robustness (σ = 0.1 × channel scale, label kept), sparsity (Gini), cost, agreement |
| 8 Orchestrator + pilot | done (pilot) | smoke run + pilot | resumable (refuses to resume into a run with another model source/budget); batch 16 (128 thrashed memory: 227 ms/eval); see "Pilot" |
| 9 Reliability | done | `test_step9_reliability.py` | ICC(2,1), degradation, redundancy, randomised-weights control |
| 10 Disagreement | done | `test_step10_11_disagreement_stats.py` | corrected agreement (> 1 flagged), MixedLM, KS-vs-CS |
| 11 Statistics | done | same | subject level; Friedman, Wilcoxon-Holm, rank-biserial, cluster bootstrap, baycomp ROPE |
| 12 Ranking | done | `test_step12_13_ranking_export.py` | reliability-based criteria; TOPSIS; SMAA × 3 MC seeds |
| 13 C4 export | done | same | JSON validates against schema 1.0 |
| 14 Dashboard | done | `test_step14_dashboard.py` | 11 pages; AppTest renders each page from the pilot results |
| 15 C2 adapters | waiting for Component 2 | – | interface ready (`adapters/base.py`) |

## Measured costs (budget study, `results/pilot/budget_study.json`)

3 real windows, 2 seeds each, 3 parallel single-threaded workers. Timings in this table were taken
while another job shared the CPU (≈ 28–31 ms per evaluation; uncontended 4-worker figure below:
12.4 ms) — use them for the RELATIVE cost of budgets only. A larger grid (first study,
`budget_study_large_grid_contended.json`) gives consistent stability values: KernelSHAP 1000×8 0.47,
2000×4 0.50; LIME 1000×8 0.54; IG 32×16 0.85.

| Method | Budget | Mean stability (Spearman between seeds) | Seconds / explanation |
|---|---|---|---|
| KernelSHAP | 300 coalitions × 2 donors | 0.19 | 19 |
| KernelSHAP | 1000 × 2 | 0.35 | 58 |
| KernelSHAP | 500 × 8 | 0.35 | 121 |
| KernelSHAP | 1000 × 4 | 0.48 | 119 |
| LIME | 300 × 2 | 0.14 | 17 |
| LIME | 1000 × 4 | 0.49 | 121 |
| IG (donor) | 16 steps × 2 donors | 0.50 | ~2 |
| IG (donor) | 32 × 4 | 0.74 | ~5 |
| IG (donor) | 32 × 8 | 0.74 | ~9 |
| IG (donor) | 32 × 16 | 0.91 | ~19 |

Reading: stability is limited mostly by the number of donor windows (the "absent" distribution)
and by the coalition count. Conditional SHAP costs the same as KernelSHAP per evaluation.

## Pilot run (`results/pilot/`) — labelled **PILOT + IN-SAMPLE**

Command: `.venv/bin/python scripts/run_experiment.py --pilot --workers 4`. Duration **30.5 min**,
no errors. Caveat: an earlier budget-study job (thought to have died — macOS lists it as "Python")
was still running on 3 cores during the pilot, so the pilot's timings are ≈ 3× slower than
uncontended (41 ms vs 12.4 ms per evaluation per worker). An uncontended pilot would take ≈ 12–15 min.

Scope: 6 subjects (one or more per C1 fold), 2 windows each (rest + task) = 12 windows; 2 seeds;
300 coalitions × 2 donors (KernelSHAP, Conditional SHAP, LIME); IG 16 steps × 2 donors;
negative control: 5 windows (one per fold). Deployed model → every output IN-SAMPLE.

Results — **PILOT + IN-SAMPLE, not for final conclusions:**

* Fidelity sanity (Step 6 done-check): for every method and both removal modes the random-order
  AUC lies between the method's own order and its reverse (e.g. on-manifold deletion: KernelSHAP
  0.42 < random 0.69 < reverse 0.83). ✔
* Checks: KernelSHAP / Conditional SHAP efficiency error 0.0; IG completeness error 1.3 % (donor),
  0.3 % (zero) relative; LIME surrogate R² 0.72.
* Mean per window: fidelity (realistic removal) IG 0.22, KernelSHAP 0.21, LIME 0.20, Conditional
  SHAP 0.12; stability IG 0.47, LIME 0.25, KernelSHAP 0.22, Conditional SHAP 0.14 (IG-zero 1.0,
  not rewarded); robustness ≈ 0.997 for all (not discriminating); sparsity IG 0.76 vs 0.42–0.49.
* Reliability: ICC fidelity_on 0.51 (moderate → weight × 0.51), fidelity_off 0.76, sparsity 0.96,
  cost 0.98; stability / robustness ICC not assessable (2 seeds, 1 noise draw → assessable in the
  full run for stability: 3 seeds). All criteria fall monotonically under degradation (ρ −0.9 to −1.0).
  No redundant pair (max |ρ| 0.61).
* Negative control: fidelity collapses on the random-weights model (on-manifold 0.218 → 0.005,
  ratio 0.02); every method passes the randomisation sanity check (mean |Spearman(trained, random)| 0.11–0.22).
* Disagreement: raw agreement between main methods 0.17–0.42; corrected values > 1 for 4 pairs
  (flagged: self-agreement too low at pilot budget). MixedLM converged (72 window-pairs, 6 subjects):
  higher model confidence and more bad epochs → more disagreement, more locally linear model (LIME R²)
  → less. KernelSHAP vs Conditional SHAP disagreement 0.75 vs 0.70 for other pairs.
* Statistics (6 subjects): Friedman p < 0.05 for stability, robustness, sparsity, cost; not for fidelity.
* Ranking: IG (donor) P(rank 1) = 1.00 on all 3 Monte-Carlo seeds (max difference 0.0); IG vs
  KernelSHAP not practically equivalent (P(IG better) 0.9998). With 6 subjects, an in-sample model
  and a tiny budget this is a pipeline check, not a finding.
* Export: `results/pilot/component4_export.json` — 1 context, 12 windows, validates against schema 1.0.

## Recommended full-run budget (D6) and time estimate

Measured uncontended throughput (`scripts/measure_throughput.py`, `results/pilot/throughput.json`):
1 worker 8.6 ms/eval (116 eval/s); 2 workers 219 eval/s; **4 workers 322 eval/s** (12.4 ms each).

Recommended (`configs/run.yaml` → `full`): 48 subjects × **4 windows** (2 rest + 2 task) = 192
windows; **3 seeds**; KernelSHAP / Conditional SHAP / LIME **1000 coalitions × 4 donors**; IG
**32 steps × 4 donors**; IG-zero 32 steps; fidelity 4 donors; 1 noise copy; negative control
2 subjects per fold × 2 windows = 20 windows; fidelity-degradation check on 24 windows.
Why: the budget study shows stability rising with donors and coalitions (KernelSHAP 300×2 → 0.19–0.29,
1000×4 → 0.42–0.48, 2000×4 → 0.50; LIME 1000×4 → 0.49); 1000×4 is the best value per second
measured; every method uses the same K = 4 so all share the same "absent" distribution (giving IG
K = 16 would raise its stability to ~0.9 but would favour it — offered only as a sensitivity analysis).

Cost per window ≈ 56k model evaluations (3 seeds + 1 noisy copy × ~12k, + ~6k fidelity) →
≈ 175 s at 322 eval/s → 192 windows ≈ 9.3 h + control ≈ 1 h + analysis ≈ 0.3 h → **≈ 10–11 h**
with 4 workers and nothing else running (2 windows per subject: ≈ 6 h). Resumable at any time.

## Commands for the student

```bash
cd component3
# D1 fold models (≈ 40–45 min, all 5 folds; or one fold at a time with --folds 0 ... --folds 4)
.venv/bin/python scripts/train_fold_models.py --context eeg_T1_A8
# then the full experiment (resumable; re-run the same command after an interruption)
.venv/bin/python scripts/run_experiment.py
# dashboard
.venv/bin/streamlit run dashboard/app.py
```
After the fold models exist, `ModelBank` switches to them automatically and the run is labelled FINAL.
Delete or rename `results/full/` before re-running if the model source changed (resume would otherwise
mix in-sample and out-of-sample explanations).
