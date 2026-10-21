# AI use log (Claude Code)

One row per step. "Verify" = what the student must check personally before relying on it
(read the code, re-run the test, compare with the paper, be able to explain it in the viva).

| Date | Step | What Claude generated | What the student must verify |
|---|---|---|---|
| 2026-10-20 | 0 Setup | `requirements.txt`, `pyproject.toml`, `.gitignore`, `env.sh`/`env.ps1`, `conftest.py`, `_bootstrap.py` (C1 path + pycache redirect), `paths.safe_path` write guard, configs | venv uses Python 3.13; `git status` shows nothing outside component3/; `component1/` has no new `__pycache__`; predictor facts (130,616 params, T 1.101) |
| 2026-10-20 | 1 Data | `data/eeg_data.py`, `scripts/build_data.py` (C1 library calls in C1's order, outputs in component3/data) | the function order matches C1 `run_preprocess.py:60-101`; 7,104 epochs; counts vs C1 `docs/labels.md`; config hash 6d7dd612 |
| 2026-10-20 | 2 Fold models | `models/fold_training.py`, `scripts/train_fold_models.py` (C1 `run_fold` imported unchanged) | the loop mirrors `deep_runner.py:31-56`; run the full retraining yourself; pooled macro-F1 within ±0.03 of 0.774 |
| 2026-10-20 | 3 Adapter + bands | `representation/eeg_bands.py`, `groups.py`, `adapters/*`, `models/loader.py`; decision to add the < 1 Hz "slow" band (measured: ~2/3 of power, 21 % label flips without it) | parity with C1's predictor (≤ 1e-5); filter bank sums to 1; whether the slow band is acceptable to you and your supervisor (or switch it off in the context YAML) |
| 2026-10-20 | 4 Manifest + screening | `data/instances.py`, `screening/applicability.py`, `adapters/tabular_component1.py` | the method requirements table (what each method needs) against the original papers; the DiCE/Anchors exclusion reasons |
| 2026-10-20 | 5 Explainers | `replacement/strategies.py`, `replacement/checks.py`, `explainers/*` | Gaussian conditioning formula; KernelSHAP efficiency and IG completeness numbers; that LIME's sampling matches the LIME text explainer |
| 2026-10-20 | 6–7 Metrics | `metrics/*` (fidelity, stability, robustness, sparsity, agreement, cost) | the fidelity score definition; the on-manifold circularity caveat; robustness label-flip rule |
| 2026-10-20 | 8 Orchestrator | `run/orchestrator.py`, `scripts/run_experiment.py`, `scripts/budget_study.py`; batch size 16 after measuring memory thrash at 128 | resume behaviour (stop and restart a run); run_info labels (PILOT / IN-SAMPLE / FINAL) |
| 2026-10-20 | 9 Reliability | `reliability/*` (ICC(2,1), degradation, redundancy, randomisation control) | ICC formula (Shrout & Fleiss); thresholds in `configs/ranking.yaml` are justified in your write-up |
| 2026-10-20 | 10 Disagreement | `disagreement/*` (correction for attenuation, MixedLM) | the model formula; interpretation of corrected values > 1 |
| 2026-10-20 | 11 Statistics | `stats/*` (Friedman, Wilcoxon-Holm, rank-biserial, cluster bootstrap, baycomp) | baycomp's probability order (P(A better), P(ROPE), P(B better)) — checked by a test; ROPE widths in `configs/stats.yaml` |
| 2026-10-20 | 12 Ranking | `ranking/*` (criteria rules, TOPSIS, SMAA) | that no weight favours a method (Dirichlet(1) × reliability factors only); P(rank 1) stable across MC seeds |
| 2026-10-20 | 13 Export | `export/schema.py`, `export/component4.py` | the schema with the Component 4 owner; trust-indicator definition |
| 2026-10-20 | 14 Dashboard | `dashboard/*`, `.streamlit/config.toml`; palette validated with a colour-blindness checker (LIME = sky blue, green kept for status) | every page reads naturally for a non-specialist; the plain-language summaries are correct for your final results |
| 2026-10-20 | Docs | `PLAN.md`, `README.md`, `CLAUDE.md`, `PROGRESS.md`, `CODE_GUIDE.md`, this log, `questions_for_team.md` | you can explain each module in your own words (CODE_GUIDE) |
