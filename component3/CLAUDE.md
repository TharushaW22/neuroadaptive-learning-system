# CLAUDE.md — Component 3: Comparative XAI evaluation (component3/)

SLIIT IT4010 project J26-DS-333, Component 3. Compares XAI methods (KernelSHAP, Conditional SHAP,
LIME, Integrated Gradients) on Component 1's EEG workload model, checks that the quality metrics are
themselves reliable, ranks the methods under uncertainty and exports a recommendation for Component 4.
The rest of the repository belongs to teammates. Read `docs/PLAN.md` and `docs/PROGRESS.md` first.

## Safety rules (override everything)
1. Create/edit/delete files ONLY inside `component3/`. Never touch component1/, `Component 4/`, the
   repo-root README/.gitignore, CI files or any other folder. Every writer in code goes through
   `xaicomp.paths.safe_path()`, which raises outside component3/.
2. Component 1 is READ-ONLY. Import via `PYTHONPATH=../component1/src` (`source env.sh`, or the
   scripts' `_bootstrap.py`). NEVER `pip install -e` Component 1. Bytecode caches go to
   `component3/.pycache` (sys.pycache_prefix / PYTHONPYCACHEPREFIX) — keep it that way.
3. NEVER run Component 1's scripts (run_all, run_preprocess, run_features, run_deep, run_baselines,
   train_final_*, export_report, api, frontend): they overwrite C1's committed results.
4. Before calling any C1 function, read it and check for side effects. If it writes inside
   component1/, pass a component3 path or write a thin wrapper in component3/ citing file:lines.
5. Never persist changes to C1 objects (checkpoints, configs). Load copies only.
6. Do not edit Component 4; the C4 export lives in component3/.
7. Python 3.13 venv in `component3/.venv`; install only there.
8. No state-changing git commands (add/commit/push/merge/rebase/reset/checkout/stash/mv). After each
   step run `git status --porcelain` from the repo root: anything outside component3/ → STOP, undo, report.
9. Never start a command expected to run > 30 min; give the student the command instead.
10. Never invent or hard-code results/p-values/rankings; never tune weights/settings to make a method win.

## Decisions (details in docs/PLAN.md)
D1 fold models retrained with C1's unchanged `run_fold` → artifacts/fold_models/ (deployed model = IN-SAMPLE fallback, flagged) ·
D2 Conditional SHAP: Gaussian on log band power, Ledoit-Wolf on fold-training subjects, donor rescaled; power check reported ·
D3 IG donor baseline = main (stochastic); IG zero baseline reported separately, never ranked ·
D4 primary eeg/T1/A8; negative control = re-initialised weights; xgb screening only ·
D5 regions in the context YAML · D6 budgets in configs/run.yaml, `--pilot` < 30 min ·
D7 C4 export = JSON schema + functions · D8 adapter interface ready for C2, no fake C2 data ·
D9 dashboard for non-specialists (plain language, Okabe–Ito, blue↔orange attributions, run-type banner).
Deviation: 6th "slow" (< 1 Hz) band → 84 features (see PLAN §4).

## Layout
- `configs/` data.yaml · run.yaml (budgets, pilot) · explainers.yaml · metrics.yaml · stats.yaml · ranking.yaml · contexts/*.yaml
- `src/xaicomp/` `_bootstrap.py` (C1 path + pycache) · `paths.py` (safe_path) · `config.py` ·
  `data/` (eeg_data: build + load; instances: manifest + selection) · `models/` (loader: fold/deployed/randomised; fold_training) ·
  `representation/` (eeg_bands, groups) · `replacement/` (strategies, checks) · `adapters/` (base, eeg_component1, tabular_component1) ·
  `screening/` · `explainers/` · `metrics/` · `reliability/` · `disagreement/` · `stats/` · `ranking/` · `export/` · `run/` (orchestrator, tables, analysis)
- `scripts/` build_data · train_fold_models · run_experiment (--pilot, --stage) · budget_study
- `dashboard/` app.py, common.py, views_a.py, views_b.py (reads results/ only) · `.streamlit/config.toml`
- `tests/` one file per step; `fixtures.py` = toy test fixtures used ONLY in tests
- `data/`, `artifacts/`, `results/_scratch/`, `.venv/`, `.pycache/` are git-ignored

## Commands (from component3/)
- Setup: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Data: `.venv/bin/python scripts/build_data.py` · Tests: `.venv/bin/python -m pytest` (`-m "not slow"` for quick)
- Fold models (student runs; ~30–45 min): `.venv/bin/python scripts/train_fold_models.py --context eeg_T1_A8`
- Pilot: `.venv/bin/python scripts/run_experiment.py --pilot` · Full: `.venv/bin/python scripts/run_experiment.py`
- Dashboard: `.venv/bin/streamlit run dashboard/app.py`

## Conventions
- Small, commented code a final-year student can explain in a viva; docs/CODE_GUIDE.md explains every module.
- Update docs/PROGRESS.md after every step and docs/ai_use_log.md for every AI-generated piece.
- Results are labelled PILOT / IN-SAMPLE / FINAL everywhere (run_info.json → banner).
