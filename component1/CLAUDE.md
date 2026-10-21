# CLAUDE.md — EEG Neuro-Cognitive Understanding Engine (component1/)

SLIIT IT4010 final-year research, project **J26-DS-333**, Component 1 of the group project
"Neuroadaptive Learning System Using EEG, Computer Vision, and Agentic AI".
This folder (`component1/`) is the `eeg-engine` root. The rest of the git repo belongs to teammates.

## Goal
EEG acquisition → signal quality → preprocessing → epoching → features (spectral, entropy,
wavelet, PLV, coherence, graph) → classical baselines + hybrid CNN-LSTM-Transformer-GAT →
calibrated cognitive-state probabilities → timestamped JSON for other components → FastAPI + React dashboard.

## Dataset: STEW (data/raw/STEW/, read-only, git-ignored)
- 48 subjects; `subXX_lo.txt` = 2.5 min rest, `subXX_hi.txt` = 2.5 min SIMKAP task.
- 19200 rows × 14 cols, whitespace, no header, 128 Hz, Emotiv EPOC channel order
  AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4. Loader returns (14, 19200) in µV.
- ~4000 µV DC offset → always `remove_dc` (mean removal + 0.5 Hz zero-phase HP) first.
- `ratings.txt`: `subject, rest, task` (1–9). Subjects **5, 24, 42 have no rating** → NaN + mask.

## Labels — be honest everywhere (code, reports, UI)
- T1 `workload_binary`: lo=0, hi=1, all 48 subjects.
- T2 `workload_level`: 1–3 low(0), 4–6 medium(1), 7–9 high(2); each recording uses its own
  rating; missing → -1 / `level_mask=False`. Never drop subjects from T1.
- T2 is heavily confounded with condition (rest is almost always "low").
- Attention / fatigue / readiness have **no labels** in STEW: heads exist but are masked off
  (config flag). The dashboard shows heuristic indices labelled "heuristic index — not a trained prediction".

## Methodology rules (non-negotiable)
- Split by subject. LOSO for classical; 5-fold GroupKFold by subject for deep models on CPU.
- Scalers / feature selection / PCA / calibration fit on training folds only.
- Rest-baseline normalisation uses the rest file of a rest-vs-task task → report with AND without.
- Fixed seeds; every experiment is a YAML config → `results/<experiment_id>/`
  (metrics.json, per-fold CSV, confusion matrix PNG, config copy).
- Metrics: macro-F1, balanced acc, precision, recall, AUROC (OvR), confusion matrix, latency ms; mean ± std over folds.
- Cross-subject result > ~97% → stop and audit for leakage.
- Never invent numbers: reports are generated from `results/` files.

## Environment & commands (run from component1/)
- Python 3.13 venv in `.venv` (proposal says 3.11; 3.13 is what is installed — both fine). CPU only.
- Setup: `py -3.13 -m venv .venv` → `.venv\Scripts\python -m pip install -r requirements.txt` → `.venv\Scripts\python -m pip install -e .`
- Tests: `.venv\Scripts\python -m pytest`
- EDA: `.venv\Scripts\python scripts\explore_dataset.py` → `results/eda/`
- Quality report: `.venv\Scripts\python scripts\run_quality.py` → `results/quality/`
- Whole pipeline: `.venv\Scripts\python scripts\run_all.py [--quick] [--from <step>]` (README lists every step)
- API: `.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` · dashboard: `cd frontend; npm run dev` (127.0.0.1:5173; Vite proxies `/api` → 127.0.0.1:8000, so the browser never needs CORS)
- The first STEW load parses 96 text files (~30 s) and caches to `data/processed/stew_raw_cache.npz`.
- Machine limits: 4 cores / 16 GB shared with other apps → classical LOSO with 4 workers; never run deep
  training and the classical suites at the same time (D14).

## Layout
- `src/eegengine/`: `io/stew.py` loader · `labels.py` T1/T2/T3 · `quality/` · `preprocess/` (filters, bad channels,
  ICA, epoching, normalise, pipeline) · `features/` · `models/classical.py`, `models/deep/` · `training/` (cv, trainer,
  calibration, runners) · `evaluation/` · `inference/` (engine, schema, heuristics, predictors) · `data.py` (hash-located files)
- `configs/` YAML · `scripts/` one per step + `run_all.py` · `tests/` (103) · `results/` · `api/main.py` · `frontend/` (React)
- `docs/`: decisions.md (D1–D18), labels.md, output_contract.md (+schema, example), leakage_audit.md, viva_notes.md

## Conventions
- Small, readable, commented code — the author must explain everything in a viva.
- Signals are always `(channels, samples)` float64 in µV inside the package; convert to V only for MNE.
- Quality checks MARK windows (good / warn / bad); they never delete data.
- One git commit per finished step. Work phase by phase; stop after each phase with a summary.

## Status
- Phases 1–8 done (see PROGRESS.md). Results: README "Results" + results/summary_*.md.
