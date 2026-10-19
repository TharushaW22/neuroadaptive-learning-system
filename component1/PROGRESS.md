# PROGRESS — EEG engine (component1/)

Running log so work can resume after context compaction. Newest entries at the bottom of each phase.

| Phase | Status | Key numbers | Next step |
|---|---|---|---|
| 1 Setup + loader + quality | done | 96 recs, 31 tests pass, raw quality 35% good / 45% warn / 20% bad | — |
| 2 Preprocessing + epoching | done | 7104 epochs (74/rec), hash 6d7dd612; final flags 37% good / 51% warn / 12% bad; 30 ch interpolated in 26 recs; 2 recs <50% usable (kept); 53 tests | — |
| 3 Features | done | 1499 features (spectral 185, entropy 84, wavelet 266, plv 364, coh 364, graph 236), hash 4041d9e6; 55 s; 40.5 ms/epoch single-window; 18 tests | — |
| 4 Classical + ablation | in progress | | |
| 5 Deep + hybrid | todo | | |
| 6 Engine + API | code done, tested (13 tests) | engine streams on the training epoch grid; online vs offline band power < 5 % apart | export contract/demo once final models exist |
| 7 Dashboard | in progress | 6 pages written, tsc clean | npm build, demo snapshot, smoke test |
| 8 Verification + packaging | todo | | |

## Log
- Phase 2 started. docs/proposal.docx is still missing → methods follow the phase prompts (see docs/decisions.md D1).
- Phase 2 done: epochs data/processed/epochs_6d7dd612.npz (+_meta.csv). 2 s variant via configs/preprocess_2s.yaml. Decisions D5–D11.
- Phase 3 done: data/processed/features_4041d9e6.parquet + _arrays.npz (bandpower, plv, coh). Load with eegengine.data.load_features().
- Phase 4 running: `scripts/run_baselines.py` (log results/logs/classical.log) → then `scripts/run_ablation.py`.
  RF/XGB shrunk (D12); folds run in parallel processes.
- LESSON: do NOT run deep training while the classical suite runs — 8 sklearn workers + 8 torch threads
  starve torch (deep smoke made no progress in 10 min). Run them sequentially.
- Phase 5 code written (models/deep/*, training/trainer.py, deep_runner.py, scripts/run_deep.py,
  configs/experiments/deep.yaml). Next: smoke (`run_deep.py --smoke`), benchmark, then full runs.
- Memory: laptop commit charge ~31/35 GB (VS Code, Chrome, WSL, SQL Server). Classical with 4 workers;
  grid split (D14): classical.yaml (no baseline, all tasks) + classical_rest.yaml (T1 rest baseline).
- First results: T1 LOSO pooled macro-F1 LR 0.845 / SVM 0.859 / RF 0.851 / XGB 0.865; rest baseline
  inflates T1 (SVM 0.951). T3 near chance (LR 0.501, SVM 0.481).
- Remaining order: classical (running) -> run_ablation.py -> classical_kfold5 -> deep smoke -> deep full
  (background, ~hours) -> train_final_classical + run_deep --final -> export_contract / export_demo /
  export_report -> frontend build + smoke -> Phase 8 docs.
