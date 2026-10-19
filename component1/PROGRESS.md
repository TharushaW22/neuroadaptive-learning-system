# PROGRESS — EEG engine (component1/)

Running log so work can resume after context compaction. Newest entries at the bottom of each phase.

| Phase | Status | Key numbers | Next step |
|---|---|---|---|
| 1 Setup + loader + quality | done | 96 recs, 31 tests pass, raw quality 35% good / 45% warn / 20% bad | — |
| 2 Preprocessing + epoching | done | 7104 epochs (74/rec), hash 6d7dd612; final flags 37% good / 51% warn / 12% bad; 30 ch interpolated in 26 recs; 2 recs <50% usable (kept); 53 tests | — |
| 3 Features | done | 1499 features (spectral 185, entropy 84, wavelet 266, plv 364, coh 364, graph 236), hash 4041d9e6; 55 s; 40.5 ms/epoch single-window; 18 tests | — |
| 4 Classical + ablation | in progress | | |
| 5 Deep + hybrid | todo | | |
| 6 Engine + API | todo | | |
| 7 Dashboard | todo | | |
| 8 Verification + packaging | todo | | |

## Log
- Phase 2 started. docs/proposal.docx is still missing → methods follow the phase prompts (see docs/decisions.md D1).
- Phase 2 done: epochs data/processed/epochs_6d7dd612.npz (+_meta.csv). 2 s variant via configs/preprocess_2s.yaml. Decisions D5–D11.
- Phase 3 done: data/processed/features_4041d9e6.parquet + _arrays.npz (bandpower, plv, coh). Load with eegengine.data.load_features().
