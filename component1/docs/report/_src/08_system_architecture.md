# 8. System architecture — engine, API, dashboard

## 8.1 Repository structure (`component1/`)
```
component1/
├── CLAUDE.md, README.md, PROGRESS.md      project conventions, user guide (with generated results), progress log
├── pyproject.toml, requirements.txt       package definition (eegengine), Python dependencies
├── configs/                               YAML for preprocessing, features and every experiment
│   ├── quality.yaml  preprocess.yaml  preprocess_2s.yaml  features.yaml
│   └── experiments/ classical.yaml  classical_rest.yaml  classical_kfold5.yaml  deep.yaml
├── data/                                  (git-ignored) raw/STEW/*.txt + processed/ epochs & features (hash-named)
├── src/eegengine/                         the Python package
│   ├── io/stew.py                         STEW loader, labels helper, MNE conversion
│   ├── labels.py                          T1/T2/T3 label builder + masks
│   ├── quality/                           checks.py (per-window checks), report.py (dataset report)
│   ├── preprocess/                        dc, filters, bad_channels, ica, epoching, normalize, pipeline
│   ├── features/                          spectral, entropy, wavelet, connectivity, graph, extract
│   ├── models/                            classical.py; deep/ eegnet, cnn_lstm, gat, hybrid
│   ├── training/                          cv, sequences, classical_runner, trainer, deep_runner, calibration
│   ├── evaluation/                        metrics, ablation (Wilcoxon + Holm), reports
│   ├── inference/                         engine, schema (output contract), heuristics, predictors
│   ├── config.py, data.py                 paths/YAML/seeding; hash-based data location
├── scripts/                               one entry point per pipeline step + run_all.py + build_report.py
├── api/main.py                            FastAPI app (REST + WebSocket)
├── frontend/                              React + Vite + TypeScript + Tailwind dashboard (+ public/demo snapshot)
├── models/                                deployable models (*.joblib / *.pt) + JSON metadata
├── results/                               every experiment's outputs + summaries + figures
├── tests/                                 pytest suite
├── docs/                                  decisions, labels, output contract (+schema, example), leakage audit, viva notes, this report
└── Dockerfile.api, frontend/Dockerfile, frontend/nginx.conf, docker-compose.yml   optional containers
```
The full file-by-file index is in the appendix.

## 8.2 CognitiveEngine — the real-time flow
`src/eegengine/inference/engine.py:CognitiveEngine` (default `model="T1_xgb_all"`, `ema_alpha=0.4`,
`context_s=20`, `connectivity_band="alpha"`). `push(chunk)` appends raw samples (14 × n, µV, DC offset
allowed); every time a new 4-s window is complete (a new one every 2 s, the same grid as the training epochs)
`_process(end)` runs:

```mermaid
sequenceDiagram
    participant Src as Headset / replay
    participant E as CognitiveEngine
    participant P as Predictor (XGB .joblib or deep .pt)
    Src->>E: push(chunk 14×n µV)
    E->>E: filter last 20 s (mean removal, FIR 0.5–45 Hz, notch 50 Hz), keep newest 4 s
    E->>E: assess_window → quality flags; rolling history of hard-bad channels (30 windows)
    E->>E: channel bad in ≥ 60 % of ≥ 10 recent windows → spherical-spline interpolation, re-assess
    E->>E: extract_features (same function as training) → features, band power, PLV
    E->>P: predict(buffer of the last 1 or 8 windows)
    P-->>E: calibrated class probabilities
    E->>E: heuristic indices; EMA smoothing s_t = α·x_t + (1−α)·s_{t−1}, α = 0.4
    E-->>Src: CognitiveStateOutput (pydantic) incl. latency_ms
```
* **Predictors** (`inference/predictors.py`): `ClassicalPredictor` (joblib bundle: estimator + feature
  columns; uses the newest window) and `DeepPredictor` (checkpoint with state_dict, model/scaler config and
  temperature; uses the last ≤ 8 windows; EEGNet averages per-epoch probabilities). `list_models()` reads the
  JSON sidecars in `models/`.
* **Streaming = batch:** pushing odd-sized chunks gives exactly the same outputs as one big push
  (`tests/test_engine_api.py:test_engine_streaming_matches_batch`).
* **Online vs offline filtering:** online uses 20 s of context instead of the whole recording; band powers
  agree within 5 % (`test_engine_band_power_close_to_offline`).
* **End-to-end latency** (filter + quality + features + model + JSON) measured on the replayed demo
  recordings: median 98.4 [`results/engine_latency.json` → `median_ms`] ms per window, p95
  121.4 [`results/engine_latency.json` → `p95_ms`] ms, with XGBoost — far below the 2-s budget.

### Heuristic indices — exact formulas and why they are heuristic
`src/eegengine/inference/heuristics.py:heuristic_indices` (band power summed over the listed channels):
* **attention = engagement index** `E = β / (α + θ)` over all 14 channels (Pope, Bogart & Bartolome, 1995).
* **fatigue** `F = (θ + α) / β` over P7, P8, O1, O2 (Jap, Lal, Fischer & Bekiaris, 2009). Computed over the
  *same* channels it would be exactly 1/E, so the posterior sites — where alpha/theta increase most with
  fatigue — are used to give it information of its own.
* **readiness** `R = 0.5·E/(1+E) + 0.5·1/(1+F)` — a transparent 0–1 combination (high engagement and low
  fatigue → near 1) chosen for the dashboard, **not a validated measure**.
They are **heuristic, not trained**, because STEW has no attention/fatigue/readiness labels: there is nothing
to train or validate a model against. Every output marks them `source: "heuristic_index"` with the note
"heuristic index — not a trained prediction", and the dashboard shows a visible badge. Being ratios, their
scale differs between people; they should be compared within a person over time.

## 8.3 Output contract
`src/eegengine/inference/schema.py` (pydantic, schema version 1.0.0); documentation `docs/output_contract.md`;
JSON Schema `docs/output_contract.schema.json` (also served at `GET /schema`); real example
`docs/output_example.json` (also `GET /contract/sample`), regenerated by `scripts/export_contract.py`.

| Field | Meaning |
|---|---|
| `schema_version`, `participant_id`, `session_id`, `window_index` | identification; window 0, 1, 2, … |
| `timestamp_start/end` (ISO 8601 UTC), `t_start_s/t_end_s` | window edges |
| `signal_quality.overall` (0–1), `.flag` (good/warn/bad), `.bad_channels`, `.interpolated_channels` | trust in this window |
| `states.workload.label/probabilities/raw_probabilities/confidence/source="model"/model_version/task` | the trained prediction (smoothed and raw) |
| `states.attention/fatigue/readiness.value/raw_value/source="heuristic_index"/formula/note` | heuristic indices |
| `band_power` {channel: {band: µV²}}, `band_power_unit` | evidence for XAI |
| `connectivity` {metric "plv", band, channels, 14×14 matrix} | evidence for XAI / brain view |
| `latency_ms` | processing time of the window |

`task` tells consumers the label set (`workload_binary`: low/high; `workload_within_task`: moderate/high).
Stability promise: fields are never removed within a major version; when attention/fatigue/readiness become
trained, their `source` switches to `"model"`.

Full real example (`docs/output_example.json`):
<!-- TABLE:output_example -->

## 8.4 FastAPI backend
`api/main.py` (FastAPI 0.146, uvicorn). Run: `.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.
<!-- TABLE:api_endpoints -->

| Endpoint | Request | Response | Errors |
|---|---|---|---|
| `GET /health` | — | status, version, number of models, fs, channels | — |
| `GET /subjects` | — | 48 subjects with ratings, T2 levels, T3 label, rated flag; falls back to `results/eda/recordings.csv` if the raw data is absent | 503 if neither exists |
| `GET /models` | — | deployable models from `models/*.json` | — |
| `GET /results/summary` | — | `results/summary.json` (all tables, built by `scripts/export_report.py`) | 404 if not generated |
| `GET /results/{experiment_id}` | — | that experiment's `metrics.json` + list of files | 404 unknown / path traversal |
| `GET /results/{id}/files/{name}` | — | a PNG/CSV/YAML of the experiment | 404 |
| `GET /quality` | — | per-recording good/warn/bad counts, usable fraction, interpolated channels + reasons | 503 if processed data missing |
| `GET /quality/{subject}/{condition}` | — | per-window flag, score, bad channels | 404 unknown recording |
| `GET /schema` | — | JSON Schema of `CognitiveStateOutput` | — |
| `GET /contract/sample?model=` | — | one real output (first 10 s of sub01 task) | 404 unknown model, 503 data missing |
| `POST /predict` | multipart: `file` (STEW text samples×14, or JSON `{"data": [[14 × n]]}`), `model`, `participant_id`, `session_id` | list of `CognitiveStateOutput` | 400 unparsable / not 14 channels / < 4 s |
| `WS /stream/{subject}/{condition}?model=&speed=` | — | `meta` once; per window a `raw` message (newest 2 s of filtered signal at 64 Hz; first window 4 s) then a `state` message; finally `end`; `error` on problems | error message if recording/model missing |

**WebSocket streaming design.** The server replays a STEW recording in 2-s chunks, waits `2 s / speed` between
chunks (speed 1–50), and runs `engine.push` in a worker thread (`run_in_executor`) so the event loop stays
responsive. The display signal is decimated to 64 Hz to keep messages small.

**Errors as 503, and the "CORS" bug story.** During testing the dashboard reported *"blocked by CORS policy:
No 'Access-Control-Allow-Origin' header"* although `CORSMiddleware` allowed `localhost:5173`. Diagnosis with
`curl -i -H "Origin: http://localhost:5173"`: `/health` carried the CORS header but `/subjects` returned a bare
**500** — the raw STEW folder had been deleted, `ratings.txt` was missing, and Starlette's 500 error page
carries **no CORS headers**, so the browser misreported a server crash as a CORS block. Fixes (commit `59d5fb8`):
a `FileNotFoundError` handler returns **503 with a hint** (and keeps the CORS headers); `/subjects` falls back
to the committed EDA table; the WebSocket explains missing recordings; tests cover both
(`test_missing_data_is_503_with_cors`, `test_subjects_fallback_without_raw_data`). The dataset was restored
from the original download and the rebuilt features were byte-identical to the originals.

**CORS + Vite proxy.** The dashboard calls the relative base `/api`; the Vite dev server (and `vite preview`)
proxies `/api` — HTTP and the `/api/stream` WebSocket — to `http://127.0.0.1:8000` (`frontend/vite.config.ts`),
so the browser never makes a cross-origin request. `127.0.0.1` is used instead of `localhost` because on
Windows `localhost` may resolve to IPv6 `::1`, where uvicorn is not listening. `VITE_API_URL` bypasses the
proxy (then CORS applies, allowed for ports 5173/4173); `API_TARGET` retargets it; nginx does the same proxying
in Docker (`frontend/nginx.conf`).

## 8.5 React dashboard
Stack: React 19, Vite 6, TypeScript 5.8, Tailwind CSS 4, Recharts 2, React Router 7 (HashRouter so the built
site runs from any folder), IBM Plex fonts; colour tokens for light/dark themes in `src/index.css`.

| Page (route) | What it shows | Main components |
|---|---|---|
| Live Monitor (`/`) | subject/recording/model/speed pickers; scrolling 14-channel EEG (canvas, ±75 µV per row, bad channels red, interpolated marked *); workload card (label, confidence, probability bars, model version); attention/fatigue/readiness cards with sparklines and a visible **"heuristic index — not a trained prediction"** badge; workload probability timeline with a confidence band (±1 SD of the last 5 per-window predictions); per-window quality squares | `StreamControls`, `EEGTrace`, `StateCards`, `WorkloadTimeline`, `QualityIndicator` |
| Brain View (`/brain`) | SVG 10-20 head map coloured by band power (band selector; relative/absolute), PLV network graph with a threshold slider (edge width/opacity = PLV, node size = strength) — both update live | `HeadMap`, `NetworkGraph` |
| Signal Quality (`/quality`) | summary stats, sortable/filterable per-recording table (good/warn/bad, usable %, interpolated channels with reasons), channel × window grid of flagged windows | `SignalQuality` |
| Results (`/results`) | task selector; baseline table; feature ablation A1–A5 bars (norm toggle); architecture ablation A6–A10 bars; 5-fold comparison table; confusion matrices (selectable); reliability diagram; latency chart; Wilcoxon table — **all from `/results/summary`, nothing hard-coded** | `MetricBars`, `Reliability`, `Confusion` |
| Integration / API (`/integration`) | endpoint list, output-contract field table rendered from `/schema`, live JSON of the latest output (or `/contract/sample`) | `Integration` |
| About (`/about`) | pipeline diagram (SVG), dataset description, limitations (workload-only labels, heuristic indices, rest-vs-task confound, why T3 matters, rest baseline, consumer headset, CPU) | `About` |

**Data flow.** `lib/api.ts:getJSON` (REST via `/api`) and `lib/stream.tsx:StreamProvider` (one WebSocket for the
whole app, shared by Live Monitor and Brain View, survives page changes) → React state → Recharts / SVG /
canvas. `lib/useApi.ts` refetches when demo mode toggles.
**Demo mode (offline).** `scripts/export_demo.py` calls the same functions the API serves and writes
`frontend/public/demo/*.json` (subjects, models, results summary, quality tables, schema, sample, and the exact
WebSocket message sequences of 8 recordings: subjects 1, 2, 16, 38 × rest/task). `lib/settings.tsx` switches to
demo mode automatically when the API is unreachable; `stream.tsx` replays the messages with the same timing.
Header toggles: Demo/Live, Projector (larger type), Light/Dark (remembered in `localStorage`, guarded).

## 8.6 How to run
```powershell
cd component1
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
# put STEW files in data\raw\STEW\  (sub01_lo.txt … sub48_hi.txt, ratings.txt)
.venv\Scripts\python scripts\run_all.py            # whole pipeline (hours); --quick skips long runs; --from <step>
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
cd frontend; npm install; npm run dev              # http://127.0.0.1:5173
.venv\Scripts\python -m pytest                     # tests
.venv\Scripts\python scripts\build_report.py       # this report
```
`scripts/run_all.py` steps: eda → quality → preprocess (+ICA demo) → preprocess_2s → features → baselines →
baselines_rest → ablation → classical_kfold5 → deep_smoke → deep → deep_stats → final_classical → final_deep →
report → contract → audit → demo → tests (each logged to `results/logs/`, stops at the first failure).

## 8.7 Reproducibility
* Fixed seeds (42; deep folds 42 + fold); fixed, saved 5-fold assignment (`results/folds_5fold.json`).
* Every experiment is a YAML config; each results folder holds a copy (`config.yaml`), `metrics.json`,
  `folds.csv`, `predictions.csv`, `confusion_matrix.png` (deep: + `history.csv`, `reliability.png`).
* Generated data are named by configuration hash (`epochs_6d7dd612`, `features_4041d9e6`).
* **Byte-identical rebuild:** `run_all.py --quick` from a deleted `data/processed/` produced a features file
  with the same MD5 (`65dc139f8d39…`) as the original; the same check was repeated after restoring the raw
  dataset from the original download.
* Summary tables are generated, never typed (`scripts/export_report.py`, `scripts/build_report.py`).

## 8.8 Docker (optional)
`docker-compose.yml` defines `api` (from `Dockerfile.api`, Python 3.11-slim, CPU PyTorch) and `frontend`
(Node build → nginx with `/api` proxy), mounting `data/`, `results/`, `models/` read-only.
**The Docker images were written but not built or tested** on the development machine (NOT AVAILABLE).

## 8.9 Tests
<!-- TABLE:tests -->

Coverage in brief: loader shapes/labels/missing ratings/orientation/MNE conversion; each quality check on
synthetic defects; filter responses, zero phase, notch, epoch counts, blink tolerance, bad-channel repair,
normalisation (rest baseline uses only rest), label rules and real class counts; PLV limits, coherence vs SciPy,
band-power sums, entropy/Hjorth/wavelet properties, graph metrics on known graphs, feature structure;
**CV leakage guard** (LOSO, fixed folds, overlap assertion, scaler inside pipeline), sequences within recordings,
every hybrid branch combination, masked loss, GAT adjacency, temperature scaling, metrics, Wilcoxon/Holm;
engine full recording, streaming = batch, online ≈ offline band power, heuristic formulas, every API endpoint,
WebSocket, upload, 503/CORS and fallback. Tests that need the raw data skip cleanly when it is absent.

## 8.10 Tech stack (versions)
<!-- TABLE:versions -->
