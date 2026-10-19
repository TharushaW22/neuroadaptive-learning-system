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
<!-- BEGIN TABLE:output_example -->
```json
{
  "schema_version": "1.0.0",
  "participant_id": "sub01",
  "session_id": "example-sub01-hi",
  "window_index": 8,
  "timestamp_start": "2026-10-19T14:53:44.162120+00:00",
  "timestamp_end": "2026-10-19T14:53:48.162120+00:00",
  "t_start_s": 16.0,
  "t_end_s": 20.0,
  "signal_quality": {
    "overall": 0.9285714285714286,
    "flag": "warn",
    "bad_channels": [
      "AF3"
    ],
    "interpolated_channels": []
  },
  "states": {
    "workload": {
      "label": "high",
      "probabilities": {
        "low": 0.058183277882667174,
        "high": 0.9418167221173328
      },
      "raw_probabilities": {
        "low": 0.06814136609961711,
        "high": 0.9318586339003829
      },
      "confidence": 0.9418167221173328,
      "source": "model",
      "model_version": "A10_hybrid-T1-20261019",
      "task": "workload_binary"
    },
    "attention": {
      "value": 0.3720720888940098,
      "raw_value": 0.5138092523912298,
      "source": "heuristic_index",
      "formula": "engagement = beta / (alpha + theta), all channels (Pope et al., 1995)",
      "note": "heuristic index \u2014 not a trained prediction"
    },
    "fatigue": {
      "value": 2.0113450543829896,
      "raw_value": 2.4474272969254836,
      "source": "heuristic_index",
      "formula": "(theta + alpha) / beta, channels P7 P8 O1 O2 (Jap et al., 2009)",
      "note": "heuristic index \u2014 not a trained prediction"
    },
    "readiness": {
      "value": 0.3009565440055719,
      "raw_value": 0.31474308690206165,
      "source": "heuristic_index",
      "formula": "0.5*E/(1+E) + 0.5*1/(1+F), E = engagement, F = fatigue index",
      "note": "heuristic index \u2014 not a trained prediction"
    }
  },
  "band_power": {
    "AF3": {
      "delta": 110.75843048095703,
      "theta": 20.783336639404297,
      "alpha": 8.348448753356934,
      "beta": 19.349763870239258,
      "gamma": 6.644127368927002
    },
    "F7": {
      "delta": 274.59295654296875,
      "theta": 49.26381301879883,
      "alpha": 13.707437515258789,
      "beta": 21.706527709960938,
      "gamma": 11.24522876739502
    },
    "F3": {
      "delta": 25.103782653808594,
      "theta": 5.944693088531494,
      "alpha": 7.127309322357178,
      "beta": 9.699406623840332,
      "gamma": 5.452876091003418
    },
    "FC5": {
      "delta": 71.22528839111328,
      "theta": 13.248261451721191,
      "alpha": 4.13751745223999,
      "beta": 12.077031135559082,
      "gamma": 6.392486572265625
    },
    "T7": {
      "delta": 51.396480560302734,
      "theta": 6.120129585266113,
      "alpha": 3.723794937133789,
      "beta": 6.83806037902832,
      "gamma": 5.176726341247559
    },
    "P7": {
      "delta": 58.926937103271484,
      "theta": 8.264906883239746,
      "alpha": 3.197747230529785,
      "beta": 8.67247200012207,
      "gamma": 5.52169132232666
    },
    "O1": {
      "delta": 140.0385284423828,
      "theta": 21.865211486816406,
      "alpha": 8.788727760314941,
      "beta": 8.367042541503906,
      "gamma": 4.035991668701172
    },
    "O2": {
      "delta": 56.313079833984375,
      "theta": 13.507505416870117,
      "alpha": 11.93719482421875,
      "beta": 10.40793514251709,
      "gamma": 5.905566215515137
    },
    "P8": {
      "delta": 93.63929748535156,
      "theta": 16.901254653930664,
      "alpha": 8.860452651977539,
      "beta": 10.683611869812012,
      "gamma": 6.831902503967285
    },
    "T8": {
      "delta": 97.138916015625,
      "theta": 21.19915771484375,
      "alpha": 10.347604751586914,
      "beta": 11.547039985656738,
      "gamma": 7.321165561676025
    },
    "FC6": {
      "delta": 382.5666198730469,
      "theta": 23.34512710571289,
      "alpha": 10.680864334106445,
      "beta": 16.667661666870117,
      "gamma": 6.2488884925842285
    },
    "F4": {
      "delta": 858.5726318359375,
      "theta": 20.60382843017578,
      "alpha": 10.7993745803833,
      "beta": 18.579910278320312,
      "gamma": 9.950177192687988
    },
    "F8": {
      "delta": 416.6779479980469,
      "theta": 40.217376708984375,
      "alpha": 21.798734664916992,
      "beta": 19.0502872467041,
      "gamma": 7.097566604614258
    },
    "AF4": {
      "delta": 93.14506530761719,
      "theta": 19.432640075683594,
      "alpha": 18.049301147460938,
      "beta": 43.28441619873047,
      "gamma": 30.396207809448242
    }
  },
  "band_power_unit": "uV^2",
  "connectivity": {
    "metric": "plv",
    "band": "alpha",
    "channels": [
      "AF3",
      "F7",
      "F3",
      "FC5",
      "T7",
      "P7",
      "O1",
      "O2",
      "P8",
      "T8",
      "FC6",
      "F4",
      "F8",
      "AF4"
    ],
    "matrix": [
      [
        1.0,
        0.6753000020980835,
        0.46959999203681946,
        0.5468999743461609,
        0.3140999972820282,
        0.19009999930858612,
        0.039000000804662704,
        0.14790000021457672,
        0.08349999785423279,
        0.11469999700784683,
        0.22030000388622284,
        0.4032999873161316,
        0.3066999912261963,
        0.49230000376701355
      ],
      [
        0.6753000020980835,
        1.0,
        0.5202999711036682,
        0.7307000160217285,
        0.3546999990940094,
        0.2240999937057495,
        0.08250000327825546,
        0.026900000870227814,
        0.1282999962568283,
        0.2125999927520752,
        0.1615000069141388,
        0.38989999890327454,
        0.10779999941587448,
        0.27489998936653137
      ],
      [
        0.46959999203681946,
        0.5202999711036682,
        1.0,
        0.6031000018119812,
        0.3578000068664551,
        0.2517000138759613,
        0.11020000278949738,
        0.2554999887943268,
        0.3328999876976013,
        0.25690001249313354,
        0.49079999327659607,
        0.6873000264167786,
        0.5078999996185303,
        0.6420000195503235
      ],
      [
        0.5468999743461609,
        0.7307000160217285,
        0.6031000018119812,
        1.0,
        0.4537999927997589,
        0.2676999866962433,
        0.0357000008225441,
        0.1632000058889389,
        0.19550000131130219,
        0.20360000431537628,
        0.2797999978065491,
        0.4300000071525574,
        0.26460000872612,
        0.29330000281333923
      ],
      [
        0.3140999972820282,
        0.3546999990940094,
        0.3578000068664551,
        0.4537999927997589,
        1.0,
        0.48739999532699585,
        0.09000000357627869,
        0.14000000059604645,
        0.271699994802475,
        0.19550000131130219,
        0.33980000019073486,
        0.24959999322891235,
        0.3644999861717224,
        0.35670000314712524
      ],
      [
        0.19009999930858612,
        0.2240999937057495,
        0.2517000138759613,
        0.2676999866962433,
        0.48739999532699585,
        1.0,
        0.2248000055551529,
        0.4438000023365021,
        0.2605000138282776,
        0.4088999927043915,
        0.3059999942779541,
        0.164900004863739,
        0.3336000144481659,
        0.14480000734329224
      ],
      [
        0.039000000804662704,
        0.08250000327825546,
        0.11020000278949738,
        0.0357000008225441,
        0.09000000357627869,
        0.2248000055551529,
        1.0,
        0.3978999853134155,
        0.4027999937534332,
        0.31709998846054077,
        0.38999998569488525,
        0.21140000224113464,
        0.30329999327659607,
        0.21240000426769257
      ],
      [
        0.14790000021457672,
        0.026900000870227814,
        0.2554999887943268,
        0.1632000058889389,
        0.14000000059604645,
        0.4438000023365021,
        0.3978999853134155,
        1.0,
        0.8057000041007996,
        0.8050000071525574,
        0.7303000092506409,
        0.3968000113964081,
        0.6855000257492065,
        0.4009999930858612
      ],
      [
        0.08349999785423279,
        0.1282999962568283,
        0.3328999876976013,
        0.19550000131130219,
        0.271699994802475,
        0.2605000138282776,
        0.4027999937534332,
        0.8057000041007996,
        1.0,
        0.8355000019073486,
        0.8614000082015991,
        0.5059000253677368,
        0.7806000113487244,
        0.4941999912261963
      ],
      [
        0.11469999700784683,
        0.2125999927520752,
        0.25690001249313354,
        0.20360000431537628,
        0.19550000131130219,
        0.4088999927043915,
        0.31709998846054077,
        0.8050000071525574,
        0.8355000019073486,
        1.0,
        0.7213000059127808,
        0.3215999901294708,
        0.7598999738693237,
        0.44749999046325684
      ],
      [
        0.22030000388622284,
        0.1615000069141388,
        0.49079999327659607,
        0.2797999978065491,
        0.33980000019073486,
        0.3059999942779541,
        0.38999998569488525,
        0.7303000092506409,
        0.8614000082015991,
        0.7213000059127808,
        1.0,
        0.614799976348877,
        0.8553000092506409,
        0.657800018787384
      ],
      [
        0.4032999873161316,
        0.38989999890327454,
        0.6873000264167786,
        0.4300000071525574,
        0.24959999322891235,
        0.164900004863739,
        0.21140000224113464,
        0.3968000113964081,
        0.5059000253677368,
        0.3215999901294708,
        0.614799976348877,
        1.0,
        0.5454999804496765,
        0.5740000009536743
      ],
      [
        0.3066999912261963,
        0.10779999941587448,
        0.5078999996185303,
        0.26460000872612,
        0.3644999861717224,
        0.3336000144481659,
        0.30329999327659607,
        0.6855000257492065,
        0.7806000113487244,
        0.7598999738693237,
        0.8553000092506409,
        0.5454999804496765,
        1.0,
        0.733299970626831
      ],
      [
        0.49230000376701355,
        0.27489998936653137,
        0.6420000195503235,
        0.29330000281333923,
        0.35670000314712524,
        0.14480000734329224,
        0.21240000426769257,
        0.4009999930858612,
        0.4941999912261963,
        0.44749999046325684,
        0.657800018787384,
        0.5740000009536743,
        0.733299970626831,
        1.0
      ]
    ]
  },
  "latency_ms": 45.18480000842828
}
```
<!-- END TABLE:output_example -->

## 8.4 FastAPI backend
`api/main.py` (FastAPI 0.146, uvicorn). Run: `.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.
<!-- BEGIN TABLE:api_endpoints -->
| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/subjects` |
| GET | `/models` |
| GET | `/results/summary` |
| GET | `/results/{experiment_id}` |
| GET | `/results/{experiment_id}/files/{name}` |
| GET | `/quality` |
| GET | `/quality/{subject}/{condition}` |
| GET | `/schema` |
| GET | `/contract/sample` |
| POST | `/predict` |
| WS | `/stream/{subject}/{condition}` |

Source: decorators in `api/main.py`
<!-- END TABLE:api_endpoints -->

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
<!-- BEGIN TABLE:tests -->
| Test file | What it covers | Test functions |
|---|---|---|
| `tests/test_engine_api.py` | Phase 6 tests: inference engine, output contract, heuristics, and the FastAPI app. | 15 |
| `tests/test_features.py` | Phase 3 tests: connectivity, band powers, graph metrics, feature table structure. | 18 |
| `tests/test_models_cv.py` | Tests for the deep models, the masked multi-task loss, calibration, metrics and the CV leakage guard. | 14 |
| `tests/test_preprocess.py` | Phase 2 tests: filters, epoching, bad-channel repair, normalisation, labels. | 17 |
| `tests/test_quality.py` | Tests for the signal quality checks, using synthetic signals with known defects. | 12 |
| `tests/test_stew_loader.py` | Tests for the STEW loader and label builder. | 13 |

Test functions: 89; collected test cases (after `@pytest.mark.parametrize` expansion): 105.
<!-- END TABLE:tests -->

Coverage in brief: loader shapes/labels/missing ratings/orientation/MNE conversion; each quality check on
synthetic defects; filter responses, zero phase, notch, epoch counts, blink tolerance, bad-channel repair,
normalisation (rest baseline uses only rest), label rules and real class counts; PLV limits, coherence vs SciPy,
band-power sums, entropy/Hjorth/wavelet properties, graph metrics on known graphs, feature structure;
**CV leakage guard** (LOSO, fixed folds, overlap assertion, scaler inside pipeline), sequences within recordings,
every hybrid branch combination, masked loss, GAT adjacency, temperature scaling, metrics, Wilcoxon/Holm;
engine full recording, streaming = batch, online ≈ offline band power, heuristic formulas, every API endpoint,
WebSocket, upload, 503/CORS and fallback. Tests that need the raw data skip cleanly when it is absent.

## 8.10 Tech stack (versions)
<!-- BEGIN TABLE:versions -->
| Package | Required (requirements.txt / package.json) | Installed in the dev environment |
|---|---|---|
| Python | ≥ 3.11 (pyproject) | 3.13.7 |
| numpy | numpy>=1.26 | 2.5.4 |
| scipy | scipy>=1.11 | 1.18.1 |
| pandas | pandas>=2.1 | 3.0.6 |
| pyarrow | pyarrow>=14 | 26.0.0 |
| matplotlib | matplotlib>=3.8 | 3.11.2 |
| PyYAML | PyYAML>=6.0 | 6.0.3 |
| mne | mne>=1.6 | 1.13.2 |
| PyWavelets | PyWavelets>=1.5 | 1.10.0 |
| networkx | networkx>=3.2 | 3.7 |
| scikit-learn | scikit-learn>=1.4 | 1.9.1 |
| xgboost | xgboost>=2.0 | 3.4.1 |
| torch | torch>=2.2 | 2.14.1 |
| joblib | (dependency) | 1.6.0 |
| fastapi | fastapi>=0.110 | 0.146.0 |
| uvicorn | uvicorn[standard]>=0.27 | 0.54.0 |
| pydantic | pydantic>=2.5 | 2.14.1 |
| python-multipart | python-multipart>=0.0.9 | 0.0.32 |
| httpx | httpx>=0.27 | 0.28.1 |
| websockets | (dependency) | 17.2 |
| pytest | pytest>=8.0 | 9.1.1 |
| react | ^19.1.0 | 19.3.1 |
| react-dom | ^19.1.0 | 19.3.1 |
| react-router-dom | ^7.6.2 | 7.18.4 |
| recharts | ^2.15.3 | 2.15.3 |
| @tailwindcss/vite | ^4.1.10 | 4.1.10 |
| @vitejs/plugin-react | ^4.4.1 | 4.7.0 |
| tailwindcss | ^4.1.10 | 4.1.10 |
| typescript | ~5.8.3 | 5.8.3 |
| vite | ^6.3.5 | 6.4.4 |
<!-- END TABLE:versions -->
