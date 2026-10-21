# Vision-Based Cognitive Behaviour Understanding for Intelligent Learning Systems

Computer-vision component of an undergraduate research project. A single ordinary RGB webcam
is used to extract visual features (eye aspect ratio, blinks, gaze proxies, head pose, facial
geometry, quality flags), which are analysed over time windows and related to learning-related
labels from the **DAiSEE** dataset (engagement, boredom, confusion, frustration; levels 0-3).

> Scope: computer vision only. There is **no EEG functionality** in this repository.

The project runs in two modes:

| Mode | When | What works |
|---|---|---|
| **DEMO MODE** | No dataset in `data/raw/daisee/` (default today) | Dashboard, live/mock webcam feature extraction, synthetic data, DEMO ONLY models, all tests. **No research predictions or metrics.** |
| **RESEARCH MODE** | DAiSEE videos + official label CSVs present | Audit, feature extraction, real training, held-out evaluation, validated live predictions. |

Safeguards built into the code (not just the docs):
* Real training/evaluation **refuse to run** without real labelled features (exit code 2).
* Synthetic rows are flagged `is_synthetic=True`, stored only in `data/demo/`, and are rejected by real training.
* Synthetic models are named `DEMO_ONLY_*`, stored in `data/demo/models/`, and refused for live inference unless explicitly selected.
* Live predictions require a **validated model**: trained on real data *and* named in a real evaluation report. Otherwise the UI shows **NO VALIDATED MODEL AVAILABLE**.
* Low-quality input yields **UNKNOWN**, not a guess.
* Without a dataset the evaluation page shows **NOT EVALUATED: REAL DATASET NOT AVAILABLE**.
* Raw video is never written to disk (no code path exists for it; a test checks this).

---

## 1. Installation (Windows PowerShell, Python 3.11)

```powershell
# from the project root (the folder containing app.py)
py -3.11 -m venv .venv                         # 1. create virtual environment
.\.venv\Scripts\Activate.ps1                   # 2. activate it
python -m pip install --upgrade pip
pip install -r requirements.txt                # 3. install requirements
python scripts\check_environment.py            # 4. check Python/library compatibility
python scripts\check_environment.py --camera   #    ...and try the webcam
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once), then re-run step 2.

### Dependency conflicts
* **MediaPipe**: `mediapipe==0.10.14` provides the legacy FaceMesh (`mediapipe.solutions`) API and
  needs NumPy 1.x and protobuf < 5, hence the caps in `requirements.txt`. If you install a newer
  MediaPipe (which removed `solutions`), the project automatically uses the **Tasks API** instead;
  run `python scripts\download_face_model.py` once to fetch `models\face_landmarker.task`.
* **OpenCV**: MediaPipe pulls in `opencv-contrib-python`. Do **not** install `opencv-python`
  alongside it (both provide `cv2` and overwrite each other). Fix: `pip uninstall -y opencv-python opencv-contrib-python` then `pip install opencv-contrib-python`.
* **Fallback**: if no MediaPipe backend is usable, OpenCV Haar face boxes are used; landmark
  features are then missing (NaN) and quality flags say so. If even that is unavailable the
  backend is `none` - the app still runs and tests still pass (they use a mock detector).
* **GPU**: optional. Install a CUDA build of torch from pytorch.org first; `device: auto` uses it.

## 2. Workflows (PowerShell commands)

| # | Task | Command |
|---|---|---|
| 5 | Run all automated tests (no dataset, no webcam) | `python -m pytest` |
| 6 | Generate synthetic demo data (+ DEMO ONLY models/report) | `python scripts\generate_demo_data.py --train-demo-models --evaluate` |
| 7 | Run the dashboard (demo mode) | `streamlit run app.py` |
| 8 | Live webcam feature extractor (Q to quit) | `python scripts\run_webcam.py` |
|   | ...without hardware | `python scripts\run_webcam.py --mock --seconds 10 --no-display` |
| 9 | Audit the dataset (after adding videos) | `python scripts\audit_dataset.py` |
| 10 | Extract and cache real features | `python scripts\extract_features.py` (`--limit 20` to try first) |
| 11 | Train baseline on real labelled data | `python scripts\train_baseline.py` |
| 12 | Train GRU/LSTM on real labelled data | `python scripts\train_temporal.py` (`--cell lstm`) |
| 13 | Evaluate both models, write reports | `python scripts\evaluate.py` |

Exit codes: `0` success, `2` refused because real data/models are missing, `1` other error.
Before the dataset exists, steps 10-13 print a clear refusal and exit with `2` - that is expected.

What changed between versions: `CHANGELOG.md`. A combined check is in `scripts\run_checks.ps1` (`powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1`).

## 3. Adding DAiSEE later
See `data/raw/daisee/README.md` and `docs/dataset_guide.md`. In short: copy the official video
folder into `data/raw/daisee/videos/` and the label CSVs into `data/raw/daisee/annotations/`
without renaming anything, then run steps 9-13. The dashboard switches to RESEARCH MODE automatically.

## 4. Configuration
`configs/default.yaml` holds every parameter (camera, thresholds, sampling rate, window/stride,
normalisation, label scheme, model hyper-parameters, privacy). Create
`configs/my_experiment.yaml` with only the keys you change and pass `--config configs/my_experiment.yaml`
to any script. CLI flags (e.g. `--epochs`, `--window`) override both.

Label schemes (`labels.scheme`): `daisee_original` (default, 4 levels), `daisee_binary`, and
`operational` (FOCUSED / REDUCED_ATTENTION / DISTRACTED) which is **refused** unless a mapping and an
existing justification document are configured - see `docs/labeling_protocol.md`.

**Experimental heuristic scores** (`interpretation` section): attention, fatigue and learning
readiness as floats in [0, 1], computed by fixed formulas from the predicted engagement level and
visual proxies. They are **not trained or validated**; the model predicts DAiSEE engagement 0-3 only.
Scores are null (UNKNOWN) when face quality is poor. Formulas and weights: `docs/labeling_protocol.md`
section 6.

## 5. Project structure

```
app.py                     Streamlit dashboard (8 pages)
configs/default.yaml       all settings
src/webcam/camera.py       webcam + mock camera, guaranteed release
src/vision/                landmarks (MediaPipe/Tasks/Haar/mock), EAR+blinks, gaze proxy, head pose,
                           facial geometry, per-frame feature extractor
src/data/                  schema, annotation parser, video loader, audit, validation,
                           feature extraction/caching, sequence builder, synthetic generator
src/processing/            imputation + normalisation (train-only), quality control
src/models/                label schemes, baseline, GRU/LSTM, training, registry, inference
src/evaluation/            metrics, plots, report writer
src/utils/                 config, logging, privacy, status
scripts/                   CLI entry points (see table above)
tests/                     pytest suite (no dataset / webcam needed)
docs/                      methodology, dataset guide, labelling & evaluation protocols,
                           limitations & ethics, architecture diagrams, demo script, checklist
data/ models/ outputs/     data and artefacts (git-ignored except placeholders)
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open camera index 0` | Close Teams/Zoom/Camera app; Windows Settings > Privacy & security > Camera > allow desktop apps; try `camera.index: 1`; try `camera.backend: msmf`. |
| Face backend `opencv_haar` or `none` | Install `mediapipe==0.10.14` on Python 3.11, or run `scripts\download_face_model.py` with newer MediaPipe. |
| `numpy.core.multiarray failed to import` / protobuf errors | NumPy 2 or protobuf 5 got installed: `pip install "numpy<2" "protobuf<5"`. |
| `ModuleNotFoundError: src` | Run commands from the project root. |
| Everything is UNKNOWN in live view | Improve lighting, face the camera, one person only; check the quality warnings. |
| Training says REFUSED | Expected without real data. Run audit -> extraction first. |
| Streamlit page blank after long capture | Use shorter capture durations; the capture loop runs inside the page script. |
| `Capture stopped: '{' was never closed` | You are running a pre-1.1.0 `app.py`. Check the version in the sidebar; update the files (see CHANGELOG) and restart Streamlit. |
| Browser console shows `500` for `.jpg` media files | Pre-1.1.0 behaviour; fixed in 1.1.0. A single message right after reloading the page is harmless. |
| Capture ends with "N frame(s) had processing errors" | Open "Error details" under the message and report the text; the capture itself continues. |
| Which version am I running? | Dashboard sidebar, or `python scripts\check_environment.py` (first line). |

## 7. Documentation
* `docs/methodology.md` - research question, features, models, split protocol, metrics
* `docs/dataset_guide.md` - adding DAiSEE, audit outputs, local modification log
* `docs/labeling_protocol.md` - engagement vs. distraction; when operational labels are allowed;
  heuristic attention / fatigue / readiness formulas (section 6)
* `docs/evaluation_protocol.md` - evaluation procedure + **research evaluation template**
* `docs/limitations_and_ethics.md` - limitations, privacy, deletion, licence
* `docs/architecture.md` - Mermaid architecture and data-flow diagrams
* `docs/supervisor_demo_script.md` - step-by-step presentation script
* `docs/requirements_checklist.md` - final requirements checklist with verification status
