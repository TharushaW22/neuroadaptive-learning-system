# Component 2 analysis (Phase 1, read-only) — 2026-10-21

Analysed: `component2/vision_cbu/` at commit `6c7607e` ("computer vision full component"; the only
commit that touches component2/). Nothing was run, installed or written in component2/. Line numbers
refer to that commit.

> **Update 2026-10-21 (decisions D-V1–D-V5, PLAN.md §3b):** the student chose to build the vision
> context ahead (binary engagement GRU trained by the C2 owner) and to drop multimodal. Built; see
> PROGRESS.md "Step 15 — vision context" and VISION_RUNBOOK.md. The analysis below is unchanged.

**Outcome (at the time of analysis): Phase 2 is blocked.** Three STOP conditions apply (§10). Component 3 has nothing to run
yet: no DAiSEE data, no trained model, and no fusion model.

---

## a) Folder structure and entry points

```
component2/vision_cbu/            project root (imports use the top-level package name `src`)
  app.py                          Streamlit dashboard (8 pages; opens the webcam on the live page)
  configs/default.yaml            all settings (paths, window, labels, model hyper-parameters)
  src/vision/                     MediaPipe Face Mesh landmarks → EAR/blinks, gaze proxy, head pose, facial ratios
  src/data/                       schema, DAiSEE annotation parser, video loader, audit, feature extraction, windows, synthetic (DEMO)
  src/processing/                 bounded forward-fill, Normalizer (fit on train only), quality control
  src/models/                     labels, baseline (sklearn), temporal (PyTorch GRU/LSTM), train, registry, inference, interpretation
  src/evaluation/                 metrics, plots, report writer
  scripts/                        audit_dataset · extract_features · train_baseline · train_temporal · evaluate ·
                                  generate_demo_data · run_webcam · download_face_model · check_environment
  data/raw/daisee/{videos,annotations}/   EMPTY (only .gitkeep)
  data/processed/{features,sequences}/    EMPTY
  models/  outputs/                       EMPTY
  tests/                          pytest suite (fixtures / mock detector; no dataset)
```

| Purpose | Entry point | Library function |
|---|---|---|
| Dataset audit | `scripts/audit_dataset.py` | `src/data/dataset_audit.py` |
| Feature extraction (video → per-frame CSV) | `scripts/extract_features.py` | `src/data/extract_features.py:38 run_extraction` |
| Training | `scripts/train_baseline.py`, `scripts/train_temporal.py` | `src/models/train.py:294 train_real` → `prepare_data` (:96), `train_baseline` (:168), `train_temporal` (:196) |
| Inference | — (dashboard / webcam) | `src/models/inference.py:28 predict_proba`, `Predictor` |
| Evaluation | `scripts/evaluate.py` | `src/evaluation/reports.py` |

## b) Dataset

* **DAiSEE** (public; Indian Institute of Technology Hyderabad; licence: research use only, no
  redistribution, verify on the official page). It contains short webcam clips of students watching
  online lectures, with crowd-annotated levels 0–3 for engagement, boredom, confusion and frustration,
  plus official train/validation/test split files.
  Published figures (≈ 112 participants, ≈ 9,068 clips of 10 s) come from the DAiSEE paper and have
  **not** been checked here: C2's own docs say to verify counts against the paper.
* **Present on this machine: NO.** `component2/vision_cbu/data/raw/daisee/videos/` and
  `annotations/` hold only `.gitkeep`. A search of `~/Desktop`, `~/Downloads`, `~/Documents` and
  `/Volumes` found no DAiSEE files or label CSVs. There are no extracted features either
  (`data/processed/features/` is empty). C2's README states it runs in "DEMO MODE (default today)".
* **Subject IDs:** yes, from the folder layout `DataSet/<Split>/<subjectID>/<clipID>/<clipID>.avi`
  (`video_loader.participant_from_path`, used at `dataset_audit.py:77`).
* **Split:** the official DAiSEE split files are kept (`configs/default.yaml` `split.use_official_splits: true`).
  Validation is carved from train **by participant** only if no validation split exists
  (`train.py:45-61`, GroupShuffleSplit on `participant_id`). Video-level leakage raises an error.
  Participant overlap between official splits is only a **warning** (`train.py:117-124`), so the
  split is subject-wise only if the official DAiSEE split is subject-disjoint. The audit reports this.
* **Labels:** clip-level; every window inherits its clip's label (`broadcast_video_label`,
  `sequence_builder.py:1-8`). These are weak window labels.
* **Synthetic data:** `src/data/synthetic.py` and `scripts/generate_demo_data.py` produce DEMO ONLY
  data. Component 3 will **not** use it (rule 9).

## c) Models

| | Baseline | Temporal (main) |
|---|---|---|
| Framework | scikit-learn `Pipeline` (median imputer → StandardScaler → LogisticRegression, or RandomForest) | **PyTorch** `TemporalClassifier` (`src/models/temporal_model.py`) |
| Architecture | window statistics mean/std/min/max × 15 features = 60 inputs (`baseline.py:16-27`) | GRU (default) or LSTM, hidden 64, 1 layer; representation = [masked mean of RNN outputs ; output at the last valid step] → dropout → linear (`temporal_model.py:28-42`) |
| Saved as | `models/<name>/model.joblib` + `metadata.json` | `models/<name>/model.pt` (state_dict) + `metadata.json` |
| Exists? | **No** (`models/` empty; no `data/demo/models/` either) | **No** |
| Loading | `registry.load_model(dir)` (`registry.py:88`): checks metadata, refuses demo models, schema-version and feature-list mismatches | same; rebuilds `TemporalClassifier(**meta["model_params"])`, `torch.load(weights_only=True)` |
| Inference | `inference.predict_proba(model, meta, X, mask)` (`inference.py:28`): softmax(logits) for temporal; `baseline_predict_proba` for the baseline | |

Can weights be produced in < 30 min? **No.** Training needs extracted features. Extraction needs
the DAiSEE videos plus MediaPipe, and MediaPipe 0.10.14 is not installable on component3's Python 3.13
(see h). Rough extraction cost (not measured; assumes ~20–40 ms per MediaPipe frame):
9,068 clips × 10 s × 5 fps ≈ 450k frames ≈ 2.5–5 h of CPU time. Training itself is cheap (GRU, 15 inputs).

## d) Input representation

* Per sampled frame (5 fps for videos), **15 tabular features**, schema v1.0.0 (`feature_schema.py:46-53`), in this order:
  * **eye**: `ear_left, ear_right, ear_mean, blink_event, blink_rate_per_min`
  * **gaze**: `gaze_h, gaze_v` (uncalibrated iris-in-eye proxy)
  * **head_pose**: `head_yaw, head_pitch, head_roll` (solvePnP to the canonical face)
  * **facial**: `mouth_aspect_ratio, brow_raise, face_scale, face_center_x, face_center_y`
    (geometric ratios, **not** FACS action units)
  * plus 9 quality flags (not model inputs except `quality_ok` gating).
* **Window:** `(T=20, F=15)` = 4 s at 5 fps, stride 10 = 2 s (`configs/default.yaml` `sequences`), plus a
  mask `(T,)` for zero-padded short clips (`sequence_builder.py:66-90`). Windows with < 60 % quality-OK
  frames are dropped.
* **Pre-processing:** frames failing quality are NaN; forward-fill ≤ 3 samples within a video
  (`preprocessing.py:16`); remaining NaN → training medians; standardised with training mean/std
  (`Normalizer`, `preprocessing.py:41-66`). Normaliser parameters are **stored in the checkpoint's
  `metadata.json`** (`train.py:149`, `Normalizer.to_dict`).
* No raw frames, face crops, CNN or embeddings: the model never sees pixels. The face mesh is used only
  to compute the 15 features.

## e) Outputs

Softmax probabilities over the label scheme's classes. The default is `daisee_original`, 4 classes
`engagement_0 … engagement_3`. `daisee_binary` gives low/high with threshold 2 (`labels.py:56-62`). There is
no calibration step (no temperature scaling). Evaluation writes `outputs/evaluation/<timestamp>/`
(`report.json`, `window_predictions.csv`, …). The heuristic attention/fatigue/readiness scores
(`interpretation.py`) are fixed formulas, explicitly **not** model outputs.

## f) Multimodal fusion

**It does not exist.** No file in component2/ mentions EEG, fusion or multimodal, except
statements that rule them out:

```
README.md:8        > Scope: computer vision only. There is **no EEG functionality** in this repository.
app.py:58          ... vision only (no EEG)
docs/requirements_checklist.md:52   | No EEG functionality | ✅ |
```

There is no pairing code to quote. There is also no paired data: STEW (EEG; 48 people, rest vs
SIMKAP) and DAiSEE (video; different people, watching lectures) are unrelated datasets. A fusion
model trained on them would combine samples from different people at different times. Its
explanations, such as "EEG contributed 60 %", would describe an artificial pairing, not real
multimodal learners. **The multimodal context is therefore not possible now.** It would also be
invalid if someone built it from STEW + DAiSEE.

## g) Classes and time alignment versus EEG

* **Classes differ.** EEG (C1, T1) is rest vs SIMKAP task, labelled "workload low/high". Vision is DAiSEE
  engagement 0–3 (or binary), crowd-annotated by observers. These are different constructs, different
  label sources and different tasks. A cross-modality comparison can only compare **how the XAI methods
  rank** on two different prediction problems. It cannot say "which modality explains workload better".
* **Window grid:** 20 samples / 5 fps = **4 s window, 2 s stride**. This matches EEG's 4 s / 2 s epoch
  grid numerically. There is no shared timeline, though, because the recordings are not simultaneous.
  The EEG model A8 also uses 8 epochs (18 s), while the vision model uses one 4-s window.

## h) Dependencies and Python version

| | C2 (`requirements.txt`) | component3/.venv (measured) |
|---|---|---|
| Python | 3.11 (also verified on 3.13 per CHANGELOG 1.1.0) | 3.13.7 |
| numpy | `>=1.26,<2.0` | 2.5.4 |
| scikit-learn | `>=1.4,<1.6` | 1.9.1 |
| torch | `>=2.2,<2.6` | 2.14.1 |
| pandas | `>=2.1,<2.3` | 3.0.6 |
| mediapipe | `==0.10.14` | not installed (no 3.13 wheel) |
| opencv-contrib-python | `>=4.9,<4.11` | not installed |

Consequences:
* **Temporal model (PyTorch) is usable from component3.** It is plain `nn.GRU`/`nn.Linear`, the state_dict
  loads across torch versions, and the Normalizer is pure numpy. Component 3 can import
  `src.models.temporal_model`, `src.processing.preprocessing` and `src.models.registry` (no side effects
  on import or load) without installing anything.
* **Baseline (joblib pickle)** made with scikit-learn ≤ 1.5 may not unpickle in 1.9. It would need a check, or
  C2's own environment.
* **Feature extraction must run in C2's own Python 3.11 environment** (MediaPipe). Component 3 never runs it.
* **Import clash:** C2's top-level package is called `src`. Adding `component2/vision_cbu` to PYTHONPATH
  makes `import src` mean C2. Component 3's package is `xaicomp` (under `component3/src/`, which is not
  itself a package), so there is no clash, but the name is generic and worth noting in env.sh.

Side effects found (functions Component 3 must **not** call, or must call with a component3 path):
`run_extraction` (writes `data/processed/features/`), `save_model`/`train_*` with `save=True` (writes
`models/`), `SequenceSet.save`, `reports._write` (`outputs/evaluation/`), `privacy._salt` (creates
`outputs/.pseudonym_salt`), `setup_logging(enable_file=True)` (creates a log dir), `run_audit(write=True)`,
`generate_demo_data`/`write_synthetic` (`data/demo/`), `src/webcam/*` and `app.py` (webcam). Safe to
call: `load_config` (read-only), `TemporalClassifier`, `Normalizer`, `registry.load_model`/`read_metadata`,
`inference.predict_proba`, `baseline.aggregate_windows`, `build_sequences`, `forward_fill_within_video`,
`prepare_data`, and `train_temporal(..., save=False)` (returns the model in memory; Component 3 would
save it itself into `component3/artifacts/`).

## i) Existing explainability in C2

None in the XAI sense. There is a feature-group **ablation** (baseline retrained without each of
eye / gaze / head_pose / facial; `configs/default.yaml` `evaluation.ablation: true`). That is global importance
by retraining, not per-prediction attribution. `interpretation.py` contains heuristic scores, not
explanations.

---

## 9. What the vision context would look like once data and models exist (design sketch, not built)

* **Interpretable space (V2):** tabular, as-is. The 15 features × a few time segments of the 4-s window
  (for example 4 × 1 s = 60 features), with groups eye / gaze / head_pose / facial. Temporal segments
  make sense because the GRU uses order. A teacher-facing summary would aggregate per group
  (and per schematic face region: eyes ← eye+gaze, brows ← brow_raise, mouth ← mouth_aspect_ratio,
  head ← head_pose + face position/scale).
* **"Absent":** off-manifold = training mean (0 after C2's standardisation, which is also IG-zero).
  On-manifold = values from windows of training subjects (donors), consistent with EEG.
* **Conditional SHAP:** Gaussian on the 60-d (or 15-d) vector, Ledoit-Wolf on training subjects.
  Blink event is binary and the quality-gated values are not Gaussian, so the Gaussian fit must be
  checked, with the empirical nearest-donor mode as fallback.
* **IG:** applicable to the GRU (differentiable). It is excluded for the sklearn baseline (or
  computed analytically for logistic regression; to be decided at screening).
* **Out-of-sample:** C2 uses one official train/val/test split, not folds. If the split is
  subject-disjoint, the test-split windows of C2's deployed model are already out-of-sample, and
  explanations should be computed on the **test** split only. Training-split windows would be IN-SAMPLE.
  Donors and the Gaussian come from the train split. Fold retraining is optional, for a subject-level
  analysis across more people.
* **Display:** the DAiSEE licence does not allow showing faces, so the dashboard would use a schematic face.
* **Classes:** binary engagement (`daisee_binary`) would match the EEG "P(high)" format. The 4-class
  default would need a choice of explained class (predicted class, or P(engagement ≥ 2)).

## 10. STOP conditions

| Condition | Status |
|---|---|
| C2 model weights missing and not producible in < 30 min | **APPLIES**: `models/` is empty, and producing weights needs DAiSEE + MediaPipe extraction (hours, in C2's 3.11 env) |
| C2 dataset not on this machine | **APPLIES**: DAiSEE is absent everywhere searched |
| Framework not PyTorch and no sound way to run the 4 explainers | Does **not** apply: the main model is PyTorch (IG via Captum works); the baseline is sklearn (IG screened out) |
| Fusion pairing not genuine | **APPLIES (stronger)**: no fusion model exists, and no paired EEG+video data exists. STEW + DAiSEE would be unrelated people |
| Deployed vision model trained on all subjects | Not applicable yet. C2's design trains on the train split only, so the test split would be out-of-sample if DAiSEE's split is subject-disjoint (to verify with C2's audit) |

Phase 2 was therefore **not started**. Nothing was built from C2 synthetic data (rule 9).
