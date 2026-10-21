# Final requirements checklist (version 1.1.0)

Verified environments:
* **Python 3.11.15** with the pinned `requirements.txt`: numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2,
  torch 2.5.1, OpenCV 4.10.0, MediaPipe 0.10.14 (legacy FaceMesh backend), Streamlit 1.60.0, protobuf 4.25.9.
* Python 3.13 with newer libraries (numpy 2.5, pandas 3.0, Streamlit 1.66, MediaPipe 0.10.32).

Both were Linux machines. The original Windows-only failures (`\r\n` empty CSV; Streamlit magic crash)
were reproduced and fixed, and regression tests were added. **Not available while building: a physical
webcam and a Windows PC.** The real MediaPipe backend was exercised on a real face photo streamed through
the dashboard's capture loop and through `run_webcam.py`.

Legend: ✅ implemented and verified · 🟡 implemented, verify on your machine · ⏳ requires the real dataset

## Automated tests - 90 passed, 0 failed on both environments (`python -m pytest`)
| File | Tests | Covers |
|---|---|---|
| test_dataset_audit.py | 12 | empty dataset (module + script exit 0), refusal, column discovery incl. `"Frustration "`, invalid rows, split inference, fake DAiSEE audit, missing/unreadable videos, participant leakage, partial dataset, leakage assertion, Windows empty-CSV regression |
| test_feature_schema.py | 28 | schema, validation, synthetic rejection, version/columns, invalid timestamps, label range, EAR, blinks, gaze, **head-pose recovery of 16 known rotations**, sign convention, canonical model file, extractor edge cases (no face / multiple faces / invalid / dark frame / bad timestamps) |
| test_sequence_builder.py | 9 | windows/stride, padding+mask, low-quality windows, timestamp cleaning, no cross-video windows, save/load, bounded fill, train-only normaliser, participant-disjoint splits |
| test_model_shapes.py | 8 | GRU/LSTM (uni/bi) shapes, padding invariance, shape errors, baseline aggregation, label schemes incl. operational refusal |
| test_inference.py | 7 | checkpoint metadata, DEMO ONLY refusal, incompatible features/labels/schema, NO VALIDATED MODEL, UNKNOWN states, validation via real report |
| test_evaluation.py | 7 | metric values, missing class, video-level & latency, refusal without data, labelled demo report, **full audit→extract→train→evaluate on a DAiSEE-shaped fixture**, synthetic rows rejected |
| test_privacy_and_exports.py | 8 | pseudonyms, consent-gated saving + deletion, no video-writing code, exports, mock camera sampling/cleanup/failure, missing camera, duration |
| test_dashboard.py | 11 | no bare expressions in app.py, magic disabled, every page renders, mock capture end to end |

Additional manual/harness checks (Python 3.11): all CLI workflows (exit codes 0/2 as documented);
real Streamlit server start (health OK); 41 scripted clicks across all dashboard pages with 0 failures;
live capture with real MediaPipe on a real face (75/75 samples, with and without a DEMO ONLY model).

## Requirement status
| Requirement | Status |
|---|---|
| Directory structure + `.gitkeep` placeholders + `data/raw/daisee/README.md` | ✅ |
| Dataset audit: runs empty, exits 0, JSON+CSV, discovered columns, metadata, unreadable, exclusions, participants, leakage, no downloads | ✅ (on generated fixture) ⏳ (real DAiSEE) |
| Engagement 0-3 first experiment; operational categories gated by documented mapping | ✅ |
| Synthetic generator in `data/demo/`, every row flagged, never mixed into real training | ✅ |
| DEMO ONLY models named/stored separately, refused for live inference by default | ✅ |
| Webcam: config index/res/fps/sampling, cleanup on stop/exception/exit, mock frames | ✅ (mock) 🟡 (physical webcam) |
| Face detection + landmark drawing, EAR, blinks, blink rate, gaze proxy, head pose, facial geometry, quality flags | ✅ (mock + real face photo via MediaPipe 0.10.14) 🟡 (live webcam) |
| No face / multiple faces / poor lighting / camera failure / invalid frames | ✅ |
| Versioned schema, caching, missing values, windows/stride, masking, length validation, normalisation, label policy | ✅ |
| Baseline (LR/RF) and GRU/LSTM, configurable hyper-parameters, checkpoint metadata, load validation, seeds, logging | ✅ |
| Real training refuses without valid real data | ✅ (exit 2 verified) |
| Evaluation: accuracy, macro P/R/F1, per-class, confusion, curves, latency, comparison, ablation | ✅ (pipeline verified on fixture/synthetic) ⏳ (real results) |
| Participant-level / official splits, no cross-split windows, train-only normalisation | ✅ |
| `NOT EVALUATED: REAL DATASET NOT AVAILABLE` / `NO VALIDATED MODEL AVAILABLE` / `UNKNOWN` | ✅ |
| Streamlit dashboard, 8 pages, mode banner, exports, graceful errors | ✅ (automated page tests + 41-step click-through on Python 3.11) |
| Privacy: no raw video, consent, logging off by default, pseudonyms, deletion, licence docs | ✅ |
| Docs: README, methodology, dataset guide, labelling protocol, evaluation template, limitations/ethics, Mermaid diagrams, demo script, checklist | ✅ |
| PowerShell workflows 1-13 documented (`README.md`, `scripts/run_checks.ps1`) | 🟡 (commands verified on Linux; PowerShell syntax not executed here) |
| No EEG functionality | ✅ |
| `requirements.txt` for Python 3.11 / Windows with conflict notes | ✅ installed and tested on Python 3.11 (Linux) · 🟡 Windows install confirmed by user |

## Known limitations of this build
* Haar fallback needs an OpenCV build that includes `CascadeClassifier` (opencv-contrib-python 4.x on Windows does).
* The webcam loop in Streamlit runs for a fixed duration; interacting with the page during a capture stops it early (the camera is still released).
* Real-data performance is unknown until DAiSEE is processed - by design nothing is claimed.
