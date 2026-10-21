# Changelog

The running version is shown in the dashboard sidebar and by `python scripts\check_environment.py`.

## 1.2.0 - 2026-10-21

### Added
- **Experimental heuristic scores** `attention_score`, `fatigue_score`, `readiness_score` (floats in
  [0, 1], null when UNKNOWN), computed by fixed formulas (`heuristic_v1`) from the predicted
  engagement level and window-level visual proxies (face visibility, eye openness, blink rate,
  head-pose stability). **Not trained and not validated**; the trained model still predicts the
  DAiSEE engagement level 0-3 only. Formulas: `docs/labeling_protocol.md` section 6.
- New per-window fields: `engagement_level`, `quality_state` (`OK`/`UNKNOWN`),
  `interpretation_source`, `interpretation_reason`.
- Live webcam page and `scripts/run_webcam.py` show the scores (0-1 and %) labelled EXPERIMENTAL.
- Session analytics: score summary (mean/median/min/max, display bins), score timeline, disclaimer.
- Exports: score columns in CSV/JSON; JSON carries a `heuristic_scores` block with the disclaimer and weights.
- Config section `interpretation` (enable flag, version, weights, proxy ranges); toggle in Settings.
- Tests: `tests/test_interpretation.py` and a dashboard test.

### Dashboard redesign
- Light theme enforced globally (`.streamlit/config.toml` `[theme]`), independent of the OS
  dark-mode setting. Teal accent, card-style status panels, status pills, consistent page
  headers and section dividers, cleaner sidebar with mode / model status.
- Removed setup walkthroughs from the UI (PowerShell quick start, DAiSEE download and folder
  instructions, script commands); these remain in `README.md` and `docs/`.
- Dataset audit results shown as metrics and tables instead of raw JSON; leakage checks as pills.
- Session analytics: grouped sections, two-column feature charts, state counts with shares.
- **Fixed:** evaluation figures appeared as broken images on Streamlit 1.60 (its media-file
  server fails to send `st.image` files). Figures are now embedded inline.
- Replaced deprecated `use_container_width` with `width="stretch"`.

### Unchanged
- Engagement model, training, feature schema (1.0.0) and evaluation.

## 1.1.0 - 2026-10-16
Verified on Python 3.11 with the pinned requirements (numpy 1.26, pandas 2.2, scikit-learn 1.5,
torch 2.5, OpenCV 4.10, MediaPipe 0.10.14, Streamlit 1.60) and on Python 3.13, including the
real MediaPipe face-landmark backend on a real face image.

### Fixed
- **Live capture stopped after one frame with `'{' was never closed`.** Caused by Streamlit
  "magic" rendering a one-line `if/else` expression. The expression is rewritten, magic is
  disabled in `.streamlit/config.toml`, and a test forbids bare expressions in `app.py`.
- **Browser console `500` / "Image source error" during capture.** Frames are now sent inline
  instead of through Streamlit's media storage, and the screen refreshes about 10 times per
  second (every frame is still analysed).
- **Dashboard crashed with `EmptyDataError` on Windows before any dataset existed.** The empty
  audit table (`\r\n` on Windows) is now read safely. Empty/invalid uploaded CSVs are reported
  instead of crashing.
- **Gaze unit test failed on NumPy 1.26** (exact boundary comparison).
- **Mock camera ran far longer than `--seconds`** (duration now uses the camera clock).

### Improved
- **Head pose** now matches all 468 landmarks to MediaPipe's canonical 3D face model
  (`src/vision/data/canonical_face_vertices.csv`) instead of a generic 6-point model. Known
  rotations are recovered to < 0.5 deg in tests; a mirrored real photo gives opposite yaw and
  pitch within ~1 deg. **Pitch sign corrected: positive now means looking down.**
- **Capture is fault-tolerant**: a failing frame is skipped and counted instead of ending the
  capture; full error details are shown in an expander.
- **Final screen refresh** at the end of a capture, so the state shown reflects the whole capture.
- **Webcam opening on Windows** tries DirectShow, then Media Foundation, then the default backend,
  and discards warm-up frames.
- New dashboard tests (every page, mock capture end to end). Test count: 90.

## 1.0.0 - 2026-10-16
Initial release.
