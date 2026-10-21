"""Streamlit dashboard - Vision-Based Cognitive Behaviour Understanding.

Run from the project root:   streamlit run app.py
"""
from __future__ import annotations

import html
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.dashboard.components import (image_file, inject_css, mode_banner, note, page_header, pill, score_box, section,
                                      show_frame, state_box, status_card, timeline)
from src.data.dataset_audit import load_latest_audit, run_audit
from src.data.feature_schema import FEATURE_COLUMNS, FEATURE_GROUPS, QUALITY_COLUMNS, schema_description
from src.evaluation.reports import DEMO_BANNER, NOT_EVALUATED, list_reports, load_report
from src.models.inference import NO_MODEL, UNKNOWN, load_live_predictor
from src.models.interpretation import (DISCLAIMER, SCORE_COLUMNS, empty_interpretation, export_metadata,
                                       interpret_window, interpretation_config, summarize_scores)
from src.processing.quality_control import quality_score, quality_warnings
from src.utils.config import load_config, resolve_path
from src.utils.privacy import (ConsentRequiredError, delete_sessions, pseudonym_for, save_session, to_csv_bytes,
                               to_json_bytes)
from src import __version__
from src.utils.status import RESEARCH_MODE, get_status

st.set_page_config(page_title="Vision CBU", page_icon="👁", layout="wide")
inject_css()

PAGES = ["Home & status", "Live webcam", "Feature explorer", "Dataset setup & audit", "Model training",
         "Model evaluation", "Session analytics", "Settings & privacy"]


def cfg() -> dict:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load_config()
    return st.session_state["cfg"]


def live_rows() -> list[dict]:
    return st.session_state.setdefault("live_rows", [])


status = get_status(cfg())
st.sidebar.markdown('<p class="vcbu-brand">Vision CBU</p>'
                    '<p class="vcbu-brand-sub">Cognitive behaviour understanding<br>for intelligent learning systems</p>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
_v = status["validated_model"]
st.sidebar.markdown(
    f'<div class="vcbu-side-foot">{pill(status["mode"], "ok" if status["mode"] == RESEARCH_MODE else "warn")}'
    f'<br><br>Model: {pill("validated" if _v else "none", "ok" if _v else "bad")}'
    f'<br><br>v{__version__} · vision only (no EEG)</div>', unsafe_allow_html=True)
mode_banner(status)


# ------------------------------------------------------------------ 1. Home
def page_home():
    page_header("Vision-Based Cognitive Behaviour Understanding",
                "Webcam visual features, temporal modelling and evaluation against DAiSEE engagement labels.")
    section("System status", first=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        status_card("Dataset", f"{status['n_videos']} videos · {status['n_annotation_files']} label files",
                    "ok" if status["dataset_present"] else "warn",
                    "present" if status["dataset_present"] else "not found")
        audit_ok = status["audit_status"] == "DATASET_READY"
        status_card("Last audit", status["audit_status"], "ok" if audit_ok else "warn", "ready" if audit_ok else "check")
    with c2:
        status_card("Extracted feature files", status["n_feature_files"], "ok" if status["n_feature_files"] else "warn",
                    "available" if status["n_feature_files"] else "none")
        status_card("Trained models (real data)", len(status["real_models"]), "ok" if status["real_models"] else "warn",
                    "available" if status["real_models"] else "none")
    with c3:
        v = status["validated_model"]
        status_card("Validated model", v["model_name"] if v else "None", "ok" if v else "bad",
                    "validated" if v else "missing")
        status_card("Synthetic demo data", "Generated" if status["demo_data_present"] else "Not generated",
                    "warn" if status["demo_data_present"] else "info", "demo only")
    note("A model counts as validated only when it was trained on real labelled data and has a held-out "
         "evaluation report. Without one, live engagement prediction is disabled.")


# ------------------------------------------------------------------ 2. Live webcam
def page_live():
    from src.vision.face_landmarks import draw_landmarks
    from src.vision.feature_extractor import FrameFeatureExtractor
    from src.webcam.camera import Camera, CameraError, MockCamera

    page_header("Live webcam", "Frame-level feature extraction with window-level engagement prediction. "
                               "Raw video is never stored.")
    c = cfg()
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="center")
        source = col1.selectbox("Input", ["Webcam", "Mock camera (no hardware)"])
        seconds = col2.slider("Capture duration (s)", 5, 120, 20)
        show_lm = col3.toggle("Draw landmarks", value=True)
        use_demo = st.checkbox("Use a DEMO ONLY model (software test, not a valid estimate)", value=False,
                               disabled=not status["demo_models"])
    predictor, model_msg = load_live_predictor(c)
    if predictor is None and use_demo and status["demo_models"]:
        predictor, model_msg = load_live_predictor(c, status["demo_models"][0]["path"], allow_demo=True)
    if predictor is None:
        st.error(f"{NO_MODEL}. Features are still extracted; engagement prediction is disabled.")
    elif predictor.meta["demo_only"]:
        st.warning(f"{model_msg}: {predictor.meta['model_name']}. Output is labelled DEMO ONLY.")
    else:
        st.success(model_msg)
    show_scores = bool(interpretation_config(c)["enabled"])
    if show_scores:
        note(f"Heuristic scores: {DISCLAIMER}")

    if not st.button("Start capture", type="primary"):
        if live_rows():
            st.caption(f"Last capture: {len(live_rows())} samples (see Session analytics).")
        return
    detector = None
    if source.startswith("Mock"):
        from src.vision.mock_landmarks import MockLandmarkDetector
        detector = MockLandmarkDetector([{"eye_open": 1.0}] * 10 + [{"eye_open": 0.05}] * 3 + [{"gaze_shift": 0.6}] * 5)
    try:
        extractor = FrameFeatureExtractor(c, detector=detector, source="webcam",
                                          participant_id=pseudonym_for(c, st.session_state.get("participant", "anonymous")),
                                          video_id=f"live_{int(time.time())}")
    except Exception as exc:
        st.error(f"Could not initialise the face backend: {exc}")
        return
    st.caption(f"Face backend: {extractor.backend}")
    img_slot, info_slot = st.columns([3, 2])
    frame_ph, state_ph, score_ph, feat_ph, warn_ph = (img_slot.empty(), info_slot.empty(), info_slot.empty(),
                                                      info_slot.empty(), info_slot.empty())
    rows: list[dict] = []
    ui_interval_s = 0.1                         # extract every frame, redraw and predict ~10x per second
    max_consecutive_errors = 15
    last_ui = 0.0
    last_pred = (NO_MODEL, "no validated model") if predictor is None else (UNKNOWN, "waiting for data")
    last_interp = empty_interpretation(c, "waiting for data")
    win = predictor.window if predictor is not None else int(c["sequences"]["window_size"])

    def predict_and_interpret(recent: list[dict]) -> tuple[tuple[str, str], dict]:
        """Engagement prediction (unchanged model) + heuristic scores on the same window."""
        df = pd.DataFrame(recent)
        pred = predictor.predict_frames(df) if predictor is not None else None
        p = (pred["state"], pred["reason"]) if pred is not None else last_pred
        return p, interpret_window(pred, df.tail(win), c)
    n_errors, consecutive, first_error = 0, 0, None
    stopped_reason = None
    def render(frame, res, row) -> int:
        """Draw one frame + its features; returns 1 on a display error, else 0."""
        nonlocal first_error
        try:
            vis = draw_landmarks(frame, res) if (show_lm and res is not None) else frame
            show_frame(frame_ph, vis)
            state_box(row["prediction"],
                      row["prediction_reason"] if row["prediction"].startswith(UNKNOWN) else None, target=state_ph)
            if show_scores:
                score_box(row, target=score_ph)
            values = [round(float(row[f]), 4) if pd.notna(row[f]) else None for f in FEATURE_COLUMNS]
            feat_ph.dataframe(pd.DataFrame({"feature": FEATURE_COLUMNS, "value": values}),
                              hide_index=True, height=300, width="stretch")
            warnings_now = quality_warnings(row)
            if warnings_now:
                warn_ph.warning(" | ".join(warnings_now))
            else:
                warn_ph.success("Frame quality OK")
            return 0
        except Exception:
            first_error = first_error or traceback.format_exc()
            return 1

    last_shown = None
    cam = MockCamera(sample_every_n=c["camera"]["sample_every_n_frames"]) if source.startswith("Mock") else Camera(c)
    try:
        with cam:
            if getattr(cam, "api_name", None):
                st.caption(f"Camera opened with the {cam.api_name} backend")
            for i, (ts, frame) in enumerate(cam.frames(max_seconds=seconds)):
                # 1. feature extraction - one bad frame must not end the capture
                try:
                    row, res = extractor.extract(frame, ts, i)
                except Exception as exc:
                    n_errors += 1
                    consecutive += 1
                    first_error = first_error or traceback.format_exc()
                    if consecutive >= max_consecutive_errors:
                        raise RuntimeError(f"{consecutive} consecutive frames failed: {exc}") from exc
                    continue
                consecutive = 0
                last_shown = (frame, res, row)          # newest frame + its features
                if source.startswith("Mock"):
                    time.sleep(0.03)            # mock camera: roughly real-time pacing
                now = time.monotonic()
                refresh = now - last_ui >= ui_interval_s
                # 2. prediction (only when a model is loaded) + heuristic scores, on UI ticks
                if refresh:
                    try:
                        last_pred, last_interp = predict_and_interpret(rows[-300:] + [row])
                    except Exception as exc:
                        n_errors += 1
                        first_error = first_error or traceback.format_exc()
                        if predictor is not None:
                            last_pred = (UNKNOWN, f"prediction error: {exc}")
                        last_interp = empty_interpretation(c, f"prediction error: {exc}")
                row["prediction"], row["prediction_reason"] = last_pred
                row.update(last_interp)
                rows.append(row)
                if not refresh:
                    continue
                last_ui = now
                # 3. display
                n_errors += render(frame, res, row)
    except CameraError as exc:
        stopped_reason = f"Camera unavailable: {exc}"
    except Exception as exc:
        stopped_reason = f"Capture stopped: {exc}"
        first_error = first_error or traceback.format_exc()
    finally:
        extractor.close()                       # the camera itself is released by the context manager
    if last_shown is not None and rows:         # final refresh so the screen reflects the whole capture
        frame, res, _ = last_shown
        try:
            (rows[-1]["prediction"], rows[-1]["prediction_reason"]), interp = predict_and_interpret(rows[-300:])
            rows[-1].update(interp)
        except Exception as exc:
            if predictor is not None:
                rows[-1]["prediction"], rows[-1]["prediction_reason"] = UNKNOWN, f"prediction error: {exc}"
            rows[-1].update(empty_interpretation(c, f"prediction error: {exc}"))
        n_errors += render(frame, res, rows[-1])
    st.session_state["live_rows"] = rows
    if stopped_reason:
        st.error(stopped_reason)
    if n_errors:
        st.warning(f"{n_errors} frame(s) had processing errors and were skipped or shown without overlay.")
    if first_error:
        with st.expander("Error details"):
            st.code(first_error, language="text")
    st.success(f"Captured {len(rows)} samples. Camera released; results are in Session analytics.")


# ------------------------------------------------------------------ 3. Feature explorer
def page_features():
    page_header("Feature explorer", "Per-frame visual features and quality flags for one video or session.")
    c = cfg()
    sources = ["Live session (this browser)", "Synthetic demo data (DEMO ONLY)", "Real extracted features",
               "Upload CSV"]
    s1, s2 = st.columns(2)
    src = s1.selectbox("Data source", sources)
    df = pd.DataFrame()
    if src.startswith("Live"):
        df = pd.DataFrame(live_rows())
    elif src.startswith("Synthetic"):
        p = resolve_path(c, "demo_dir") / "synthetic_features.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"video_id": str, "participant_id": str})
            st.warning("Synthetic data for software testing. Not real participants.")
    elif src.startswith("Real"):
        from src.data.extract_features import load_feature_table
        df = load_feature_table(c)
    else:
        up = st.file_uploader("Feature CSV (schema v1)", type="csv")
        if up:
            try:
                df = pd.read_csv(up, dtype={"video_id": str, "participant_id": str})
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not read that CSV: {exc}")
                return
            missing = [col for col in ("video_id", "timestamp_s", "quality_ok", "is_synthetic") if col not in df.columns]
            if missing:
                st.error(f"Not a feature file (schema v1): missing columns {missing}")
                return
    if df.empty:
        st.info("No data for this source.")
        with st.expander("Feature schema"):
            st.json(schema_description())
        return
    vids = sorted(df["video_id"].astype(str).unique())
    vid = s2.selectbox("Video / session", vids)
    sub = df[df["video_id"].astype(str) == vid].sort_values("timestamp_s")
    a, b, c3 = st.columns(3)
    a.metric("Samples", len(sub))
    b.metric("Quality-OK ratio", f"{quality_score(sub):.2f}")
    c3.metric("Synthetic", "Yes" if str(sub["is_synthetic"].iloc[0]).lower() == "true" else "No")
    section("Feature timelines")
    group = st.multiselect("Feature groups", list(FEATURE_GROUPS), default=["eye", "head_pose"])
    for g in group:
        timeline(sub, FEATURE_GROUPS[g], f"{g.replace('_', ' ').capitalize()} features")
    section("Data completeness", "missing values per feature (%)")
    st.bar_chart((sub[FEATURE_COLUMNS].isna().mean() * 100).rename("missing %"), height=240)
    with st.expander("Quality flags per frame"):
        st.dataframe(sub[["timestamp_s"] + QUALITY_COLUMNS], hide_index=True)
    section("Export")
    synth = bool(sub["is_synthetic"].astype(str).str.lower().eq("true").any())
    e1, e2, _ = st.columns([1, 1, 3])
    e1.download_button("Export CSV", to_csv_bytes(sub), f"features_{vid}.csv", "text/csv", width="stretch")
    e2.download_button("Export JSON", to_json_bytes(sub, synthetic=synth), f"features_{vid}.json", "application/json",
                       width="stretch")


# ------------------------------------------------------------------ 4. Dataset
def _dict_table(d: dict, key: str, value: str) -> pd.DataFrame:
    return pd.DataFrame({key: list(d.keys()), value: list(d.values())})


def page_dataset():
    page_header("Dataset setup & audit", "Integrity, label and split checks for the local DAiSEE copy.")
    c = cfg()
    note(f"Place dataset files under {c['paths']['daisee_root']}/ (videos/ and annotations/), then run the audit.")
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 1.3], vertical_alignment="bottom")
        probe = col1.toggle("Read video metadata", value=True)
        max_probe = col2.number_input("Max videos to open (0 = all)", 0, 100000, 0)
        run = col3.button("Run audit", type="primary", width="stretch")
    if run:
        with st.spinner("Auditing..."):
            run_audit(c, probe=probe, max_probe=max_probe or None)
    rep, clips = load_latest_audit(c)
    if rep is None:
        st.info("No audit has been run yet.")
        return
    ready = rep["status"] == "DATASET_READY"
    st.markdown(f'Status {pill(rep["status"], "ok" if ready else "warn")} '
                f'<span class="vcbu-note">generated {html.escape(str(rep.get("generated_utc", "")))}</span>',
                unsafe_allow_html=True)
    for m in rep.get("messages", []):
        st.warning(m)
    if rep["status"] == "DATASET_NOT_AVAILABLE":
        return
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clips", rep.get("n_clips_total", 0))
    k2.metric("Included", rep.get("n_clips_included", 0))
    k3.metric("Excluded", rep.get("n_clips_excluded", 0))
    k4.metric("Unreadable", rep.get("n_unreadable_videos", 0))
    k5.metric("Participants", rep.get("participants", {}).get("n_participants", "-"))

    section("Splits")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.caption("Clips per split")
        st.dataframe(_dict_table(rep.get("split_counts", {}), "split", "clips"), hide_index=True)
    with s2:
        st.caption("Participants per split")
        st.dataframe(_dict_table(rep.get("participants", {}).get("per_split", {}), "split", "participants"),
                     hide_index=True, width="stretch")
    with s3:
        st.caption("Exclusion reasons")
        st.dataframe(_dict_table(rep.get("exclusion_reasons", {}), "reason", "clips"), hide_index=True)

    dists = rep.get("class_distribution", {})
    if dists:
        section("Class distribution", "included clips, per split")
        tabs = st.tabs([lab.capitalize() for lab in dists])
        for tab, dist in zip(tabs, dists.values()):
            tab.dataframe(pd.DataFrame(dist).fillna(0).astype(int).sort_index())

    section("Leakage checks")
    leak = rep.get("leakage", {})
    overlap = leak.get("participant_overlap", {})
    multi = leak.get("clips_in_multiple_annotation_splits", [])
    l1, l2 = st.columns(2)
    l1.markdown("Participant overlap between splits "
                + pill("none" if not overlap else len(overlap), "ok" if not overlap else "bad"),
                unsafe_allow_html=True)
    l2.markdown("Clips in multiple splits " + pill("none" if not multi else len(multi), "ok" if not multi else "bad"),
                unsafe_allow_html=True)
    if overlap or multi:
        st.json(leak)

    with st.expander("Annotation files"):
        st.dataframe(pd.DataFrame(rep["annotation_files"]), hide_index=True)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Audit report (JSON)", json.dumps(rep, indent=2, default=str), "audit_report.json",
                       width="stretch")
    if clips is not None and not clips.empty:
        d2.download_button("Clip table (CSV)", to_csv_bytes(clips), "audit_clips.csv", width="stretch")


# ------------------------------------------------------------------ 5. Training
def page_training():
    from src.data.validate_dataset import RealDataUnavailableError
    from src.models.labels import LabelSchemeError
    from src.models.train import train_demo, train_real

    page_header("Model training", "Frame-feature baseline and temporal (GRU/LSTM) models on DAiSEE engagement.")
    c = cfg()
    section("Parameters", first=True)
    with st.form("params", border=True):
        a, b, d = st.columns(3)
        cell = a.selectbox("Temporal cell", ["gru", "lstm"], index=0 if c["temporal"]["cell"] == "gru" else 1)
        hidden = b.number_input("Hidden size", 8, 512, int(c["temporal"]["hidden_size"]))
        epochs = d.number_input("Epochs", 1, 500, int(c["temporal"]["epochs"]))
        a2, b2, d2 = st.columns(3)
        window = a2.number_input("Window (timesteps)", 2, 300, int(c["sequences"]["window_size"]))
        stride = b2.number_input("Stride", 1, 300, int(c["sequences"]["stride"]))
        base = d2.selectbox("Baseline", ["logistic_regression", "random_forest"])
        st.form_submit_button("Apply parameters")
    c["temporal"].update(cell=cell, hidden_size=int(hidden), epochs=int(epochs))
    c["sequences"].update(window_size=int(window), stride=int(stride))
    c["baseline"]["type"] = base

    section("Real data", "DAiSEE")
    real_ready = status["n_feature_files"] > 0
    if not real_ready:
        st.error("Real training unavailable: no extracted feature files found.")
    else:
        st.caption(f"{status['n_feature_files']} extracted feature files available.")
    r1, r2, _ = st.columns([1, 1, 2])
    for col, kind in ((r1, "baseline"), (r2, "temporal")):
        if col.button(f"Train {kind}", disabled=not real_ready, key=f"real_{kind}", width="stretch"):
            try:
                with st.spinner(f"Training {kind}..."):
                    meta = train_real(c, kind)
                st.success(f"Saved {meta['model_name']}. Evaluate it to obtain test metrics.")
            except (RealDataUnavailableError, LabelSchemeError) as exc:
                st.error(f"Refused: {exc}")

    section("Synthetic data", "pipeline check only")
    note("Demo models are stored separately and are never used for live inference unless explicitly selected.")
    g, d1, d2, _ = st.columns([1.3, 1, 1, 0.7])
    if g.button("Generate synthetic data", width="stretch"):
        from src.data.synthetic import generate_synthetic_features, write_synthetic
        write_synthetic(generate_synthetic_features(seed=c["project"]["seed"]), resolve_path(c, "demo_dir"))
        st.success("Synthetic data generated.")
    for col, kind in ((d1, "baseline"), (d2, "temporal")):
        if col.button(f"Train demo {kind}", disabled=not status["demo_data_present"], key=f"demo_{kind}",
                      width="stretch"):
            with st.spinner("Training on synthetic data..."):
                meta = train_demo(c, kind)
            st.warning(f"DEMO ONLY model saved: {meta['model_name']}")
            hist = meta["history"].get("epochs")
            if hist:
                st.markdown('<div class="vcbu-chart-title">Training curve (loss per epoch)</div>',
                            unsafe_allow_html=True)
                st.line_chart(pd.DataFrame(hist).set_index("epoch")[["train_loss", "val_loss"]], height=220)


# ------------------------------------------------------------------ 6. Evaluation
def show_report(path: Path, demo: bool):
    rep = load_report(path)
    if demo:
        st.warning(DEMO_BANNER)
    parts = rep["split"]["participants"]
    if isinstance(parts, dict):
        parts = " · ".join(f"{k} {v}" for k, v in parts.items())
    meta = [("Dataset", rep["dataset"]), ("Label scheme", rep["label_scheme"]["name"]),
            ("Split", rep["split"]["source"]), ("Participants", parts),
            ("Generated", rep["generated_utc"])]
    st.markdown(" ".join(pill(f"{k}: {v}", "info") for k, v in meta), unsafe_allow_html=True)
    section("Model comparison")
    st.dataframe(pd.DataFrame(rep["comparison"]), hide_index=True)
    imgs = sorted(path.glob("*.png"))
    if imgs:
        section("Figures")
        cols = st.columns(min(3, len(imgs)))
        for i, im in enumerate(imgs):
            image_file(im, im.stem.replace("_", " ").capitalize(), target=cols[i % len(cols)])
    if rep.get("ablation"):
        section("Feature-group ablation", "baseline model")
        st.dataframe(pd.DataFrame(rep["ablation"]), hide_index=True)
    section("Per-class metrics")
    for m in rep["models"]:
        with st.expander(m["model_name"]):
            st.dataframe(pd.DataFrame(m["window_level"]["per_class"]), hide_index=True)
    st.download_button("Report (JSON)", (path / "report.json").read_bytes(), f"{path.name}_report.json",
                       key=f"dl_{path}")


def page_evaluation():
    page_header("Model evaluation", "Held-out test results for DAiSEE engagement (levels 0-3).")
    c = cfg()
    reports = list_reports(c)
    if not reports:
        st.error(NOT_EVALUATED)
    else:
        sel = st.selectbox("Evaluation run", reports, format_func=lambda p: p.name)
        show_report(sel, demo=False)
    demo_reports = list_reports(c, demo=True)
    if demo_reports:
        with st.expander("Synthetic pipeline check (DEMO ONLY, not research results)"):
            sel = st.selectbox("Demo run", demo_reports, format_func=lambda p: p.name)
            show_report(sel, demo=True)


# ------------------------------------------------------------------ 7. Session analytics
def page_session():
    page_header("Session analytics", "Summary of the most recent live capture in this browser session.")
    c = cfg()
    df = pd.DataFrame(live_rows())
    if df.empty:
        st.info("No live session yet. Capture one on the Live webcam page.")
        return
    a, b, d, e = st.columns(4)
    a.metric("Samples", len(df))
    b.metric("Duration (s)", f"{df['timestamp_s'].max():.1f}")
    d.metric("Quality-OK ratio", f"{quality_score(df):.2f}")
    e.metric("Blinks detected", int(df["blink_event"].fillna(0).sum()))
    if "engagement_level" in df:
        df["engagement_level"] = pd.to_numeric(df["engagement_level"], errors="coerce").astype("Int64")
    for col in SCORE_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    has_scores = all(col in df for col in SCORE_COLUMNS)

    if "prediction" in df:
        section("Prediction states over time", "samples per engagement state")
        counts = df.groupby("prediction").size().rename("samples").reset_index()
        counts["share"] = (counts["samples"] / counts["samples"].sum()).map("{:.0%}".format)
        st.dataframe(counts, hide_index=True)
    if has_scores:
        section("Heuristic scores", "0-1, experimental")
        st.warning(DISCLAIMER)
        st.dataframe(summarize_scores(df), hide_index=True)
        st.caption("Bins are display counts; stored values are continuous. n_unknown: quality UNKNOWN or no "
                   "engagement level.")
        timeline(df, SCORE_COLUMNS, "Scores over time (gaps = UNKNOWN)")

    section("Visual features")
    note("Single blinks or gaze shifts are not evidence of distraction; interpret windows, not frames.")
    left, right = st.columns(2)
    with left:
        timeline(df, ["ear_mean"], "Eye aspect ratio")
        timeline(df, ["head_yaw", "head_pitch", "head_roll"], "Head pose (degrees)")
    with right:
        timeline(df, ["blink_rate_per_min"], "Blink rate (per min, trailing)")
        timeline(df, ["gaze_h", "gaze_v"], "Gaze proxies (uncalibrated)")
    df["quality_ok_num"] = df["quality_ok"].astype(int)
    timeline(df, ["quality_ok_num"], "Frame quality (1 = OK)", height=140)

    section("Export and save")
    export_df = df.drop(columns=["quality_ok_num"])
    x1, x2, _ = st.columns([1, 1, 2])
    x1.download_button("Session CSV", to_csv_bytes(export_df), "session_features.csv", width="stretch")
    x2.download_button("Session JSON",
                       to_json_bytes(export_df, synthetic=False,
                                     interpretation=export_metadata(c) if has_scores else None),
                       "session_features.json", width="stretch")
    consent = st.checkbox("I consent to saving these extracted features (no video) to outputs/sessions/.")
    if st.button("Save session"):
        try:
            p = save_session(export_df, c, consent, df["participant_id"].iloc[0])
            st.success(f"Saved: {p.name}")
        except ConsentRequiredError as exc:
            st.error(str(exc))


# ------------------------------------------------------------------ 8. Settings
def page_settings():
    page_header("Settings & privacy",
                "Changes apply to this browser session; persistent values are in configs/default.yaml.")
    c = cfg()
    section("Camera", first=True)
    a, b, d = st.columns(3)
    c["camera"]["index"] = a.number_input("Camera index", 0, 10, int(c["camera"]["index"]))
    c["camera"]["sample_every_n_frames"] = b.number_input("Process 1 of every N frames", 1, 30,
                                                          int(c["camera"]["sample_every_n_frames"]))
    res = d.selectbox("Resolution", ["640x480", "1280x720", "320x240"])
    c["camera"]["width"], c["camera"]["height"] = map(int, res.split("x"))
    section("Vision thresholds")
    a, b = st.columns(2)
    c["vision"]["ear_blink_threshold"] = a.slider("EAR blink threshold", 0.10, 0.35, float(c["vision"]["ear_blink_threshold"]), 0.01)
    c["dashboard"]["min_quality_for_prediction"] = b.slider("Min window quality for a prediction", 0.0, 1.0,
                                                            float(c["dashboard"]["min_quality_for_prediction"]), 0.05)
    section("Heuristic scores", "experimental")
    c.setdefault("interpretation", {})
    c["interpretation"]["enabled"] = st.toggle(
        "Show attention / fatigue / readiness scores",
        value=bool(interpretation_config(c)["enabled"]))
    note(DISCLAIMER)
    section("Privacy")
    p1, p2 = st.columns(2)
    st.session_state["participant"] = p1.text_input("Participant label (stored as a salted pseudonym)",
                                                    st.session_state.get("participant", "anonymous"))
    p2.text_input("Pseudonym used in data", pseudonym_for(c, st.session_state["participant"]), disabled=True)
    c["privacy"]["enable_session_logging"] = st.toggle("Enable session logging (features only; consent required)",
                                                       value=bool(c["privacy"]["enable_session_logging"]))
    note("Raw video is never recorded. Saved sessions contain numerical features and a pseudonymous ID. "
         "DAiSEE videos are licensed and must not be redistributed.")
    b1, b2, _ = st.columns([1.2, 1.2, 1.6])
    if b1.button("Delete saved session files", width="stretch"):
        st.success(f"Deleted {delete_sessions(c)} file(s).")
    if b2.button("Clear in-memory session", width="stretch"):
        st.session_state["live_rows"] = []
        st.success("Cleared.")
    with st.expander("Current configuration (YAML)"):
        st.code(yaml.safe_dump({k: v for k, v in c.items() if not k.startswith("_")}, sort_keys=False), language="yaml")


PAGE_FUNCS = {"Home & status": page_home, "Live webcam": page_live, "Feature explorer": page_features,
              "Dataset setup & audit": page_dataset, "Model training": page_training,
              "Model evaluation": page_evaluation, "Session analytics": page_session,
              "Settings & privacy": page_settings}
PAGE_FUNCS[page]()
