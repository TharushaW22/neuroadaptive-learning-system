"""Component 3 dashboard. Run from component3/:   .venv/bin/streamlit run dashboard/app.py

Reads results/<run>/ only (no heavy recomputation). Sidebar sections: Overview / EEG / Computer Vision /
Multimodal / Cross-context comparison / Output for Component 4 / Glossary. The EEG pages are unchanged;
the vision pages reuse the same page functions (parameterised by context) and show a friendly empty state
until Component 2's trained model has been explained.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Comparing XAI methods — Component 3", page_icon="🧠", layout="wide")

from common import apply_style, list_runs, primary_contexts, set_modality  # noqa: E402
import views_a as A  # noqa: E402
import views_b as B  # noqa: E402
import views_vision as VV  # noqa: E402

# (title, page function, url path); EEG entries and URLs are the original pages
EEG_PAGES = [
    ("Summary", A.page_overview, "overview"),
    ("Model & data", A.page_model, "model"),
    ("Which methods can be used? (Screening)", A.page_screening, "screening"),
    ("Explanations", A.page_explanations, "explanations"),
    ("How good are the explanations? (Metrics)", A.page_metrics, "metrics"),
    ("Can we trust the metrics? (Reliability)", B.page_reliability, "reliability"),
    ("Why do methods disagree? (Disagreement)", B.page_disagreement, "disagreement"),
    ("Are the differences real? (Statistics)", B.page_statistics, "statistics"),
    ("Final recommendation (Ranking)", B.page_ranking, "ranking"),
]
VISION_PAGES = [
    ("Summary", VV.page_vision_overview, "vision-overview"),
    ("Model & data", VV.page_vision_model, "vision-model"),
    ("Which methods can be used? (Screening)", A.page_screening, "vision-screening"),
    ("Explanations", VV.page_vision_explanations, "vision-explanations"),
    ("How good are the explanations? (Metrics)", A.page_metrics, "vision-metrics"),
    ("Can we trust the metrics? (Reliability)", B.page_reliability, "vision-reliability"),
    ("Why do methods disagree? (Disagreement)", B.page_disagreement, "vision-disagreement"),
    ("Are the differences real? (Statistics)", B.page_statistics, "vision-statistics"),
    ("Final recommendation (Ranking)", B.page_ranking, "vision-ranking"),
]
# vision pages that handle "no run" themselves (the others get the shared empty state)
VISION_SELF_EMPTY = {VV.page_vision_overview, VV.page_vision_model, VV.page_vision_explanations}


def main() -> None:
    eeg_runs, vision_runs = list_runs("eeg"), list_runs("vision")
    with st.sidebar:
        st.markdown("### Component 3 · Comparing explanation methods")
        eeg_run = st.selectbox("EEG results run", eeg_runs, help="pilot = small check run; full = final experiment") \
            if eeg_runs else None
        if eeg_run:
            ctxs = primary_contexts(eeg_run)
            if len(ctxs) > 1:
                st.session_state["ctx_id"] = st.selectbox("Prediction context", [c["context_id"] for c in ctxs])
        vision_run = st.selectbox("Vision results run", vision_runs) if vision_runs else None
        if not vision_runs:
            st.caption("Vision: waiting for Component 2's trained model.")
        st.toggle("Presentation mode (larger text)", key="big_text")
    apply_style()

    def guarded(draw, title):
        def page():
            try:
                draw()
            except Exception as e:  # friendly message instead of a stack trace
                if os.environ.get("XAICOMP_DASHBOARD_DEBUG"):
                    raise
                st.warning(f"This page could not be drawn from the current results ({type(e).__name__}). "
                           "Re-run the analysis: `.venv/bin/python scripts/run_experiment.py --pilot --stage analyse`.")
        page.__name__ = title
        return page

    def eeg(fn):
        def draw():
            set_modality("eeg")
            if eeg_run is None:
                st.title("Comparing explanation methods")
                st.info("No results yet — run the pilot with:\n\n`.venv/bin/python scripts/run_experiment.py --pilot`\n\n"
                        "(from the component3/ folder), then reload this page.")
                return
            fn(eeg_run)
        return guarded(draw, f"eeg_{fn.__name__}")

    def vision(fn, title):
        def draw():
            set_modality("vision")
            if vision_run is None and fn not in VISION_SELF_EMPTY:
                st.title(title)
                VV.vision_empty()
                return
            fn(vision_run)
        return guarded(draw, f"vision_{fn.__name__}")

    def both(fn):
        def draw():
            set_modality("eeg")
            fn(eeg_run, vision_run)
        return guarded(draw, fn.__name__)

    def plain(fn):
        def draw():
            set_modality("eeg")
            fn(eeg_run)
        return guarded(draw, fn.__name__)

    sections = {
        "Overview": [st.Page(both(VV.page_project_overview), title="Overview", url_path="home", default=True)],
        "EEG": [st.Page(eeg(fn), title=t, url_path=u) for t, fn, u in EEG_PAGES],
        "Computer Vision": [st.Page(vision(fn, t), title=t, url_path=u) for t, fn, u in VISION_PAGES],
        "Multimodal": [st.Page(plain(VV.page_multimodal), title="Why there is no multimodal model", url_path="multimodal")],
        "Cross-context comparison": [st.Page(both(VV.page_cross_context), title="Cross-context comparison",
                                             url_path="cross-context")],
        "Output for Component 4": [st.Page(both(VV.page_c4_all), title="Output for Component 4", url_path="component4")],
        "Glossary": [st.Page(plain(B.page_glossary), title="Glossary", url_path="glossary")],
    }
    st.navigation(sections).run()


main()
