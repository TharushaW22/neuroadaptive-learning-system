"""Component 3 dashboard. Run from component3/:   .venv/bin/streamlit run dashboard/app.py

Reads results/<run>/ only (no heavy recomputation). Pages follow the research order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Comparing XAI methods — Component 3", page_icon="🧠", layout="wide")

from common import apply_style, list_runs, primary_contexts  # noqa: E402
import views_a as A  # noqa: E402
import views_b as B  # noqa: E402

PAGES = [
    ("Overview", A.page_overview, "overview"),
    ("Model & data", A.page_model, "model"),
    ("Which methods can be used? (Screening)", A.page_screening, "screening"),
    ("Explanations", A.page_explanations, "explanations"),
    ("How good are the explanations? (Metrics)", A.page_metrics, "metrics"),
    ("Can we trust the metrics? (Reliability)", B.page_reliability, "reliability"),
    ("Why do methods disagree? (Disagreement)", B.page_disagreement, "disagreement"),
    ("Are the differences real? (Statistics)", B.page_statistics, "statistics"),
    ("Final recommendation (Ranking)", B.page_ranking, "ranking"),
    ("Output for Component 4", B.page_c4, "component4"),
    ("Glossary", B.page_glossary, "glossary"),
]


def main() -> None:
    runs = list_runs()
    with st.sidebar:
        st.markdown("### Component 3 · Comparing explanation methods")
        run = st.selectbox("Results run", runs, help="pilot = small check run; full = final experiment") if runs else None
        if run:
            ctxs = primary_contexts(run)
            if len(ctxs) > 1:
                st.session_state["ctx_id"] = st.selectbox("Prediction context", [c["context_id"] for c in ctxs])
        st.toggle("Presentation mode (larger text)", key="big_text")
    apply_style()

    def wrap(fn):
        def page():
            if run is None and fn is not B.page_glossary:
                st.title("Comparing explanation methods")
                st.info("No results yet — run the pilot with:\n\n`.venv/bin/python scripts/run_experiment.py --pilot`\n\n"
                        "(from the component3/ folder), then reload this page.")
                return
            try:
                fn(run)
            except Exception as e:  # friendly message instead of a stack trace
                if os.environ.get("XAICOMP_DASHBOARD_DEBUG"):
                    raise
                st.warning(f"This page could not be drawn from the current results ({type(e).__name__}). "
                           "Re-run the analysis: `.venv/bin/python scripts/run_experiment.py --pilot --stage analyse`.")
        page.__name__ = fn.__name__
        return page

    nav = st.navigation([st.Page(wrap(fn), title=t, url_path=u, default=(i == 0)) for i, (t, fn, u) in enumerate(PAGES)])
    nav.run()


main()
