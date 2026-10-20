"""Step 14: every dashboard page renders from real results without an exception (headless AppTest)."""

import pytest

from xaicomp._bootstrap import C3_ROOT

PAGES = ["page_overview", "page_model", "page_screening", "page_explanations", "page_metrics", "page_reliability",
         "page_disagreement", "page_statistics", "page_ranking", "page_c4", "page_glossary"]


def _run_name():
    import sys
    sys.path.insert(0, str(C3_ROOT / "dashboard"))
    from common import list_runs
    runs = list_runs()
    return runs[0] if runs else None


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    run = _run_name()
    if run is None and page != "page_glossary":
        pytest.skip("no results run yet")
    mod = "views_b" if page in PAGES[5:] else "views_a"
    code = f"""
import sys
sys.path.insert(0, {str(C3_ROOT / 'dashboard')!r})
import streamlit as st
from common import apply_style
import {mod} as V
apply_style()
V.{page}({run!r})
"""
    at = AppTest.from_string(code, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) >= 1


def test_no_results_message_is_friendly(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    code = f"""
import sys
sys.path.insert(0, {str(C3_ROOT / 'dashboard')!r})
from common import no_results
no_results()
"""
    at = AppTest.from_string(code).run()
    assert "run_experiment.py --pilot" in at.info[0].value
