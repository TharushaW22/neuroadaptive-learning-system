"""Dashboard tests via Streamlit's AppTest (headless; no browser, no webcam)."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
PAGES = ["Home & status", "Live webcam", "Feature explorer", "Dataset setup & audit", "Model training",
         "Model evaluation", "Session analytics", "Settings & privacy"]

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_has_no_bare_expressions():
    """Streamlit 'magic' renders bare expressions; on some versions this crashes the live loop
    ("'{' was never closed"). app.py must only contain explicit calls."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Expr) and not isinstance(n.value, (ast.Call, ast.Constant))]
    assert bad == [], f"bare expressions on lines {bad}"


def test_streamlit_magic_disabled():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "magicEnabled = false" in cfg


@pytest.fixture
def app(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("VCBU_PSEUDONYM_SALT", "test-salt")       # no salt file written
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_mock_capture_runs_to_completion(app):
    app.sidebar.radio[0].set_value("Live webcam").run()
    app.selectbox[0].set_value("Mock camera (no hardware)")
    app.slider[0].set_value(5)
    app.button[0].click().run()
    assert not app.exception
    assert not [e.value for e in app.error if e.value.startswith(("Capture stopped", "Camera unavailable"))]
    assert any(s.value.startswith("Captured") for s in app.success)
    rows = app.session_state["live_rows"]
    assert len(rows) > 20 and sum(r["blink_event"] == 1.0 for r in rows) >= 1
    assert any("data:image/jpeg;base64" in m.value for m in app.markdown)     # frame displayed inline
    app.sidebar.radio[0].set_value("Session analytics").run()
    assert not app.exception


def test_heuristic_scores_in_session_and_exports(app):
    """Every captured row carries the heuristic fields; Session analytics shows the disclaimer."""
    from src.models.interpretation import DISCLAIMER, INTERPRETATION_COLUMNS
    app.sidebar.radio[0].set_value("Live webcam").run()
    app.selectbox[0].set_value("Mock camera (no hardware)")
    app.slider[0].set_value(5)
    if not app.checkbox[0].disabled:                       # DEMO ONLY model, if one exists locally
        app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    rows = app.session_state["live_rows"]
    for r in rows:
        assert all(k in r for k in INTERPRETATION_COLUMNS)
        assert r["interpretation_source"] == "heuristic_v1"
        for k in ("attention_score", "fatigue_score", "readiness_score"):
            assert r[k] is None or 0.0 <= r[k] <= 1.0
            if r["quality_state"] != "OK":
                assert r[k] is None
    app.sidebar.radio[0].set_value("Session analytics").run()
    assert not app.exception
    assert any(DISCLAIMER in w.value for w in app.warning)
