"""Step 15 (vision) dashboard: navigation sections, vision empty states, Multimodal / Cross-context / C4 /
Glossary pages, and that every vision page renders once vision results exist.

The "results exist" case uses a toy run written to component3/tests/_tmp/ (git-ignored, deleted after the
module); the page code is pointed at that folder inside the test only. The real dashboard reads results/
and never sees toy data.
"""

import shutil

import pytest
import yaml

from xaicomp._bootstrap import C3_ROOT

DASH = str(C3_ROOT / "dashboard")
TMP = C3_ROOT / "tests" / "_tmp_dash"
WAITING = "Waiting for Component 2's trained model — run scripts/check_vision_ready.py to see what is missing."


def run_code(body: str, timeout: int = 120):
    from streamlit.testing.v1 import AppTest
    code = f"import sys\nsys.path.insert(0, {DASH!r})\nimport streamlit as st\nfrom common import apply_style\napply_style()\n{body}"
    at = AppTest.from_string(code, default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def eeg_run():
    import sys
    sys.path.insert(0, DASH)
    from common import list_runs
    runs = list_runs("eeg")
    return runs[0] if runs else None


def test_full_app_renders_with_sections():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C3_ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("Comparing explanation methods")


@pytest.mark.parametrize("page", ["page_vision_overview", "page_vision_model", "page_vision_explanations"])
def test_vision_pages_empty_state(page):
    at = run_code(f"from common import set_modality\nset_modality('vision')\nimport views_vision as VV\nVV.{page}(None)")
    assert at.info[0].value == WAITING


def test_reused_page_empty_state_wording():
    at = run_code("from common import set_modality\nset_modality('vision')\nimport views_vision as VV\nVV.vision_empty()")
    assert at.info[0].value == WAITING


def test_multimodal_page_explains_missing_pairing():
    at = run_code("import views_vision as VV\nVV.page_multimodal(None)")
    text = " ".join(m.value for m in at.markdown)
    assert "different people" in text and "same learners" in text.lower()


def test_cross_context_page_without_vision_explains():
    at = run_code(f"import views_vision as VV\nVV.page_cross_context({eeg_run()!r}, None)")
    assert at.info[0].value == WAITING
    assert "does not show that one modality explains workload" in " ".join(m.value for m in at.markdown)


def test_project_overview_and_c4_pages():
    run_code(f"import views_vision as VV\nVV.page_project_overview({eeg_run()!r}, None)")
    at = run_code(f"import views_vision as VV\nVV.page_c4_all({eeg_run()!r}, None)")
    assert any(i.value == WAITING for i in at.info)


def test_glossary_has_vision_terms():
    at = run_code("import views_b as B\nB.page_glossary(None)")
    text = " ".join(m.value for m in at.markdown)
    for term in ("Blink ratio", "Gaze", "Head pose", "Time segment", "Prediction context", "Subject-disjoint"):
        assert term in text, term


def test_eeg_wording_unchanged():
    import sys
    sys.path.insert(0, DASH)
    import common
    common.set_modality("eeg")
    assert common.V("screen_model") == "the EEG deep model"
    assert common.criteria_plain() is common.CRITERIA_PLAIN
    assert common.V("pilot_cmd") == common.PILOT_CMD


# ------------------------------------------------------------------ vision pages with (toy) results
@pytest.fixture(scope="module")
def toy_results():
    from fixtures_vision import toy_ctx, write_toy_c2
    from xaicomp.config import run_settings
    from xaicomp.run.analysis import analyse_run
    from xaicomp.run.orchestrator import run_context

    shutil.rmtree(TMP, ignore_errors=True)
    c2 = TMP / "c2"
    write_toy_c2(c2)
    rel = "tests/_tmp_dash/c2"
    paths = {"root": rel, "models_dir": f"{rel}/models", "features_dir": f"{rel}/data/processed/features",
             "config": f"{rel}/cfg.yaml", "model_name": "auto"}
    ctx = toy_ctx(cache_windows=False, context_id="vision/toy", component2=paths)
    ctrl = toy_ctx(cache_windows=False, context_id="vision/toy__randomised", weights="randomised",
                   negative_control=True, control_of="vision/toy", component2=paths)
    (TMP / "toy.yaml").write_text(yaml.safe_dump(ctx))
    (TMP / "toy_ctrl.yaml").write_text(yaml.safe_dump(ctrl))
    settings = run_settings(True, "run_vision.yaml")
    settings["budget"]["subjects"] = "all"
    settings["budget"]["control"] = {"subjects_per_fold": 2, "instances_per_subject": 1}
    run_root = TMP / "results" / "vision_toy"
    _, sel = run_context(str(TMP / "toy.yaml"), settings, run_root, workers=1)
    run_context(str(TMP / "toy_ctrl.yaml"), settings, run_root, workers=1, primary_selected=sel)
    analyse_run(run_root, pilot=True, workers=1, log=lambda *a: None, run_file="run_vision.yaml")
    yield TMP / "results"
    shutil.rmtree(TMP, ignore_errors=True)


PATCH = """
from pathlib import Path
import common, views_b, views_vision
for m in (common, views_b, views_vision):
    m.RESULTS = Path({root!r})
common.set_modality('vision')
"""


@pytest.mark.slow
@pytest.mark.parametrize("call", ["views_vision.page_vision_overview", "views_vision.page_vision_model",
                                  "views_vision.page_vision_explanations", "views_a.page_screening",
                                  "views_a.page_metrics", "views_b.page_reliability", "views_b.page_disagreement",
                                  "views_b.page_statistics", "views_b.page_ranking"])
def test_vision_pages_render_with_results(toy_results, call):
    mod, fn = call.split(".")
    at = run_code(PATCH.format(root=str(toy_results)) + f"import {mod}\n{mod}.{fn}('vision_toy')", timeout=180)
    assert len(at.title) >= 1
    assert not any(WAITING == i.value for i in at.info)


@pytest.mark.slow
def test_c4_and_cross_context_with_vision_results(toy_results):
    run_code(PATCH.format(root=str(toy_results)) + "views_vision.page_c4_all(None, 'vision_toy')")
    run_code(PATCH.format(root=str(toy_results)) + "views_vision.page_cross_context(None, 'vision_toy')")
