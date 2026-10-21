"""Step 15 (vision): readiness check, run gate, screening, end-to-end toy run, export and cross-context.

The REAL Component 2 folder is only read (it is currently empty, so vision must be NOT ready). Toy
Component-2-shaped folders are written to component3/tests/_tmp/ (git-ignored) and deleted afterwards;
nothing toy is written to results/ or data/.
"""

import json
import shutil
import subprocess
import sys

import pytest
import yaml

from fixtures_vision import BINARY, toy_ctx, write_toy_c2
from xaicomp._bootstrap import C3_ROOT, REPO_ROOT
from xaicomp.config import load_context

TMP = C3_ROOT / "tests" / "_tmp"


@pytest.fixture
def tmp_c2():
    root = TMP / "c2"
    shutil.rmtree(TMP, ignore_errors=True)
    yield root
    shutil.rmtree(TMP, ignore_errors=True)


def _status(res, name):
    return next(c for c in res["checks"] if c["name"] == name)["status"]


# ------------------------------------------------------------------ readiness
def test_real_component2_is_not_ready_now():
    from xaicomp.run.vision_readiness import check_vision_ready, format_report
    res = check_vision_ready(load_context("vision_engagement_gru"), compute_accuracy=False)
    c2 = REPO_ROOT / "component2" / "vision_cbu"
    has_model = any((c2 / "models").glob("*/model.pt"))
    if has_model:
        pytest.skip("Component 2 has delivered a model; the not-ready check no longer applies")
    assert res["ready"] is False and _status(res, "trained_model") == "fail"
    txt = format_report(res)
    assert "Component 2 owner" in txt and "NOT READY" in txt


def test_check_script_exit_code_matches(tmp_c2):
    out = subprocess.run([sys.executable, "scripts/check_vision_ready.py", "--no-accuracy"], cwd=C3_ROOT,
                         capture_output=True, text=True)
    from xaicomp.run.vision_readiness import check_vision_ready
    ready = check_vision_ready(load_context("vision_engagement_gru"), compute_accuracy=False)["ready"]
    assert out.returncode == (0 if ready else 1), out.stdout + out.stderr


def test_toy_c2_folder_is_ready(tmp_c2):
    from xaicomp.run.vision_readiness import check_vision_ready
    write_toy_c2(tmp_c2)
    res = check_vision_ready(toy_ctx(), compute_accuracy=True)
    assert res["ready"], res
    assert _status(res, "subject_disjoint") == "pass"


def test_wrong_label_scheme_and_overlap(tmp_c2):
    from xaicomp.run.vision_readiness import check_vision_ready
    four = dict(BINARY, name="daisee_original", class_names=[f"engagement_{i}" for i in range(4)],
                mapping={str(i): i for i in range(4)})
    write_toy_c2(tmp_c2, scheme=four, overlap=True)
    res = check_vision_ready(toy_ctx(), compute_accuracy=False)
    assert not res["ready"] and _status(res, "trained_model") == "fail"
    assert "daisee_binary" in next(c for c in res["checks"] if c["name"] == "trained_model")["detail"]
    assert _status(res, "subject_disjoint") == "warn"          # overlap -> IN-SAMPLE, not a blocker


def test_missing_features_blocks(tmp_c2):
    from xaicomp.run.vision_readiness import check_vision_ready
    write_toy_c2(tmp_c2, with_features=False)
    res = check_vision_ready(toy_ctx(), compute_accuracy=False)
    assert not res["ready"] and _status(res, "features") == "fail"


def test_synthetic_rows_are_refused(tmp_c2):
    import pandas as pd
    from xaicomp.run.vision_readiness import check_vision_ready
    write_toy_c2(tmp_c2)
    f = sorted((tmp_c2 / "data" / "processed" / "features").glob("*.csv.gz"))[0]
    df = pd.read_csv(f)
    df["is_synthetic"], df["source"] = True, "synthetic"
    df.to_csv(f, index=False, compression="gzip")
    res = check_vision_ready(toy_ctx(), compute_accuracy=False)
    assert _status(res, "features") == "fail" and not res["ready"]


# ------------------------------------------------------------------ run gate
def test_run_experiment_refuses_vision_when_not_ready():
    from xaicomp.run.vision_readiness import check_vision_ready
    if check_vision_ready(load_context("vision_engagement_gru"), compute_accuracy=False)["ready"]:
        pytest.skip("vision is ready; the refusal path cannot be tested on the real folder")
    before = sorted(p.name for p in (C3_ROOT / "results").iterdir())
    out = subprocess.run([sys.executable, "scripts/run_experiment.py", "--vision", "--pilot"], cwd=C3_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 2 and "REFUSED" in out.stdout
    assert sorted(p.name for p in (C3_ROOT / "results").iterdir()) == before      # nothing written


def test_vision_contexts_need_the_flag():
    out = subprocess.run([sys.executable, "scripts/run_experiment.py", "--pilot", "--contexts", "vision_engagement_gru",
                          "--stage", "explain"], cwd=C3_ROOT, capture_output=True, text=True)
    assert out.returncode == 2 and "--vision" in out.stdout


# ------------------------------------------------------------------ screening
def test_vision_screening_gru_and_baseline():
    from xaicomp.run.analysis import context_capabilities
    from xaicomp.screening.applicability import screen

    budgets = {"kernel_shap": {"nsamples": 10, "k_donors": 1}, "conditional_shap": {"nsamples": 10, "k_donors": 1},
               "lime": {"num_samples": 10, "k_donors": 1}, "ig": {"n_steps": 4, "k_donors": 1}, "ig_zero": {"n_steps": 4}}
    gru = screen("v", context_capabilities(load_context("vision_engagement_gru")), budgets, 1.0, 600)
    v = dict(zip(gru["method"], gru["verdict"]))
    assert all(v[m] == "applicable" for m in ("kernel_shap", "conditional_shap", "lime", "ig", "ig_zero"))
    assert v["dice"] == "excluded" and "facial" in gru.set_index("method").loc["dice", "reasons"]
    base = screen("b", context_capabilities(load_context("vision_engagement_baseline")), budgets, 1.0, 600)
    b = dict(zip(base["method"], base["verdict"]))
    assert b["ig"] == "excluded" and b["ig_zero"] == "excluded" and b["kernel_shap"] == "applicable"


def test_eeg_screening_unchanged():
    from xaicomp.run.analysis import context_capabilities
    caps = context_capabilities(load_context("eeg_T1_A8"))
    assert caps.n_features == 84 and caps.differentiable and caps.notes == {}


# ------------------------------------------------------------------ end-to-end toy run (tests/_tmp only)
@pytest.mark.slow
def test_toy_vision_end_to_end(tmp_c2):
    from xaicomp.config import run_settings
    from xaicomp.export.component4 import build_export
    from xaicomp.run.analysis import analyse_run
    from xaicomp.run.orchestrator import run_context

    write_toy_c2(tmp_c2)
    ctx = toy_ctx(cache_windows=False, context_id="vision/toy")
    ctrl = toy_ctx(cache_windows=False, context_id="vision/toy__randomised", weights="randomised",
                   negative_control=True, control_of="vision/toy")
    p1, p2 = TMP / "toy.yaml", TMP / "toy_ctrl.yaml"
    p1.write_text(yaml.safe_dump(ctx)), p2.write_text(yaml.safe_dump(ctrl))
    settings = run_settings(True, "run_vision.yaml")
    settings["budget"]["subjects"] = "all"
    settings["budget"]["control"] = {"subjects_per_fold": 2, "instances_per_subject": 1}
    run_root = TMP / "run"
    _, sel = run_context(str(p1), settings, run_root, workers=1)
    run_context(str(p2), settings, run_root, workers=1, primary_selected=sel)
    info = json.loads((run_root / "vision__toy" / "run_info.json").read_text())
    assert info["modality"] == "vision" and info["space_kind"] == "vision" and info["in_sample"] is False
    assert info["conditional_check"]["mode"] in ("gaussian", "empirical")
    assert info["run_type"] == "PILOT"
    assert len(list((run_root / "vision__toy" / "metrics").glob("*.json"))) == len(sel)
    # the whole analysis + export, with the same code as EEG
    analyse_run(run_root, pilot=True, workers=1, log=lambda *a: None, run_file="run_vision.yaml")
    doc = build_export(run_root, save=False)
    assert doc["contexts"][0]["modality"] == "vision" and doc["windows"] == []
    assert len(doc["vision_windows"]) == len(sel)
    assert set(doc["vision_windows"][0]["top_k"]) >= {"group", "segment", "feature_x_segment"}
    assert doc["modalities"]["vision"]["status"] == "available"


def test_export_of_real_eeg_pilot_keeps_format():
    from xaicomp.export.component4 import build_export
    run = C3_ROOT / "results" / "pilot"
    if not (run / "eeg__T1__A8_cnn_lstm_trans" / "analysis" / "ranking.json").exists():
        pytest.skip("no EEG pilot results")
    doc = build_export(run, save=False)                   # nothing is written
    assert doc["schema_version"] == "1.0" and len(doc["windows"]) > 0 and "vision_windows" not in doc
    assert set(doc["windows"][0]["top_k"]) == {"region", "band", "channel_x_band"}
    assert doc["modalities"]["vision"]["status"] in ("not_available", "available_in_other_run")


# ------------------------------------------------------------------ cross-context
def test_cross_context_not_available_without_vision():
    from xaicomp.compare.cross_context import compare_contexts, find_context_dir
    eeg = find_context_dir(C3_ROOT / "results" / "pilot", "eeg")
    doc = compare_contexts(eeg, None)
    assert doc["available"] is False and any("vision" in m for m in doc["missing"])
    assert "does not show that one modality explains workload" in doc["framing"]


def test_cross_context_compares_two_finished_contexts():
    from xaicomp.compare.cross_context import compare_contexts, find_context_dir
    eeg = find_context_dir(C3_ROOT / "results" / "pilot", "eeg")
    if eeg is None or not (eeg / "analysis" / "ranking.json").exists():
        pytest.skip("no EEG pilot results")
    doc = compare_contexts(eeg, eeg)                       # shape check only: same context twice
    assert doc["available"] and doc["same_recommendation"] is True
    assert doc["ranking_agreement"]["kendall_tau_expected_rank"] == pytest.approx(1.0)
    assert {r["criterion"] for r in doc["criteria_reliability"]}
