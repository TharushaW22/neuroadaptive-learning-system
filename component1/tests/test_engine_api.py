"""Phase 6 tests: inference engine, output contract, heuristics, and the FastAPI app.

A tiny classical model (logistic regression on spectral features of 6 subjects) is trained into a
temporary models/ folder, so these tests do not depend on the long training runs.
"""

import json

import joblib
import numpy as np
import pytest

from eegengine.config import load_config
from eegengine.data import load_features
from eegengine.features.extract import extract_features, family_columns
from eegengine.inference import predictors
from eegengine.inference.heuristics import heuristic_indices
from eegengine.inference.schema import CognitiveStateOutput
from eegengine.io.stew import CHANNELS, DEFAULT_STEW_DIR, FS, load_recording
from eegengine.labels import TASKS, task_view
from eegengine.models.classical import fit_pipeline, make_pipeline
from eegengine.preprocess.filters import filter_recording
from eegengine.training.classical_runner import FoldModel

needs_data = pytest.mark.skipif(not (DEFAULT_STEW_DIR / "ratings.txt").exists(), reason="no STEW data")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    try:
        F, meta, _ = load_features()
    except FileNotFoundError:
        pytest.skip("features not generated (run scripts/run_preprocess.py + run_features.py)")
    keep = meta["subject_id"].isin([1, 2, 3, 4, 6, 7]).to_numpy()
    mask, y = task_view(meta[keep].reset_index(drop=True), "workload_binary")
    cols = family_columns(F.columns, ["spectral"])
    X = F[keep].reset_index(drop=True)[cols].to_numpy(np.float32)[mask]
    fm = FoldModel(fit_pipeline(make_pipeline("lr"), X, y), np.arange(2), 2)
    info = {"version": "test-lr", "task": "workload_binary", "classes": TASKS["workload_binary"]["classes"],
            "model": "lr", "feature_set": "A1_spectral"}
    joblib.dump({**info, "estimator": fm, "columns": cols}, d / "T1_test.joblib")
    (d / "T1_test.json").write_text(json.dumps(info))
    old = predictors.MODELS_DIR
    predictors.MODELS_DIR = d
    yield "T1_test"
    predictors.MODELS_DIR = old


# ----------------------------------------------------------------- heuristics
def test_heuristic_formulas():
    bp = np.ones((14, 5))
    bp[:, 3] = 2.0                                     # beta = 2, alpha = theta = 1
    h = heuristic_indices(bp, CHANNELS)
    assert h["attention"] == pytest.approx(2 / (1 + 1))
    assert h["fatigue"] == pytest.approx((1 + 1) / 2)
    assert h["readiness"] == pytest.approx(0.5 * 1 / 2 + 0.5 * 1 / 2)


# ----------------------------------------------------------------- engine
@needs_data
def test_engine_full_recording(tiny_model):
    from eegengine.inference.engine import CognitiveEngine
    eng = CognitiveEngine(tiny_model, participant_id="sub01", session_id="t")
    outs = eng.run_recording(load_recording(1, "hi").data)
    assert len(outs) == 74                              # same epoch grid as training
    o = outs[10]
    assert isinstance(o, CognitiveStateOutput)
    assert o.window_index == 10 and o.t_start_s == 20.0 and o.t_end_s == 24.0
    assert abs(sum(o.states.workload.probabilities.values()) - 1) < 1e-6
    assert o.states.workload.source == "model"
    for k in ("attention", "fatigue", "readiness"):
        assert getattr(o.states, k).source == "heuristic_index"
        assert "not a trained prediction" in getattr(o.states, k).note
    assert o.signal_quality.flag in ("good", "warn", "bad")
    assert len(o.connectivity.matrix) == 14 and o.connectivity.matrix[0][0] == pytest.approx(1.0, abs=1e-3)
    assert set(o.band_power) == set(CHANNELS)
    json.dumps(o.model_dump())                          # JSON-serialisable


@needs_data
def test_engine_streaming_matches_batch(tiny_model):
    """Pushing small chunks gives exactly the same outputs as one big push."""
    from eegengine.inference.engine import CognitiveEngine
    data = load_recording(2, "lo").data[:, : 30 * FS]
    a = CognitiveEngine(tiny_model, session_id="x").run_recording(data)
    eng = CognitiveEngine(tiny_model, session_id="x")
    b = []
    for s in range(0, data.shape[1], 37):              # odd chunk size on purpose
        b += eng.push(data[:, s:s + 37])
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.states.workload.raw_probabilities == pytest.approx(y.states.workload.raw_probabilities)


@needs_data
def test_engine_band_power_close_to_offline():
    """Online (20-s context) filtering vs offline whole-recording filtering: band powers agree."""
    cfg = load_config("preprocess.yaml")
    rec = load_recording(3, "hi")
    off = filter_recording(rec.data, FS, cfg["filter"])
    end = 60 * FS
    on = filter_recording(rec.data[:, end - 24 * FS:end], FS, cfg["filter"])[:, -512:]
    fcfg = load_config("features.yaml")
    _, a = extract_features(off[None, :, end - 512:end], FS, CHANNELS, {**fcfg, "families": ["spectral"]})
    _, b = extract_features(on[None], FS, CHANNELS, {**fcfg, "families": ["spectral"]})
    ratio = b["bandpower"][0, :, 1:] / a["bandpower"][0, :, 1:]      # theta..gamma
    assert np.all(np.abs(ratio - 1) < 0.05)


# ----------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(tiny_model):
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def bare_client():
    """API client that needs neither the raw data nor the generated features."""
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_health(bare_client):
    r = bare_client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@needs_data
def test_subjects(client):
    s = client.get("/subjects").json()
    assert len(s) == 48
    assert not next(x for x in s if x["subject_id"] == 5)["rated"]


def test_models(client):
    assert any(m["id"] == "T1_test" for m in client.get("/models").json())


def test_schema(bare_client):
    s = bare_client.get("/schema").json()
    assert "states" in s["properties"] and "signal_quality" in s["properties"]


@needs_data
def test_quality(client):
    q = client.get("/quality").json()
    assert len(q) == 96
    w = client.get("/quality/1/lo").json()
    assert len(w) == 74 and w[0]["flag"] in ("good", "warn", "bad")
    assert client.get("/quality/99/lo").status_code == 404


@needs_data
def test_predict_upload(client):
    data = load_recording(1, "hi").data[:, : 8 * FS]
    body = json.dumps({"data": data.tolist()}).encode()
    r = client.post("/predict", files={"file": ("w.json", body, "application/json")}, data={"model": "T1_test"})
    assert r.status_code == 200
    outs = r.json()
    assert len(outs) == 3                               # windows ending at 4, 6, 8 s
    CognitiveStateOutput.model_validate(outs[0])


def test_predict_rejects_bad_shape(client):
    body = json.dumps({"data": [[0.0] * 600] * 3}).encode()
    r = client.post("/predict", files={"file": ("w.json", body, "application/json")}, data={"model": "T1_test"})
    assert r.status_code == 400


@needs_data
def test_stream_websocket(client):
    with client.websocket_connect("/stream/1/hi?model=T1_test&speed=50") as ws:
        meta = ws.receive_json()
        assert meta["type"] == "meta" and meta["classes"] == ["low", "high"]
        raw = ws.receive_json()
        assert raw["type"] == "raw" and len(raw["data"]) == 14 and len(raw["data"][0]) == 256  # 4 s at 64 Hz
        state = ws.receive_json()
        assert state["type"] == "state"
        CognitiveStateOutput.model_validate(state["output"])


def test_missing_data_is_503_with_cors(bare_client, monkeypatch):
    """A missing file must not become a bare 500: those carry no CORS header and the browser
    misreports them as a CORS error. It must be a 503 with a hint, and keep the CORS header."""
    import api.main as m

    def boom():
        raise FileNotFoundError(2, "No such file", "data/raw/STEW/ratings.txt")
    monkeypatch.setattr(m, "subject_ratings", boom)
    r = bare_client.get("/subjects", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 503
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "hint" in r.json()


def test_subjects_fallback_without_raw_data(bare_client, monkeypatch):
    """/subjects still works from results/eda/recordings.csv when data/raw/STEW is absent."""
    import api.main as m

    def missing(*_a, **_k):
        raise FileNotFoundError("ratings.txt")
    monkeypatch.setattr(m, "load_ratings", missing)
    s = bare_client.get("/subjects").json()
    assert len(s) == 48
    assert next(x for x in s if x["subject_id"] == 1)["rating_hi"] == 8.0
    assert not next(x for x in s if x["subject_id"] == 42)["rated"]


def test_unknown_result_404(bare_client):
    assert bare_client.get("/results/does_not_exist").status_code == 404
