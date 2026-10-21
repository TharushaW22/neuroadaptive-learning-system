import json
from pathlib import Path

import numpy as np
import pytest

from src.data.feature_schema import FEATURE_COLUMNS
from src.data.synthetic import generate_synthetic_features, write_synthetic
from src.models.inference import NO_MODEL, UNKNOWN, Predictor, find_validated_model, load_live_predictor
from src.models.registry import ModelCompatibilityError, load_model, read_metadata, validate_metadata
from src.models.train import train_demo
from src.utils.config import resolve_path


@pytest.fixture
def demo_models(cfg):
    write_synthetic(generate_synthetic_features(12, 2, 8, 5), resolve_path(cfg, "demo_dir"))
    return {k: train_demo(cfg, k) for k in ("baseline", "temporal")}


def test_metadata_contains_required_fields(demo_models):
    for meta in demo_models.values():
        m = read_metadata(Path(meta["path"]))
        assert m["demo_only"] and m["data_source"] == "synthetic" and m["model_name"].startswith("DEMO_ONLY_")
        for key in ("label_scheme", "feature_schema_version", "feature_columns", "training_data",
                    "normalizer", "model_params", "window", "seed"):
            assert key in m


def test_demo_models_refused_by_default(demo_models):
    for meta in demo_models.values():
        with pytest.raises(ModelCompatibilityError):
            load_model(Path(meta["path"]))
        load_model(Path(meta["path"]), allow_demo=True)


def test_incompatible_features_and_labels_refused(demo_models):
    m = read_metadata(Path(demo_models["baseline"]["path"]))
    with pytest.raises(ModelCompatibilityError):
        validate_metadata(m, expected_features=FEATURE_COLUMNS[:-1], allow_demo=True)
    with pytest.raises(ModelCompatibilityError):
        validate_metadata(m, allow_demo=True, expected_classes=["a", "b"])
    with pytest.raises(ModelCompatibilityError):
        validate_metadata({**m, "feature_schema_version": "0.1"}, allow_demo=True)
    with pytest.raises(ModelCompatibilityError):
        validate_metadata({k: v for k, v in m.items() if k != "normalizer"}, allow_demo=True)


def test_no_validated_model_without_real_eval(cfg, demo_models):
    assert find_validated_model(cfg) is None
    pred, msg = load_live_predictor(cfg)
    assert pred is None and msg == NO_MODEL


@pytest.mark.parametrize("kind", ["baseline", "temporal"])
def test_predictor_unknown_and_demo_flag(cfg, demo_models, kind):
    p = Predictor.from_dir(Path(demo_models[kind]["path"]), cfg, allow_demo=True)
    df = generate_synthetic_features(1, 1, 10, 5, missing_rate=0.0)
    assert p.predict_frames(df.head(5))["state"] == UNKNOWN                     # too few samples
    bad = df.copy()
    bad["quality_ok"] = False
    bad[FEATURE_COLUMNS] = np.nan
    r = p.predict_frames(bad)
    assert r["state"] == UNKNOWN and "quality" in r["reason"]                   # missing data -> UNKNOWN
    ok = p.predict_frames(df)
    assert "DEMO ONLY" in ok["state"] and ok["demo_only"]
    assert sum(ok["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    partial = df.copy()
    partial.loc[partial.index[-3:], FEATURE_COLUMNS] = np.nan                   # a few missing frames
    partial.loc[partial.index[-3:], "quality_ok"] = False
    assert "DEMO ONLY" in p.predict_frames(partial)["state"]
    assert p.predict_frames(df.drop(columns=["ear_mean"]))["state"] == UNKNOWN


def test_validated_model_requires_real_evaluation_report(cfg, demo_models):
    # Pretend a REAL model existed: only a real eval report naming it may validate it.
    src = Path(demo_models["baseline"]["path"])
    dst = resolve_path(cfg, "models_dir") / "real_baseline_x"
    dst.mkdir(parents=True)
    meta = read_metadata(src)
    meta.update(model_name="real_baseline_x", data_source="real", demo_only=False)
    (dst / "metadata.json").write_text(json.dumps(meta))
    (dst / "model.joblib").write_bytes((src / "model.joblib").read_bytes())
    assert find_validated_model(cfg) is None
    ev = resolve_path(cfg, "evaluation_dir") / "20260101_000000"
    ev.mkdir(parents=True)
    (ev / "report.json").write_text(json.dumps({"data_source": "real", "models": [{"model_name": "real_baseline_x"}]}))
    assert find_validated_model(cfg)["model_name"] == "real_baseline_x"
    pred, _ = load_live_predictor(cfg)
    assert pred is not None and not pred.meta["demo_only"]
