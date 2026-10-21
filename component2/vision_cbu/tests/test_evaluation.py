import json
from pathlib import Path

import numpy as np
import pytest

from src.data.extract_features import load_feature_table, run_extraction
from src.data.synthetic import generate_synthetic_features, write_synthetic
from src.data.validate_dataset import RealDataUnavailableError, get_clip_manifest
from src.evaluation.metrics import classification_metrics, measure_latency, video_level
from src.evaluation.reports import DEMO_BANNER, list_reports, run_evaluation
from src.models.train import train_demo, train_real
from src.utils.config import resolve_path
from src.vision.mock_landmarks import MockLandmarkDetector


def test_metrics_known_values():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class"][0]["precision"] == 1.0 and m["per_class"][0]["recall"] == 0.5
    assert m["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["majority_class_accuracy"] == 0.5


def test_metrics_missing_class_and_empty():
    m = classification_metrics([0, 0], [0, 0], ["a", "b", "c"])
    assert m["per_class"][2]["support"] == 0 and len(m["confusion_matrix"]) == 3
    with pytest.raises(ValueError):
        classification_metrics([], [], ["a"])


def test_video_level_and_latency():
    yt, yp = video_level([1, 1, 0], np.array([[0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]), ["v1", "v1", "v2"])
    assert list(yt) == [1, 0] and list(yp) == [0, 0]   # v1 mean probs [0.55, 0.45]
    lat = measure_latency(lambda x: x * 2, 3, repeats=5)
    assert lat["repeats"] == 5 and lat["mean_ms"] >= 0


def test_real_evaluation_refuses_without_data(cfg):
    with pytest.raises(RealDataUnavailableError):
        run_evaluation(cfg)
    with pytest.raises(RealDataUnavailableError):
        train_real(cfg, "baseline")


def test_demo_evaluation_is_labelled_and_kept_out_of_real_reports(cfg):
    write_synthetic(generate_synthetic_features(12, 2, 8, 5), resolve_path(cfg, "demo_dir"))
    for k in ("baseline", "temporal"):
        train_demo(cfg, k)
    rep = run_evaluation(cfg, demo=True)
    out = Path(rep["output_dir"])
    assert rep["title"] == DEMO_BANNER and rep["data_source"] == "synthetic"
    assert DEMO_BANNER in (out / "report.md").read_text()
    for f in ("report.json", "metrics_summary.csv", "per_class_metrics.csv", "window_predictions.csv",
              "ablation.csv", "confusion_baseline.png", "confusion_temporal.png", "model_comparison.png"):
        assert (out / f).exists(), f
    assert list_reports(cfg) == []                          # nothing under outputs/evaluation
    assert len(rep["ablation"]) == 1 + 4


def test_full_research_pipeline_on_fake_daisee(cfg, fake_daisee):
    """Audit -> extract (mock landmarks, generated fixture videos) -> train -> evaluate."""
    cfg["sequences"].update(window_size=4, stride=2)
    cfg["extraction"]["target_fps"] = 5
    manifest = get_clip_manifest(cfg)
    idx = run_extraction(cfg, manifest, detector=MockLandmarkDetector())
    assert (idx["status"] == "extracted").all()
    idx2 = run_extraction(cfg, manifest, detector=MockLandmarkDetector())
    assert (idx2["status"] == "cached").all()               # caching works
    feats = load_feature_table(cfg)
    assert not feats["is_synthetic"].astype(str).str.lower().eq("true").any()
    assert set(feats["source"]) == {"daisee"}
    train_real(cfg, "baseline")
    train_real(cfg, "temporal")
    rep = run_evaluation(cfg)
    assert rep["data_source"] == "real" and rep["dataset"] == "DAiSEE"
    assert rep["split"]["participants"] == {"test": 2, "train": 4, "validation": 1}
    saved = json.loads((Path(rep["output_dir"]) / "report.json").read_text())
    assert {m["model_kind"] for m in saved["models"]} == {"baseline", "temporal"}


def test_real_training_rejects_synthetic_rows(cfg):
    feats = resolve_path(cfg, "features_dir")
    generate_synthetic_features(6, 1, 6, 5).to_csv(feats / "SYN.csv.gz", index=False, compression="gzip")
    with pytest.raises(RealDataUnavailableError):
        train_real(cfg, "baseline")
