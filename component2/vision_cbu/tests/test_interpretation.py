"""Heuristic attention / fatigue / readiness scores (EXPERIMENTAL, heuristic_v1)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.models.interpretation import (DISCLAIMER, QUALITY_OK, QUALITY_UNKNOWN, SCORE_COLUMNS, compute_proxies,
                                       compute_scores, engagement_level_from_prediction, export_metadata,
                                       format_score, interpret_window, interpretation_config, summarize_scores)
from src.utils.privacy import to_csv_bytes, to_json_bytes


def window(n=20, ear=0.30, rate=10.0, yaw_jitter=0.0, quality=True, face=True):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp_s": np.arange(n) * 0.2, "face_detected": face, "quality_ok": quality,
        "ear_mean": ear, "blink_rate_per_min": rate,
        "head_yaw": rng.normal(0, yaw_jitter, n) if yaw_jitter else 0.0, "head_pitch": 0.0, "head_roll": 0.0})


def pred(level, reason="ok"):
    probs = {f"engagement_{i}": 0.1 for i in range(4)}
    probs[f"engagement_{level}"] = 0.7
    return {"state": f"engagement_{level}", "reason": reason, "probabilities": probs}


def test_formulas_match_spec_exactly(cfg):
    icfg = interpretation_config(cfg)
    prox = {"face_vis": 0.9, "ear_open": 0.6, "blink_high": 0.2, "pose_stable": 0.8, "quality": 1.0}
    s = compute_scores(2, prox, icfg)
    e = 2 / 3
    att = 0.55 * e + 0.20 * 0.9 + 0.15 * 0.8 + 0.10 * 0.6
    fat = 0.40 * (1 - e) + 0.30 * 0.2 + 0.20 * (1 - 0.6) + 0.10 * (1 - 0.8)
    rdy = 0.50 * e + 0.30 * att + 0.20 * (1 - fat)
    assert s["attention_score"] == pytest.approx(att)
    assert s["fatigue_score"] == pytest.approx(fat)
    assert s["readiness_score"] == pytest.approx(rdy)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_scores_are_floats_in_unit_interval(cfg, level, p):
    s = compute_scores(level, {"face_vis": p, "ear_open": p, "blink_high": p, "pose_stable": p}, interpretation_config(cfg))
    for v in s.values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0


def test_extreme_cases_and_clipping(cfg):
    icfg = interpretation_config(cfg)
    best = compute_scores(3, {"face_vis": 1, "ear_open": 1, "blink_high": 0, "pose_stable": 1}, icfg)
    worst = compute_scores(0, {"face_vis": 0, "ear_open": 0, "blink_high": 1, "pose_stable": 0}, icfg)
    assert best == {"attention_score": pytest.approx(1.0), "fatigue_score": pytest.approx(0.0),
                    "readiness_score": pytest.approx(1.0)}
    assert worst == {"attention_score": pytest.approx(0.0), "fatigue_score": pytest.approx(1.0),
                     "readiness_score": pytest.approx(0.0)}
    cfg["interpretation"]["attention_weights"] = {"engagement": 2.0, "face_vis": 2.0, "pose_stable": 0, "ear_open": 0}
    assert compute_scores(3, {"face_vis": 1, "ear_open": 1, "blink_high": 0, "pose_stable": 1},
                          interpretation_config(cfg))["attention_score"] == 1.0


def test_weights_are_configurable_and_validated(cfg):
    cfg["interpretation"]["attention_weights"] = {"engagement": 1.0, "face_vis": 0, "pose_stable": 0, "ear_open": 0}
    s = compute_scores(1, {"face_vis": 1, "ear_open": 1, "blink_high": 0, "pose_stable": 1}, interpretation_config(cfg))
    assert s["attention_score"] == pytest.approx(1 / 3)
    cfg["interpretation"]["attention_weights"]["engagement"] = -0.1
    with pytest.raises(ValueError):
        interpretation_config(cfg)
    cfg["interpretation"]["attention_weights"] = {"typo": 0.5}
    with pytest.raises(ValueError):
        interpretation_config(cfg)


def test_default_config_matches_spec(cfg):
    icfg = interpretation_config(cfg)
    assert icfg["version"] == "heuristic_v1" and icfg["enabled"]
    assert icfg["attention_weights"] == {"engagement": 0.55, "face_vis": 0.20, "pose_stable": 0.15, "ear_open": 0.10}
    assert icfg["fatigue_weights"] == {"low_engagement": 0.40, "blink_high": 0.30, "eyes_closed": 0.20,
                                       "pose_unstable": 0.10}
    assert icfg["readiness_weights"] == {"engagement": 0.50, "attention": 0.30, "low_fatigue": 0.20}


def test_proxies(cfg):
    icfg = interpretation_config(cfg)
    calm = compute_proxies(window(ear=0.30, rate=10), icfg)
    assert calm == pytest.approx({"face_vis": 1.0, "ear_open": 1.0, "blink_high": 0.0, "pose_stable": 1.0,
                                  "quality": 1.0})
    tired = compute_proxies(window(ear=0.15, rate=40, yaw_jitter=30), icfg)
    assert tired["ear_open"] == 0.0 and tired["blink_high"] == 1.0 and tired["pose_stable"] < 0.5
    bad = compute_proxies(window(quality=False, face=False), icfg)
    assert bad["quality"] == 0.0 and bad["face_vis"] == 0.0 and math.isnan(bad["ear_open"])


def test_interpret_window_ok(cfg):
    out = interpret_window(pred(3), window(), cfg)
    assert out["quality_state"] == QUALITY_OK and out["engagement_level"] == 3
    assert out["interpretation_source"] == "heuristic_v1"
    for col in SCORE_COLUMNS:
        assert 0.0 <= out[col] <= 1.0


def test_unknown_quality_suppresses_scores(cfg):
    for w in (window(quality=False), window(quality=False, face=False)):
        out = interpret_window(pred(3), w, cfg)
        assert out["quality_state"] == QUALITY_UNKNOWN
        assert all(out[c] is None for c in SCORE_COLUMNS)
    # the model itself returned UNKNOWN (e.g. too few samples) -> no level, no scores
    out = interpret_window({"state": "UNKNOWN", "reason": "insufficient data: 3/20", "probabilities": None},
                           window(), cfg)
    assert out["quality_state"] == QUALITY_UNKNOWN and out["engagement_level"] is None
    assert all(out[c] is None for c in SCORE_COLUMNS)


def test_no_model_or_non_engagement_scheme_gives_no_scores(cfg):
    out = interpret_window(None, window(), cfg)
    assert out["engagement_level"] is None and all(out[c] is None for c in SCORE_COLUMNS)
    binary = {"state": "engagement_high(>=2)", "reason": "ok",
              "probabilities": {"engagement_low(<2)": 0.2, "engagement_high(>=2)": 0.8}}
    assert engagement_level_from_prediction(binary) is None
    assert all(interpret_window(binary, window(), cfg)[c] is None for c in SCORE_COLUMNS)


def test_disabled_gives_no_scores(cfg):
    cfg["interpretation"]["enabled"] = False
    out = interpret_window(pred(2), window(), cfg)
    assert out["interpretation_source"] == "disabled" and all(out[c] is None for c in SCORE_COLUMNS)


def test_format_and_summary():
    assert format_score(0.1) == "0.10" and format_score(0.4) == "0.40" and format_score(0.72, percent=True) == "72%"
    assert format_score(None) == "UNKNOWN" and format_score(float("nan")) == "UNKNOWN"
    df = pd.DataFrame({"attention_score": [0.1, 0.5, 0.9, None], "fatigue_score": [0.0, 1 / 3, 1.0, None],
                       "readiness_score": [None] * 4})
    s = summarize_scores(df).set_index("score")
    assert s.loc["attention", "n_valid"] == 3 and s.loc["attention", "n_unknown"] == 1
    assert s.loc["attention", "mean"] == 0.5 and s.loc["attention", "median"] == 0.5
    assert [s.loc["attention", b] for b in ("0.00-0.33", "0.33-0.66", "0.66-1.00")] == [1, 1, 1]
    assert [s.loc["fatigue", b] for b in ("0.00-0.33", "0.33-0.66", "0.66-1.00")] == [1, 1, 1]
    assert s.loc["readiness", "n_valid"] == 0 and pd.isna(s.loc["readiness", "mean"])


def test_exports_contain_scores_as_floats_or_empty(cfg):
    rows = [{**interpret_window(pred(2), window(), cfg), "timestamp_s": 0.0},
            {**interpret_window(pred(2), window(quality=False), cfg), "timestamp_s": 0.2}]
    df = pd.DataFrame(rows)
    csv = pd.read_csv(pd.io.common.BytesIO(to_csv_bytes(df)))
    assert 0 <= csv.loc[0, "attention_score"] <= 1 and pd.isna(csv.loc[1, "attention_score"])
    assert list(csv["quality_state"]) == ["OK", "UNKNOWN"]
    assert set(csv["interpretation_source"]) == {"heuristic_v1"}
    payload = json.loads(to_json_bytes(df, synthetic=False, interpretation=export_metadata(cfg)))
    assert isinstance(payload["data"][0]["fatigue_score"], float) and payload["data"][1]["fatigue_score"] is None
    assert payload["heuristic_scores"]["disclaimer"] == DISCLAIMER
    assert "not validated" in payload["heuristic_scores"]["status"]
