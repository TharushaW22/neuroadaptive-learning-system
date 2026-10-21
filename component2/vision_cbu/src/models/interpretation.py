"""EXPERIMENTAL heuristic scores: attention, fatigue and learning readiness in [0, 1].

These are NOT model outputs and NOT validated measurements. The only trained model in this
project predicts the DAiSEE engagement level (0-3). The three scores below are fixed weighted
sums of that predicted level and simple visual proxies over the same window. The exact
formulas and their limitations are in docs/labeling_protocol.md (section 6).

    e           = engagement_level / 3
    attention   = clip(w_e*e + w_f*face_vis + w_p*pose_stable + w_o*ear_open)
    fatigue     = clip(w_l*(1-e) + w_b*blink_high + w_c*(1-ear_open) + w_u*(1-pose_stable))
    readiness   = clip(w_e*e + w_a*attention + w_f*(1-fatigue))

When the window's face quality is poor (quality_state == UNKNOWN) or no engagement level is
available, all three scores are None. Nothing is guessed.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

QUALITY_OK = "OK"
QUALITY_UNKNOWN = "UNKNOWN"

SCORE_COLUMNS = ["attention_score", "fatigue_score", "readiness_score"]
INTERPRETATION_COLUMNS = ["engagement_level", *SCORE_COLUMNS, "quality_state", "interpretation_source",
                          "interpretation_reason"]
PROXY_NAMES = ["face_vis", "ear_open", "blink_high", "pose_stable", "quality"]

DISCLAIMER = ("Attention, fatigue, and readiness are heuristic scores in [0,1] derived from engagement and "
              "visual proxies. They are not validated cognitive-state measurements.")

# Display-only bins. Stored values stay continuous.
SCORE_BINS = [(0.0, 1 / 3, "0.00-0.33"), (1 / 3, 2 / 3, "0.33-0.66"), (2 / 3, 1.0, "0.66-1.00")]

DEFAULTS = {
    "enabled": True,
    "version": "heuristic_v1",
    "attention_weights": {"engagement": 0.55, "face_vis": 0.20, "pose_stable": 0.15, "ear_open": 0.10},
    "fatigue_weights": {"low_engagement": 0.40, "blink_high": 0.30, "eyes_closed": 0.20, "pose_unstable": 0.10},
    "readiness_weights": {"engagement": 0.50, "attention": 0.30, "low_fatigue": 0.20},
    "proxies": {"ear_closed": 0.15, "ear_open": 0.30, "blink_rate_normal": 17.0, "blink_rate_high": 35.0,
                "pose_motion_max_deg": 15.0, "min_valid_frames": 0.6},
}

_LEVEL_RE = re.compile(r"^engagement_([0-3])$")


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _norm(x: float, lo: float, hi: float) -> float:
    """Linear map of x from [lo, hi] to [0, 1], clipped."""
    if hi <= lo:
        raise ValueError(f"normalisation range must have hi > lo (got {lo}, {hi})")
    return clip01((x - lo) / (hi - lo))


def interpretation_config(cfg: dict) -> dict:
    """The `interpretation` config section merged over DEFAULTS, with weights validated."""
    user = cfg.get("interpretation") or {}
    out = {**DEFAULTS, **{k: v for k, v in user.items() if not isinstance(v, dict)}}
    for key in ("attention_weights", "fatigue_weights", "readiness_weights", "proxies"):
        out[key] = {**DEFAULTS[key], **(user.get(key) or {})}
    for key in ("attention_weights", "fatigue_weights", "readiness_weights"):
        unknown = set(out[key]) - set(DEFAULTS[key])
        if unknown:
            raise ValueError(f"interpretation.{key}: unknown weight names {sorted(unknown)}")
        if any(float(w) < 0 for w in out[key].values()):
            raise ValueError(f"interpretation.{key}: weights must be non-negative")
    return out


def engagement_level_from_prediction(pred: dict | None) -> int | None:
    """Predicted DAiSEE engagement level 0-3, or None.

    Only the default `daisee_original` scheme on the engagement target gives a 0-3 level;
    other schemes (binary, operational, other targets) return None.
    """
    if not pred or not pred.get("probabilities") or pred.get("reason") != "ok":
        return None
    probs = pred["probabilities"]
    best = max(probs, key=probs.get)
    m = _LEVEL_RE.match(str(best))
    return int(m.group(1)) if m else None


def compute_proxies(window: pd.DataFrame, icfg: dict) -> dict:
    """Window-level visual proxies, each in [0, 1] (NaN when not measurable)."""
    p = icfg["proxies"]
    nan = math.nan
    if window is None or len(window) == 0:
        return {k: nan for k in PROXY_NAMES}
    ok = window["quality_ok"].astype(bool).to_numpy() if "quality_ok" in window else np.zeros(len(window), bool)
    good = window[ok]

    face_vis = float(window["face_detected"].astype(bool).mean()) if "face_detected" in window else nan
    quality = 1.0 if (ok.mean() >= float(p["min_valid_frames"]) and len(good) > 0) else 0.0

    ear = pd.to_numeric(good.get("ear_mean"), errors="coerce").dropna() if len(good) else pd.Series(dtype=float)
    ear_open = _norm(float(ear.mean()), float(p["ear_closed"]), float(p["ear_open"])) if len(ear) else nan

    rate = pd.to_numeric(good.get("blink_rate_per_min"), errors="coerce").dropna() if len(good) else pd.Series(dtype=float)
    blink_high = (_norm(float(rate.mean()), float(p["blink_rate_normal"]), float(p["blink_rate_high"]))
                  if len(rate) else nan)

    pose_stable = nan
    if len(good) >= 2 and all(c in good for c in ("head_yaw", "head_pitch", "head_roll")):
        angles = good[["head_yaw", "head_pitch", "head_roll"]].apply(pd.to_numeric, errors="coerce")
        stds = angles.std(ddof=0).dropna()
        if len(stds):
            pose_stable = 1.0 - _norm(float(stds.mean()), 0.0, float(p["pose_motion_max_deg"]))
    return {"face_vis": face_vis, "ear_open": ear_open, "blink_high": blink_high,
            "pose_stable": pose_stable, "quality": quality}


def compute_scores(engagement_level: int, proxies: dict, icfg: dict) -> dict:
    """The three heuristic scores (unrounded floats in [0, 1]). Proxies must all be finite."""
    if engagement_level not in (0, 1, 2, 3):
        raise ValueError(f"engagement_level must be 0..3, got {engagement_level!r}")
    e = engagement_level / 3.0
    f, o, b, s = (clip01(proxies[k]) for k in ("face_vis", "ear_open", "blink_high", "pose_stable"))
    wa, wf, wr = icfg["attention_weights"], icfg["fatigue_weights"], icfg["readiness_weights"]
    attention = clip01(wa["engagement"] * e + wa["face_vis"] * f + wa["pose_stable"] * s + wa["ear_open"] * o)
    fatigue = clip01(wf["low_engagement"] * (1 - e) + wf["blink_high"] * b + wf["eyes_closed"] * (1 - o)
                     + wf["pose_unstable"] * (1 - s))
    readiness = clip01(wr["engagement"] * e + wr["attention"] * attention + wr["low_fatigue"] * (1 - fatigue))
    return {"attention_score": attention, "fatigue_score": fatigue, "readiness_score": readiness}


def empty_interpretation(cfg: dict, reason: str) -> dict:
    icfg = interpretation_config(cfg)
    source = icfg["version"] if icfg["enabled"] else "disabled"
    return {"engagement_level": None, "attention_score": None, "fatigue_score": None, "readiness_score": None,
            "quality_state": QUALITY_UNKNOWN, "interpretation_source": source, "interpretation_reason": reason}


def interpret_window(pred: dict | None, window: pd.DataFrame, cfg: dict) -> dict:
    """Required per-window fields: engagement_level, the three scores, quality_state, source.

    `pred` is the dict returned by Predictor.predict_frames (or None if no model is loaded);
    `window` is the same recent rows the model saw.
    """
    icfg = interpretation_config(cfg)
    level = engagement_level_from_prediction(pred)
    proxies = compute_proxies(window, icfg)
    model_ok = pred is not None and pred.get("reason") == "ok"
    min_q = float(cfg.get("dashboard", {}).get("min_quality_for_prediction", 0.6))
    window_q = float(window["quality_ok"].astype(bool).mean()) if window is not None and len(window) else 0.0
    proxies_ok = all(math.isfinite(proxies[k]) for k in ("face_vis", "ear_open", "blink_high", "pose_stable"))
    quality_ok = proxies["quality"] == 1.0 and window_q >= min_q and proxies_ok and (pred is None or model_ok)

    out = {"engagement_level": level, "attention_score": None, "fatigue_score": None, "readiness_score": None,
           "quality_state": QUALITY_OK if quality_ok else QUALITY_UNKNOWN,
           "interpretation_source": icfg["version"] if icfg["enabled"] else "disabled",
           "interpretation_reason": "ok"}
    if not icfg["enabled"]:
        out["interpretation_reason"] = "interpretation disabled in config"
    elif not quality_ok:
        missing = [k for k in ("face_vis", "ear_open", "blink_high", "pose_stable") if not math.isfinite(proxies[k])]
        if pred is not None and not model_ok:
            reason = pred.get("reason", "model returned UNKNOWN")
        elif proxies["quality"] != 1.0 or window_q < min_q:
            reason = f"face quality too low ({window_q:.2f} of frames OK)"
        elif missing == ["blink_high"]:
            reason = "blink rate not available yet (needs ~10 s of video)"
        else:
            reason = f"visual proxies unavailable: {missing}"
        out["interpretation_reason"] = reason
        if pred is not None and not model_ok:
            out["engagement_level"] = None
    elif level is None:
        out["interpretation_reason"] = ("no engagement model loaded" if pred is None
                                        else "model does not output a 0-3 engagement level")
    else:
        out.update(compute_scores(level, proxies, icfg))
    return out


def format_score(value, percent: bool = False) -> str:
    """'0.72' / '72%' for display; 'UNKNOWN' for None/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return QUALITY_UNKNOWN
    return f"{float(value) * 100:.0f}%" if percent else f"{float(value):.2f}"


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Session summary per score: counts, mean/median/min/max and display-bin counts."""
    rows = []
    for col in SCORE_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)
        v = s.dropna()
        r = {"score": col.replace("_score", ""), "n_valid": int(len(v)), "n_unknown": int(s.isna().sum()),
             "mean": round(float(v.mean()), 2) if len(v) else None,
             "median": round(float(v.median()), 2) if len(v) else None,
             "min": round(float(v.min()), 2) if len(v) else None,
             "max": round(float(v.max()), 2) if len(v) else None,
             "mean_%": f"{v.mean() * 100:.0f}%" if len(v) else None}
        for i, (lo, hi, label) in enumerate(SCORE_BINS):
            last = i == len(SCORE_BINS) - 1
            r[label] = int(((v >= lo) & ((v <= hi) if last else (v < hi))).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def export_metadata(cfg: dict) -> dict:
    """Block attached to JSON exports so the scores are never mistaken for validated outputs."""
    icfg = interpretation_config(cfg)
    return {"version": icfg["version"], "status": "EXPERIMENTAL / HEURISTIC - not validated",
            "disclaimer": DISCLAIMER, "trained_model_output": "DAiSEE engagement level 0-3 only",
            "attention_weights": icfg["attention_weights"], "fatigue_weights": icfg["fatigue_weights"],
            "readiness_weights": icfg["readiness_weights"], "proxies": icfg["proxies"],
            "documentation": "docs/labeling_protocol.md#6"}
