"""Full preprocessing pipeline for one recording, driven by configs/preprocess.yaml.

  raw (uV, DC offset)
   -> mean removal + 0.5-45 Hz zero-phase FIR band-pass + 50 Hz notch      (filters.py)
   -> persistent bad-channel detection + spherical-spline interpolation    (bad_channels.py)
   -> optional ICA ocular removal (AF3/AF4 as EOG proxies, default OFF)    (ica.py)
   -> per-window quality flags on the cleaned signal                       (quality/checks.py)
   -> epochs (4 s, 50 % overlap) with metadata                             (epoching.py)
Normalisation is applied to the whole epoch set afterwards (normalize.py), because the
rest-baseline option needs both recordings of a subject.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from eegengine.io.stew import Recording
from eegengine.preprocess.bad_channels import repair_bad_channels
from eegengine.preprocess.epoching import make_epochs
from eegengine.preprocess.filters import filter_recording
from eegengine.preprocess.ica import run_ica
from eegengine.quality.checks import QualityConfig, assess_recording


def config_hash(cfg: dict) -> str:
    """Short stable hash of the parts of the config that change the epochs."""
    keys = ("filter", "bad_channels", "ica", "quality", "epoching", "normalisation")
    sub = {k: cfg.get(k) for k in keys}
    return hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:8]


def epoch_quality_config(cfg: dict) -> QualityConfig:
    """Quality windows must coincide with the epochs, so window/step come from `epoching`."""
    q = dict(cfg["quality"])
    q["window_s"] = cfg["epoching"]["window_s"]
    q["step_s"] = cfg["epoching"]["step_s"]
    return QualityConfig.from_dict(q)


def preprocess_recording(rec: Recording, cfg: dict) -> dict:
    """Clean one recording and cut it into epochs. Returns arrays + logs (nothing is dropped)."""
    fs, ch = rec.fs, list(rec.ch_names)
    qcfg = epoch_quality_config(cfg)

    filtered = filter_recording(rec.data, fs, cfg["filter"])
    cleaned, bad_log = repair_bad_channels(filtered, fs, ch, qcfg, cfg["bad_channels"])

    ica_log = None
    if cfg.get("ica", {}).get("enabled", False):
        cleaned, ica_log = run_ica(cleaned, fs, ch, cfg["ica"], seed=cfg.get("seed", 42))

    q_rows, _ = assess_recording(cleaned, fs, ch, qcfg)
    epochs, starts = make_epochs(cleaned, fs, cfg["epoching"]["window_s"], cfg["epoching"]["step_s"])
    assert len(q_rows) == len(epochs), "quality windows and epochs must align"

    meta = [{
        "subject_id": rec.subject_id,
        "condition": rec.condition,
        "rating": rec.rating,
        "epoch_index": i,
        "t_start": float(s / fs),
        "t_end": float((s + epochs.shape[2]) / fs),
        "quality_flag": q["flag"],
        "quality_score": q["score"],
        "bad_channels": q["bad_channels"],
        "interpolated": ";".join(bad_log["interpolated"]),
    } for i, (s, q) in enumerate(zip(starts, q_rows))]

    return {
        "filtered": filtered,
        "cleaned": cleaned,
        "epochs": epochs.astype(np.float32),
        "meta": meta,
        "quality_rows": q_rows,
        "bad_channel_log": bad_log,
        "ica_log": ica_log,
    }
