"""Tiny shape-level TEST FIXTURES for the vision context (used only inside tests/).

Random numbers with the SHAPES of Component 2's data (20 frames x 15 features, train/test split,
subject IDs) and a randomly initialised GRU. They exercise code paths only: they are never written to
results/, never shown on the dashboard and say nothing about DAiSEE or Component 2's model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xaicomp.data.vision_data import VisionData, split_check
from xaicomp.models.vision_c2 import (C2_FEATURE_COLUMNS, LoadedVisionModel, Normalizer, TemporalClassifier,
                                      WindowSet)

T, F = 20, 15
GROUPS = {"eyes_blink": ["ear_left", "ear_right", "ear_mean", "blink_event", "blink_rate_per_min"],
          "gaze": ["gaze_h", "gaze_v"],
          "head_pose": ["head_yaw", "head_pitch", "head_roll", "face_scale", "face_center_x", "face_center_y"],
          "mouth": ["mouth_aspect_ratio"], "brows": ["brow_raise"]}
MODEL_PARAMS = dict(n_features=F, n_classes=2, hidden_size=8, num_layers=1, cell="gru", dropout=0.0, bidirectional=False)
BINARY = {"name": "daisee_binary", "target": "engagement", "column": "label_engagement",
          "class_names": ["engagement_low(<2)", "engagement_high(>=2)"], "mapping": {"0": 0, "1": 0, "2": 1, "3": 1}}


def toy_ctx(**over) -> dict:
    ctx = {"context_id": "vision/toy", "title": "toy", "adapter": "vision_component2", "modality": "vision",
           "class_labels": ["low", "high"], "target_class": "high", "representation": {"segments": 4},
           "groups": GROUPS, "conditional": {"mode": "auto"}, "robustness_exclude": ["blink_event"],
           "component2": {"root": "tests/_tmp/c2", "models_dir": "tests/_tmp/c2/models",
                          "features_dir": "tests/_tmp/c2/data/processed/features", "config": "tests/_tmp/c2/cfg.yaml",
                          "model_name": "auto"},
           "expect": {"label_scheme": "daisee_binary", "target": "engagement", "cell": "gru", "window": 20}}
    ctx.update(over)
    return ctx


def toy_meta() -> dict:
    return {"model_name": "toy_gru", "model_kind": "temporal", "data_source": "real", "demo_only": False,
            "feature_schema_version": "1.0.0", "feature_columns": list(C2_FEATURE_COLUMNS), "label_scheme": BINARY,
            "window": T, "stride": 10, "sampling_fps": 5, "min_valid_ratio": 0.6,
            "normalizer": {"method": "standard", "fill_values": [0.0] * F, "a": [0.0] * F, "b": [1.0] * F},
            "model_params": MODEL_PARAMS, "created_utc": "2026-01-01T00:00:00+00:00",
            "library_versions": {"torch": torch.__version__, "numpy": np.__version__}}


def toy_model(seed: int = 0) -> LoadedVisionModel:
    torch.manual_seed(seed)
    m = TemporalClassifier(**MODEL_PARAMS).eval()
    meta = toy_meta()
    return LoadedVisionModel(model=m, normalizer=Normalizer.from_dict(meta["normalizer"]), meta=meta,
                             class_names=list(BINARY["class_names"]), path=Path("toy"))


def toy_windows(n_train_subj=6, n_test_subj=3, per_subj=12, overlap=False, seed=0, gaussian=True) -> WindowSet:
    rng = np.random.default_rng(seed)
    X, M, y, vid, pid, split, start, ts, q = [], [], [], [], [], [], [], [], []
    subjects = [("train", 100 + i) for i in range(n_train_subj)] + [("test", 200 + i) for i in range(n_test_subj)]
    if overlap:
        subjects.append(("test", 100))                           # a training subject also in the test split
    for sp, s in subjects:
        for j in range(per_subj):
            x = rng.normal(0, 1, (T, F)) if gaussian else rng.exponential(1, (T, F)) ** 3
            x[:, 3] = (rng.random(T) < 0.1).astype(float)          # blink_event is 0/1
            m = np.ones(T, np.float32)
            if j == 0 and sp == "test":
                m[-5:] = 0                                         # one short clip padded
                x[-5:] = 0
            X.append(x.astype(np.float32)); M.append(m); y.append(int(rng.integers(0, 4)))
            vid.append(f"{s}{j // 3:03d}"); pid.append(str(s)); split.append(sp); start.append((j % 3) * 10)
            ts.append(float((j % 3) * 2)); q.append(float(m.mean()))
    return WindowSet(np.stack(X), np.stack(M), np.array(y), np.array(vid), np.array(pid), np.array(split),
                     np.array(start), np.array(ts), np.array(q), {"window": T, "stride": 10})


def toy_data(**kw) -> VisionData:
    ws = toy_windows(**kw)
    parts = pd.DataFrame({"participant_id": ws.participant_ids, "split": ws.splits})
    mapping = {int(k): v for k, v in BINARY["mapping"].items()}
    return VisionData(windows=ws, X=(ws.X * ws.mask[..., None]).astype(np.float32),
                      y=np.array([mapping[int(v)] for v in ws.y_level]), split=split_check(parts),
                      source={"toy": True})


def toy_adapter(**kw):
    from xaicomp.adapters.vision_component2 import VisionComponent2Adapter
    ctx_over = kw.pop("ctx", {})
    return VisionComponent2Adapter(toy_ctx(**ctx_over), data=toy_data(**kw), model=toy_model(), threads=1)


def write_toy_c2(root: Path, with_model=True, with_features=True, scheme=BINARY, overlap=False) -> None:
    """A Component-2-shaped folder (model.pt + metadata.json, per-clip feature CSVs) under tests/_tmp/."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg.yaml").write_text("sequences: {max_interpolate_gap: 3}\n")
    if with_model:
        d = root / "models" / "toy_gru"
        d.mkdir(parents=True, exist_ok=True)
        meta = toy_meta()
        meta["label_scheme"] = scheme
        torch.manual_seed(0)
        torch.save(TemporalClassifier(**MODEL_PARAMS).state_dict(), d / "model.pt")
        (d / "metadata.json").write_text(json.dumps(meta))
    if with_features:
        fd = root / "data" / "processed" / "features"
        fd.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        clips = [("train", f"10{i}", lvl) for i in range(4) for lvl in (0, 3)] + \
                [("test", f"20{i}", lvl) for i in range(2) for lvl in (1, 2)]
        if overlap:
            clips.append(("test", "100", 2))
        for k, (sp, p, lvl) in enumerate(clips):
            n = 50
            df = pd.DataFrame(rng.normal(0, 1, (n, F)), columns=C2_FEATURE_COLUMNS)
            df["blink_event"] = (rng.random(n) < 0.1).astype(float)
            df = df.assign(participant_id=p, video_id=f"{p}{k:04d}", timestamp_s=np.arange(n) / 5.0, split=sp,
                           source="daisee", is_synthetic=False, label_engagement=float(lvl), quality_ok=True)
            df.to_csv(fd / f"{p}{k:04d}.csv.gz", index=False, compression="gzip")
