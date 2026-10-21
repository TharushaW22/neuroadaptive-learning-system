"""Step 15 (vision): Component 3's copies of C2's GRU, normaliser and window builder behave exactly like
the ORIGINAL files in component2/vision_cbu (loaded read-only by file path; no bytecode written there)."""

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from xaicomp._bootstrap import REPO_ROOT
from xaicomp.models import vision_c2 as C3

C2 = REPO_ROOT / "component2" / "vision_cbu" / "src"


def load_original(rel: str):
    path = C2 / rel
    if not path.exists():
        pytest.skip(f"component2 source not found: {path}")
    spec = importlib.util.spec_from_file_location(f"c2_original_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                          # dataclasses need the module registered
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_feature_columns_match_c2_schema():
    fs = load_original("data/feature_schema.py")
    assert C3.C2_FEATURE_COLUMNS == fs.FEATURE_COLUMNS
    assert C3.C2_SCHEMA_VERSION == fs.SCHEMA_VERSION


def test_temporal_classifier_copy_identical():
    tm = load_original("models/temporal_model.py")
    torch.manual_seed(0)
    a = tm.TemporalClassifier(15, 2, hidden_size=16).eval()
    b = C3.TemporalClassifier(15, 2, hidden_size=16).eval()
    b.load_state_dict(a.state_dict())                     # same parameter names -> state_dict interchangeable
    x = torch.randn(5, 20, 15)
    mask = torch.ones(5, 20)
    mask[0, -7:] = 0
    with torch.no_grad():
        assert torch.allclose(a(x, mask), b(x, mask))


def test_normalizer_copy_identical():
    pp = load_original("processing/preprocessing.py")
    rng = np.random.default_rng(0)
    X = rng.normal(3, 2, (40, 20, 15))
    X[rng.random(X.shape) < 0.05] = np.nan
    orig = pp.Normalizer("standard").fit(X)
    copy = C3.Normalizer.from_dict(orig.to_dict())
    np.testing.assert_array_equal(orig.transform(X), copy.transform(X))


def test_window_builder_copy_identical():
    pp = load_original("processing/preprocessing.py")
    sb = load_original("data/sequence_builder.py")
    rng = np.random.default_rng(1)
    rows = []
    for v, n in (("a", 47), ("b", 12), ("c", 60)):
        df = pd.DataFrame(rng.normal(size=(n, 15)), columns=C3.C2_FEATURE_COLUMNS)
        df = df.assign(video_id=v, participant_id=f"p{v}", split="test", timestamp_s=np.arange(n) / 5,
                       quality_ok=rng.random(n) > 0.2, label_engagement=2.0)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    a = pp.forward_fill_within_video(df, C3.C2_FEATURE_COLUMNS, 3)
    b = C3.forward_fill_within_video(df, C3.C2_FEATURE_COLUMNS, 3)
    pd.testing.assert_frame_equal(a, b)
    seq, _ = sb.build_sequences(a, C3.C2_FEATURE_COLUMNS, 20, 10, 0.6, label_column="label_engagement", label_transform=int)
    ws = C3.build_windows(b, C3.C2_FEATURE_COLUMNS, 20, 10, 0.6, "label_engagement")
    np.testing.assert_array_equal(seq.X, ws.X)
    np.testing.assert_array_equal(seq.mask, ws.mask)
    np.testing.assert_array_equal(seq.y, ws.y_level)
    np.testing.assert_array_equal(seq.video_ids, ws.video_ids)
    np.testing.assert_array_equal(seq.participant_ids, ws.participant_ids)
    np.testing.assert_array_equal(seq.start_ts, ws.start_ts)
    assert len(ws) > 0 and np.all((ws.quality_share >= 0.6) & (ws.quality_share <= 1))


def test_no_bytecode_written_into_component2():
    caches = list((REPO_ROOT / "component2").rglob("__pycache__"))
    assert caches == [], caches
