"""Locate and load the generated epochs / features from their config hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eegengine.config import load_config, resolve
from eegengine.preprocess.pipeline import config_hash


def feature_hash(fcfg: dict, pcfg: dict) -> str:
    keys = ("families", "welch", "wavelet", "connectivity", "graph")
    sub = {k: fcfg.get(k) for k in keys}
    sub["preprocess"] = config_hash(pcfg)
    return hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:8]


def epochs_paths(pcfg: dict) -> tuple[Path, Path]:
    h = config_hash(pcfg)
    d = resolve(pcfg["data"]["out_dir"])
    return d / f"epochs_{h}.npz", d / f"epochs_{h}_meta.parquet"


def load_epochs(preprocess_config: str | dict = "preprocess.yaml") -> tuple[np.ndarray, pd.DataFrame]:
    pcfg = load_config(preprocess_config) if isinstance(preprocess_config, str) else preprocess_config
    xp, mp = epochs_paths(pcfg)
    if not xp.exists():
        raise FileNotFoundError(f"{xp} missing — run scripts/run_preprocess.py first")
    with np.load(xp) as z:
        X = z["X"]
    return X, pd.read_parquet(mp)


def features_paths(fcfg: dict, pcfg: dict) -> tuple[Path, Path]:
    h = feature_hash(fcfg, pcfg)
    d = resolve(fcfg["out_dir"])
    return d / f"features_{h}.parquet", d / f"features_{h}_arrays.npz"


def load_features(features_config: str | dict = "features.yaml"):
    """Return (features DataFrame, meta DataFrame, arrays dict). Rows align with the epochs."""
    fcfg = load_config(features_config) if isinstance(features_config, str) else features_config
    pcfg = load_config(fcfg["preprocess_config"])
    fp, ap = features_paths(fcfg, pcfg)
    if not fp.exists():
        raise FileNotFoundError(f"{fp} missing — run scripts/run_features.py first")
    F = pd.read_parquet(fp)
    _, mp = epochs_paths(pcfg)
    meta = pd.read_parquet(mp)
    with np.load(ap) as z:
        arrays = {k: z[k] for k in z.files}
    assert len(F) == len(meta), "features and metadata are misaligned"
    return F, meta, arrays
