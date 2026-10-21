"""Inference on recent feature rows (webcam or video) with strict validity gating.

States returned:
  <class name>                     - a prediction from a compatible model
  UNKNOWN                          - data quality / quantity insufficient
  NO VALIDATED MODEL AVAILABLE     - no real model that has an evaluation report
Predictions from DEMO ONLY models are only produced when explicitly allowed and are
always flagged demo_only=True.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.feature_schema import FEATURE_COLUMNS
from src.models.baseline import baseline_predict_proba
from src.models.registry import ModelCompatibilityError, list_models, load_model
from src.processing.preprocessing import Normalizer
from src.utils.config import resolve_path

UNKNOWN = "UNKNOWN"
NO_MODEL = "NO VALIDATED MODEL AVAILABLE"


def predict_proba(model, meta: dict, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n_classes = len(meta["label_scheme"]["class_names"])
    if meta["model_kind"] == "baseline":
        return baseline_predict_proba(model, X, mask, n_classes)
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.float32)))
        return torch.softmax(logits, dim=1).numpy()


class Predictor:
    def __init__(self, model, meta: dict, cfg: dict):
        self.model, self.meta, self.cfg = model, meta, cfg
        self.window = int(meta["window"])
        self.features = list(meta["feature_columns"])
        self.normalizer = Normalizer.from_dict(meta["normalizer"])
        self.class_names = list(meta["label_scheme"]["class_names"])
        self.min_quality = max(float(cfg["dashboard"]["min_quality_for_prediction"]),
                               float(meta.get("min_valid_ratio", 0.0)))

    @classmethod
    def from_dir(cls, model_dir: Path, cfg: dict, allow_demo: bool = False) -> "Predictor":
        model, meta = load_model(model_dir, expected_features=FEATURE_COLUMNS, allow_demo=allow_demo)
        return cls(model, meta, cfg)

    def predict_frames(self, rows: pd.DataFrame) -> dict:
        base = {"model_name": self.meta["model_name"], "demo_only": bool(self.meta["demo_only"]),
                "label_scheme": self.meta["label_scheme"]["name"], "probabilities": None}
        missing = [c for c in self.features + ["quality_ok"] if c not in rows.columns]
        if missing:
            return {**base, "state": UNKNOWN, "reason": f"missing columns: {missing}", "quality": 0.0}
        recent = rows.tail(self.window)
        if len(recent) < self.window:
            return {**base, "state": UNKNOWN, "quality": float(recent["quality_ok"].astype(bool).mean()) if len(recent) else 0.0,
                    "reason": f"insufficient data: {len(recent)}/{self.window} samples"}
        q = float(recent["quality_ok"].astype(bool).mean())
        if q < self.min_quality:
            return {**base, "state": UNKNOWN, "quality": q,
                    "reason": f"quality {q:.2f} below threshold {self.min_quality:.2f}"}
        vals = recent[self.features].astype(float).copy()
        vals[~recent["quality_ok"].astype(bool).to_numpy()] = np.nan
        vals = vals.ffill(limit=int(self.cfg["sequences"].get("max_interpolate_gap", 3)))
        X = self.normalizer.transform(vals.to_numpy()[None, ...])
        mask = np.ones((1, self.window), dtype=np.float32)
        proba = predict_proba(self.model, self.meta, X, mask)[0]
        k = int(np.argmax(proba))
        state = self.class_names[k]
        if self.meta["demo_only"]:
            state = f"{state} (DEMO ONLY - not a valid cognitive-state estimate)"
        return {**base, "state": state, "quality": q, "reason": "ok",
                "probabilities": {c: float(p) for c, p in zip(self.class_names, proba)}}


def evaluated_model_names(cfg: dict) -> set[str]:
    names: set[str] = set()
    root = resolve_path(cfg, "evaluation_dir")
    if root.exists():
        for rep in root.glob("*/report.json"):
            try:
                with open(rep, encoding="utf-8") as fh:
                    r = json.load(fh)
                if r.get("data_source") == "real":
                    names.update(m["model_name"] for m in r.get("models", []))
            except Exception:
                continue
    return names


def find_validated_model(cfg: dict) -> dict | None:
    """Most recent REAL model that has a genuine evaluation report."""
    evaluated = evaluated_model_names(cfg)
    for m in list_models(resolve_path(cfg, "models_dir")):
        if m.get("data_source") == "real" and not m.get("demo_only") and m["model_name"] in evaluated:
            return m
    return None


def load_live_predictor(cfg: dict, model_dir: str | None = None, allow_demo: bool = False) -> tuple[Predictor | None, str]:
    """Return (predictor, message). Without a validated real model returns (None, NO_MODEL)
    unless a demo model is explicitly requested with allow_demo=True."""
    try:
        if model_dir:
            p = Predictor.from_dir(Path(model_dir), cfg, allow_demo=allow_demo)
            return p, "DEMO ONLY model loaded" if p.meta["demo_only"] else "Real model loaded"
        m = find_validated_model(cfg)
        if m is None:
            return None, NO_MODEL
        return Predictor.from_dir(Path(m["path"]), cfg), f"Validated model: {m['model_name']}"
    except ModelCompatibilityError as exc:
        return None, f"{NO_MODEL} ({exc})"
