"""Model checkpoint saving/loading with compatibility validation.

Each model lives in its own folder:
  <root>/<model_name>/metadata.json  + model.joblib (baseline) | model.pt (temporal)

Real models go to models/. Synthetic (DEMO ONLY) models go to data/demo/models/ and are
refused by `load_model` unless allow_demo=True.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.data.feature_schema import FEATURE_COLUMNS, SCHEMA_VERSION, schema_fingerprint

REQUIRED_META = ("model_name", "model_kind", "data_source", "demo_only", "feature_schema_version",
                 "feature_columns", "label_scheme", "window", "normalizer", "model_params")


class ModelCompatibilityError(RuntimeError):
    pass


def _versions() -> dict:
    import numpy, sklearn
    v = {"python": platform.python_version(), "numpy": numpy.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        v["torch"] = torch.__version__
    except ImportError:
        pass
    return v


def save_model(root: Path, model_name: str, kind: str, model, metadata: dict) -> Path:
    out = Path(root) / model_name
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.update(model_name=model_name, model_kind=kind,
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                feature_schema_version=SCHEMA_VERSION,
                schema_fingerprint=schema_fingerprint(meta["feature_columns"]),
                library_versions=_versions())
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"Metadata missing keys: {missing}")
    if kind == "baseline":
        joblib.dump(model, out / "model.joblib")
    elif kind == "temporal":
        import torch
        torch.save(model.state_dict(), out / "model.pt")
    else:
        raise ValueError(kind)
    with open(out / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return out


def read_metadata(model_dir: Path) -> dict:
    p = Path(model_dir) / "metadata.json"
    if not p.exists():
        raise ModelCompatibilityError(f"No metadata.json in {model_dir}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def validate_metadata(meta: dict, expected_features: list[str] | None = None,
                      allow_demo: bool = False, expected_classes: list[str] | None = None) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ModelCompatibilityError(f"Checkpoint metadata incomplete: {missing}")
    if meta["feature_schema_version"] != SCHEMA_VERSION:
        raise ModelCompatibilityError(
            f"Feature schema {meta['feature_schema_version']} != current {SCHEMA_VERSION}; retrain the model.")
    feats = expected_features if expected_features is not None else FEATURE_COLUMNS
    if list(meta["feature_columns"]) != list(feats):
        raise ModelCompatibilityError("Model feature columns do not match the input feature columns.")
    if (meta["demo_only"] or meta["data_source"] != "real") and not allow_demo:
        raise ModelCompatibilityError("DEMO ONLY model (trained on synthetic data) - refused for real inference.")
    if expected_classes is not None and list(meta["label_scheme"]["class_names"]) != list(expected_classes):
        raise ModelCompatibilityError("Model label classes do not match the requested label scheme.")


def load_model(model_dir: Path, expected_features: list[str] | None = None, allow_demo: bool = False,
               expected_classes: list[str] | None = None):
    model_dir = Path(model_dir)
    meta = read_metadata(model_dir)
    validate_metadata(meta, expected_features, allow_demo, expected_classes)
    if meta["model_kind"] == "baseline":
        f = model_dir / "model.joblib"
        if not f.exists():
            raise ModelCompatibilityError(f"Missing {f}")
        return joblib.load(f), meta
    if meta["model_kind"] == "temporal":
        import torch
        from src.models.temporal_model import TemporalClassifier
        f = model_dir / "model.pt"
        if not f.exists():
            raise ModelCompatibilityError(f"Missing {f}")
        model = TemporalClassifier(**meta["model_params"])
        model.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        model.eval()
        return model, meta
    raise ModelCompatibilityError(f"Unknown model kind {meta['model_kind']}")


def list_models(root: Path) -> list[dict]:
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            try:
                m = read_metadata(d)
                m["path"] = str(d)
                out.append(m)
            except Exception:
                continue
    return sorted(out, key=lambda m: m.get("created_utc", ""), reverse=True)


def latest_model(root: Path, kind: str, data_source: str = "real") -> dict | None:
    for m in list_models(root):
        if m.get("model_kind") == kind and m.get("data_source") == data_source:
            return m
    return None
