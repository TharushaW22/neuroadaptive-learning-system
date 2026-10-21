"""Load deployable models (classical .joblib or deep .pt) behind one small interface.

Every predictor has: id, kind, task, classes, version, seq_len (1 = single window) and
predict(buffer) -> calibrated class probabilities, where buffer is a list (oldest first) of
per-window dicts {"epoch": (C, T) uV, "features": DataFrame row, "bandpower": (C, 5), "adj": (C, C)}.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from eegengine.config import PROJECT_ROOT
from eegengine.training.calibration import softmax
from eegengine.training.trainer import Scalers, build_model, forward, is_epoch_model

MODELS_DIR = PROJECT_ROOT / "models"


class ClassicalPredictor:
    kind = "classical"

    def __init__(self, path: Path):
        b = joblib.load(path)
        self.id = path.stem
        self.model, self.columns = b["estimator"], b["columns"]
        self.task, self.classes, self.version = b["task"], b["classes"], b["version"]
        self.seq_len = 1
        self.meta = {k: v for k, v in b.items() if k not in ("estimator", "columns")}

    def predict(self, buffer: list[dict]) -> np.ndarray:
        row = buffer[-1]["features"][self.columns].to_numpy(dtype=np.float32)
        return self.model.predict_scores(row)[0]


class DeepPredictor:
    kind = "deep"

    def __init__(self, path: Path, threads: int = 1):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.id = path.stem
        self.spec, self.task, self.classes, self.version = ck["spec"], ck["task"], ck["classes"], ck["version"]
        self.temperature = ck["temperature"]
        self.scalers = Scalers.from_dict(ck["scalers"])
        n_samples = int(ck.get("n_samples", 512))
        self.model = build_model(self.spec, ck["model_config"], len(self.classes), n_samples)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.seq_len = 1 if is_epoch_model(self.spec) else ck["train_config"]["seq_len"]
        self.meta = {"cv_summary": ck.get("cv_summary"), "spec": self.spec}
        torch.set_num_threads(threads)

    def predict(self, buffer: list[dict]) -> np.ndarray:
        s = self.scalers
        seq = buffer[-self.seq_len:]                                   # the engine may hold fewer at start
        x = np.stack([b["epoch"] for b in seq])[None] / s.channel_scale[None, None, :, None]
        x = np.clip(x, -s.clip, s.clip).astype(np.float32)
        node = ((np.log10(np.stack([b["bandpower"] for b in seq]) + 1e-12) - s.node_mean) / s.node_std)[None]
        adj = np.stack([b["adj"] for b in seq])[None]
        with torch.no_grad():
            if is_epoch_model(self.spec):
                lg = self.model(torch.from_numpy(x[0]))["workload"].numpy()        # each epoch separately
                return softmax(lg, self.temperature).mean(axis=0)
            out = forward(self.model, self.spec, torch.from_numpy(x), torch.from_numpy(node.astype(np.float32)),
                          torch.from_numpy(adj.astype(np.float32)))
        return softmax(out["workload"].numpy(), self.temperature)[0]


def load_predictor(model_id_or_path: str | Path, threads: int = 1):
    p = Path(model_id_or_path)
    if not p.exists():
        for ext in (".pt", ".joblib"):
            if (MODELS_DIR / f"{model_id_or_path}{ext}").exists():
                p = MODELS_DIR / f"{model_id_or_path}{ext}"
                break
    if not p.exists():
        raise FileNotFoundError(f"model {model_id_or_path!r} not found in {MODELS_DIR}")
    return DeepPredictor(p, threads) if p.suffix == ".pt" else ClassicalPredictor(p)


def list_models() -> list[dict]:
    """Registry of deployable models (from the .json sidecar written next to each model file)."""
    out = []
    for j in sorted(MODELS_DIR.glob("*.json")):
        with open(j, encoding="utf-8") as f:
            info = json.load(f)
        files = [p for p in (j.with_suffix(".pt"), j.with_suffix(".joblib")) if p.exists()]
        if not files:
            continue
        out.append({"id": j.stem, "file": files[0].name, "kind": "deep" if files[0].suffix == ".pt" else "classical",
                    **{k: info.get(k) for k in ("version", "task", "classes", "temperature", "cv_summary", "spec",
                                                "model", "feature_set")}})
    return out
