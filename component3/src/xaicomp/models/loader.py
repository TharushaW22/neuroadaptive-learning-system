"""Load the model that explains an instance: per-fold model, deployed checkpoint, or randomised copy.

* fold model (preferred, D1): trained by Component 3 with C1's unchanged `run_fold` on the four
  OTHER folds -> the explained window comes from a subject the model never saw (out-of-sample).
* deployed checkpoint (fallback): component1/models/T1_A8_cnn_lstm_trans.pt, trained on ALL
  48 subjects (C1 decision D18) -> every explained window is IN-SAMPLE. Outputs are flagged.
* randomised (negative control, D4): same architecture, weights re-initialised with a fixed seed
  (model-parameter randomisation test, Adebayo et al., 2018). Scalers and temperature are kept.

Nothing here writes files; the C1 checkpoint is only read (torch.load into a new object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from xaicomp import _bootstrap  # noqa: F401
from xaicomp.paths import resolve_c3

N_FOLDS = 5


@dataclass
class LoadedModel:
    model: torch.nn.Module            # eval mode
    channel_scale: np.ndarray         # (14,) C1 Scalers.channel_scale of the training subjects
    clip: float
    temperature: float
    classes: list[str]
    source: str                       # "fold" | "deployed"
    in_sample: bool
    weights: str = "trained"          # "trained" | "randomised"
    fold: int | None = None
    version: str = ""
    meta: dict = field(default_factory=dict)


def _build(spec: dict, model_config: dict, n_classes: int, n_samples: int = 512) -> torch.nn.Module:
    from eegengine.training.trainer import build_model
    return build_model(spec, model_config, n_classes, n_samples)


def load_deployed(checkpoint_id: str) -> LoadedModel:
    """C1's deployed checkpoint via C1's own loader (read-only)."""
    from eegengine.inference.predictors import load_predictor

    p = load_predictor(checkpoint_id)
    return LoadedModel(model=p.model.eval(), channel_scale=p.scalers.channel_scale.astype(np.float32),
                       clip=float(p.scalers.clip), temperature=float(p.temperature), classes=list(p.classes),
                       source="deployed", in_sample=True, version=p.version,
                       meta={"spec": p.spec, "note": "deployed model trained on all subjects (C1 D18)"})


def fold_model_path(fold_dir: Path | str, fold: int) -> Path:
    return resolve_c3(fold_dir) / f"fold{fold}.pt"


def fold_models_complete(fold_dir: Path | str) -> bool:
    return all(fold_model_path(fold_dir, f).exists() for f in range(N_FOLDS))


def load_fold(fold_dir: Path | str, fold: int) -> LoadedModel:
    ck = torch.load(fold_model_path(fold_dir, fold), map_location="cpu", weights_only=False)
    model = _build(ck["spec"], ck["model_config"], len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    sc = ck["scalers"]
    return LoadedModel(model=model.eval(), channel_scale=np.asarray(sc["channel_scale"], np.float32),
                       clip=float(sc["clip"]), temperature=float(ck["temperature"]), classes=list(ck["classes"]),
                       source="fold", in_sample=False, fold=fold, version=ck.get("version", ""),
                       meta={"spec": ck["spec"], "test_subjects": ck.get("test_subjects"),
                             "max_epochs": ck.get("train_config", {}).get("max_epochs")})


def randomise(lm: LoadedModel, seed: int) -> LoadedModel:
    """Same architecture with freshly initialised weights (never trained). Original untouched."""
    from eegengine.models.deep.hybrid import HybridModel  # noqa: F401  (spec type check below)

    spec = lm.meta.get("spec") or {}
    torch.manual_seed(seed)
    from xaicomp.config import load_c1_config
    model_cfg = load_c1_config("experiments/deep.yaml")["model"]
    model = _build(spec, model_cfg, len(lm.classes)).eval()
    return LoadedModel(model=model, channel_scale=lm.channel_scale.copy(), clip=lm.clip, temperature=lm.temperature,
                       classes=list(lm.classes), source=lm.source, in_sample=lm.in_sample, weights="randomised",
                       fold=lm.fold, version=f"{lm.version}+randomised(seed={seed})", meta=dict(lm.meta))


class ModelBank:
    """Returns the model to use for an instance of a given fold, following the context config."""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        src = ctx.get("model_source", "auto")
        complete = fold_models_complete(ctx["fold_models_dir"])
        if src == "fold" and not complete:
            raise FileNotFoundError(f"fold models incomplete in {ctx['fold_models_dir']} — run scripts/train_fold_models.py")
        self.source = "fold" if (src == "fold" or (src == "auto" and complete)) else "deployed"
        self._cache: dict[int, LoadedModel] = {}

    @property
    def in_sample(self) -> bool:
        return self.source == "deployed"

    def get(self, fold: int) -> LoadedModel:
        key = fold if self.source == "fold" else -1
        if key not in self._cache:
            lm = load_fold(self.ctx["fold_models_dir"], fold) if self.source == "fold" else load_deployed(self.ctx["deployed_checkpoint"])
            if self.ctx.get("weights", "trained") == "randomised":
                lm = randomise(lm, int(self.ctx.get("randomisation_seed", 1234)) + (0 if key < 0 else key))
            self._cache[key] = lm
        return self._cache[key]
