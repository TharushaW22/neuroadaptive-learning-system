"""Component 2's scikit-learn baseline — SCREENING ONLY (vision analogue of tabular_component1.py).

C2's baseline (src/models/baseline.py) summarises each 20-frame window by mean/std/min/max of the 15
features (60 inputs) and fits a logistic regression or random forest. It has no differentiable path, so
gradient methods (Integrated Gradients) are screened out. The joblib file is never unpickled (it may
need C2's scikit-learn < 1.6); facts come from the context config and, if present, the checkpoint's
metadata.json (read-only).
"""

from __future__ import annotations

from xaicomp.adapters.base import Capabilities
from xaicomp.adapters.vision_component2 import VISION_SCREENING_NOTES
from xaicomp.models.vision_c2 import find_model_dirs, read_metadata
from xaicomp.paths import resolve_c3


def vision_baseline_capabilities(ctx: dict) -> Capabilities:
    meta = {}
    for d in find_model_dirs(resolve_c3(ctx["component2"]["models_dir"])):
        m = read_metadata(d)
        if m.get("model_kind") == "baseline" and m.get("data_source") == "real" and not m.get("demo_only"):
            meta = m
            break
    kind = (meta.get("model_params") or {}).get("type", "logistic regression / random forest")
    return Capabilities(differentiable=False, n_features=60, model_kind=f"scikit-learn baseline ({kind})",
                        supports_conditional=True, output="probability",
                        notes={**VISION_SCREENING_NOTES, "feature_groups": ctx.get("feature_groups"),
                               "checkpoint_found": bool(meta),
                               "model_name": meta.get("model_name")})
