"""Minimal adapter for Component 1's XGBoost models — SCREENING ONLY (optional context, D4).

It only reports capabilities (no explanations are run): a tree ensemble on 1,499 tabular features,
grouped into C1's 6 feature families. The model file is not even loaded; facts come from the
context config and C1's model sidecar JSON (read-only).
"""

from __future__ import annotations

import json

from xaicomp.adapters.base import Capabilities
from xaicomp.paths import C1_MODELS_DIR


def tabular_capabilities(ctx: dict) -> Capabilities:
    sidecar = C1_MODELS_DIR / f"{ctx['deployed_checkpoint']}.json"
    info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return Capabilities(differentiable=False, n_features=len(ctx.get("feature_groups", [])), model_kind="tree ensemble (XGBoost)",
                        supports_conditional=True, output="probability",
                        notes={"feature_groups": ctx.get("feature_groups"), "version": info.get("version")})
