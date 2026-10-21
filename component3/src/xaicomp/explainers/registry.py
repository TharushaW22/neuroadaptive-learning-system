"""Method name -> explainer function. Every function: (view, seed, budget, cfg, target) -> Explanation."""

from __future__ import annotations

from xaicomp.explainers.integrated_gradients import explain_ig
from xaicomp.explainers.lime_explainer import explain_lime
from xaicomp.explainers.shapley import explain_conditional_shap, explain_kernel_shap

EXPLAINERS = {
    "kernel_shap": explain_kernel_shap,
    "conditional_shap": explain_conditional_shap,
    "lime": explain_lime,
    "ig": explain_ig,
    "ig_zero": explain_ig,
}

# explainers that need donor windows (the view must be built with k_donors > 0)
NEEDS_DONORS = {"kernel_shap", "conditional_shap", "lime", "ig"}
GRADIENT_METHODS = {"ig", "ig_zero"}


def explain(method: str, view, seed: int, budget: dict, cfg: dict, target: str = "high"):
    return EXPLAINERS[method](view, seed, budget, cfg, target)
