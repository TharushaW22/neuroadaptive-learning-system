"""Applicability screening: which candidate XAI methods can be used fairly in which context?

Each candidate method declares what it NEEDS (METHOD_REQUIREMENTS: knowledge about the method,
from its paper/library). Each context declares what it OFFERS (adapter Capabilities + the run
budget). The 8 checks compare the two, so the table is computed, not written by hand:

  1 model_access        the method's access type (outputs / gradients) is available
  2 differentiable      gradient methods need a differentiable path to the explained output
  3 attribution_output  the method returns one score per interpretable feature (needed for a
                        like-for-like comparison of fidelity, stability, agreement ...)
  4 removal_semantics   'absent feature' must have a defined replacement in this space
  5 feature_space_fit   the method's assumptions about features hold (e.g. independently
                        actionable / discretisable features)
  6 compute_budget      estimated cost per instance fits the per-instance budget
  7 probability_output  the explained output is a calibrated probability
  8 repeatable          the method can be re-run with another seed (stability / test-retest);
                        deterministic methods pass with a caution (never rewarded for stability)

A method is APPLICABLE when no check fails ('caution' is allowed and shown).
"""

from __future__ import annotations

import pandas as pd

METHOD_REQUIREMENTS: dict[str, dict] = {
    "kernel_shap": {"label": "KernelSHAP", "access": "outputs", "gradients": False, "attribution": True,
                    "removal": True, "feature_assumption": None, "evals": lambda b: b["nsamples"] * b["k_donors"],
                    "stochastic": True},
    "conditional_shap": {"label": "Conditional SHAP", "access": "outputs", "gradients": False, "attribution": True,
                         "removal": True, "feature_assumption": "conditional", "evals": lambda b: b["nsamples"] * b["k_donors"],
                         "stochastic": True},
    "lime": {"label": "LIME", "access": "outputs", "gradients": False, "attribution": True, "removal": True,
             "feature_assumption": None, "evals": lambda b: b["num_samples"] * b["k_donors"], "stochastic": True},
    "ig": {"label": "Integrated Gradients (donor baseline)", "access": "gradients", "gradients": True,
           "attribution": True, "removal": False, "feature_assumption": None,
           "evals": lambda b: b["k_donors"] * (3 * b["n_steps"] + 1), "stochastic": True},
    "ig_zero": {"label": "Integrated Gradients (zero baseline)", "access": "gradients", "gradients": True,
                "attribution": True, "removal": False, "feature_assumption": None,
                "evals": lambda b: 3 * b["n_steps"] + 1, "stochastic": False},
    "dice": {"label": "DiCE (counterfactuals)", "access": "outputs", "gradients": False, "attribution": False,
             "removal": False, "feature_assumption": "actionable",
             "output_note": "returns counterfactual examples, not a score per feature",
             "evals": None, "stochastic": True},
    "anchors": {"label": "Anchors (rules)", "access": "outputs", "gradients": False, "attribution": False,
                "removal": True, "feature_assumption": "discretisable",
                "output_note": "returns an IF-THEN rule with precision/coverage, not a score per feature",
                "evals": None, "stochastic": True},
}


def screen(context_id: str, caps, budgets: dict, ms_per_eval: float | None, max_seconds_per_instance: float,
           methods: list[str] | None = None) -> pd.DataFrame:
    """Return one row per (context, method) with the 8 checks, an overall verdict and reasons."""
    rows = []
    for m in (methods or list(METHOD_REQUIREMENTS)):
        req = METHOD_REQUIREMENTS[m]
        checks, reasons = {}, []

        def put(name, status, why=""):
            checks[name] = status
            if status != "pass" and why:
                reasons.append(why)

        put("model_access", "pass" if (req["access"] == "outputs" or caps.differentiable) else "fail",
            "needs gradients, model gives outputs only")
        if req["gradients"]:
            put("differentiable", "pass" if caps.differentiable else "fail",
                f"{caps.model_kind} model is not differentiable")
        else:
            put("differentiable", "pass")
        put("attribution_output", "pass" if req["attribution"] else "fail",
            req.get("output_note", "no per-feature attribution"))
        put("removal_semantics", "pass" if (not req["removal"] or caps.n_features > 0) else "fail",
            "no defined replacement for absent features")
        fa = req["feature_assumption"]
        if fa == "conditional":
            put("feature_space_fit", "pass" if caps.supports_conditional else "fail",
                "no conditional distribution can be fitted for this feature space")
        elif fa == "actionable":
            put("feature_space_fit", "fail", caps.notes.get(
                "not_actionable", "band-limited EEG components are not independently actionable inputs a learner could change"))
        elif fa == "discretisable":
            put("feature_space_fit", "caution", caps.notes.get(
                "binning", "rules need discretised features; band components must first be binned"))
        else:
            put("feature_space_fit", "pass")
        b = budgets.get(m)
        if req["evals"] is None or b is None:
            put("compute_budget", "caution" if req["attribution"] else "not assessed",
                "cost not estimated (method excluded on earlier checks)" if not req["attribution"] else "no budget set")
            est = None
        else:
            evals = req["evals"](b)
            est = evals * ms_per_eval / 1000 if ms_per_eval else None
            if est is None:
                put("compute_budget", "caution", "per-evaluation cost not measured yet")
            else:
                put("compute_budget", "pass" if est <= max_seconds_per_instance else "fail",
                    f"estimated {est:.0f}s per instance > budget {max_seconds_per_instance:.0f}s")
        put("probability_output", "pass" if caps.output == "probability" else "caution", "output is not a probability")
        put("repeatable", "pass" if req["stochastic"] else "caution",
            "deterministic: stability is trivially 1 and is not rewarded (reported separately)")

        verdict = "applicable" if "fail" not in checks.values() else "excluded"
        rows.append({"context_id": context_id, "method": m, "label": req["label"], **checks,
                     "est_seconds_per_instance": est, "verdict": verdict,
                     "reasons": "; ".join(reasons) if reasons else ""})
    return pd.DataFrame(rows)


CHECK_COLUMNS = ["model_access", "differentiable", "attribution_output", "removal_semantics",
                 "feature_space_fit", "compute_budget", "probability_output", "repeatable"]

CHECK_PLAIN = {
    "model_access": "Can the method access the model the way it needs?",
    "differentiable": "Does the method need gradients, and can we compute them?",
    "attribution_output": "Does it give a score for every feature?",
    "removal_semantics": "Is 'removing' a feature well defined?",
    "feature_space_fit": "Do the method's assumptions about features hold?",
    "compute_budget": "Is it affordable within the time budget?",
    "probability_output": "Does it explain a probability?",
    "repeatable": "Can it be re-run to test stability?",
}
