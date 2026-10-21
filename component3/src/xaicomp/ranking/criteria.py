"""Which criteria enter the ranking, with which weight factor (decided by the reliability analysis).

Rules (configs/ranking.yaml, fixed before results; no weight is tuned for any method):
  * ICC < min_icc_keep                       -> dropped (the metric is not repeatable)
  * min_icc_keep <= ICC < full_weight_icc    -> weight multiplied by the ICC (down-weighted)
  * not monotonic under controlled degradation -> dropped (the metric does not track quality)
  * listed in must_fail_control but does not collapse on the randomised model -> dropped
  * reliability not assessable in this run   -> kept with full weight, flagged
  * redundant group (|rho| > redundancy_rho) -> merged into one criterion (mean of the scaled values)

Scaling to a fixed positive range (independent of the data, so no method is favoured):
  fidelity_*, stability, robustness: Spearman/probability-difference in [-1, 1] -> (x + 1) / 2
  sparsity (Gini) in [0, 1] -> x;  cost_seconds -> kept as a COST criterion (> 0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCALE = {"fidelity_on": "pm1", "fidelity_off": "pm1", "stability": "pm1", "robustness": "pm1",
         "sparsity": "unit", "cost_seconds": "cost"}


def scale_values(x: np.ndarray, criterion: str) -> np.ndarray:
    kind = SCALE.get(criterion, "unit")
    if kind == "pm1":
        return (np.clip(x, -1, 1) + 1) / 2
    if kind == "cost":
        return np.maximum(x, 1e-6)
    return np.clip(x, 0, 1)


def decide(criteria: list[str], icc: pd.DataFrame | None, degradation: pd.DataFrame | None,
           collapse: pd.DataFrame | None, groups: list[list[str]], rules: dict) -> pd.DataFrame:
    rows = []
    for c in criteria:
        factor, decision, reasons = 1.0, "keep", []
        v = np.nan
        if icc is not None and c in set(icc["criterion"]):
            v = float(icc.set_index("criterion").loc[c, "icc"])
        if v == v:
            if v < rules["min_icc_keep"]:
                decision, factor = "drop", 0.0
                reasons.append(f"poor repeatability (ICC {v:.2f} < {rules['min_icc_keep']})")
            elif v < rules["full_weight_icc"]:
                decision, factor = "down-weight", v
                reasons.append(f"moderate repeatability (ICC {v:.2f}) -> weight x {v:.2f}")
        else:
            reasons.append("repeatability not assessable in this run")
        mono = np.nan
        if degradation is not None and c in set(degradation["criterion"]):
            row = degradation.set_index("criterion").loc[c]
            mono = row["rho_level_vs_value"]
            if rules.get("require_monotonic", True) and not bool(row["monotonic_decrease"]) and mono == mono:
                decision, factor = "drop", 0.0
                reasons.append(f"does not fall monotonically when explanations are degraded (rho {mono:.2f})")
        elif c != "cost_seconds":
            reasons.append("degradation response not assessed")
        col = np.nan
        if c in rules.get("must_fail_control", []) and collapse is not None and c in set(collapse["criterion"]):
            col = bool(collapse.set_index("criterion").loc[c, "collapses"])
            if not col:
                decision, factor = "drop", 0.0
                reasons.append("does not collapse on the randomised-weights model")
        group = next((g for g in groups if c in g), [c])
        rows.append({"criterion": c, "icc": v, "monotonic_rho": mono, "collapses_on_control": col,
                     "decision": decision, "weight_factor": factor, "redundancy_group": " + ".join(sorted(group)),
                     "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


def final_criteria(decisions: pd.DataFrame, groups: list[list[str]]) -> list[dict]:
    """Merge kept criteria that are redundant -> list of {name, members, factor, benefit}."""
    kept = decisions[decisions["decision"] != "drop"].set_index("criterion")
    out, used = [], set()
    for c in kept.index:
        if c in used:
            continue
        g = [m for m in next((g for g in groups if c in g), [c]) if m in kept.index]
        if "cost_seconds" in g and len(g) > 1:              # never merge a cost with benefit criteria
            g = [c] if c == "cost_seconds" else [m for m in g if m != "cost_seconds"]
        used |= set(g)
        out.append({"name": " + ".join(g), "members": g, "factor": float(kept.loc[g, "weight_factor"].mean()),
                    "benefit": "cost_seconds" not in g})
    return out


def build_values(B: np.ndarray, crit_names: list[str], final: list[dict]) -> np.ndarray:
    """Bootstrap values (..., methods, raw criteria) -> (..., methods, final criteria) on fixed scales."""
    cols = []
    for f in final:
        parts = [scale_values(B[..., crit_names.index(m)], m) for m in f["members"]]
        cols.append(np.mean(parts, axis=0))
    return np.stack(cols, axis=-1)
