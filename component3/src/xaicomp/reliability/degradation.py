"""Does each metric respond MONOTONICALLY when explanations are made worse on purpose?

Controlled degradation: an attribution vector a is mixed with random noise e (same norm):
    a_alpha = (1 - alpha) * a / ||a||  +  alpha * e / ||e||,    alpha in levels (0 = original, 1 = pure noise)
A valid quality metric must get worse as alpha grows:
  fidelity      re-evaluated with the model on the degraded vectors (deletion/insertion)
  sparsity      Gini of |a_alpha|
  stability     each seed's vector degraded with INDEPENDENT noise -> agreement between seeds falls
  robustness    original and noisy-input vectors degraded independently
Monotonicity = Spearman correlation between alpha and the mean metric value (expected -1 for
higher-is-better metrics). Cost is not an attribution property -> not assessed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaicomp.metrics.sparsity import gini
from xaicomp.metrics.stability import spearman

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def degrade(a: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    e = rng.standard_normal(a.shape)
    na, ne = np.linalg.norm(a) + 1e-12, np.linalg.norm(e)
    return (1 - alpha) * a / na + alpha * e / ne


def monotonic(levels, values) -> float:
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)) or np.nanstd(v) == 0:
        return float("nan")
    return float(spearmanr(levels, v).statistic)


def attribution_only_degradation(explanations: dict, noisy: dict, seeds: list[int], levels=LEVELS,
                                 seed: int = 5) -> pd.DataFrame:
    """sparsity / stability / robustness under degradation (no model calls).
    explanations: {(instance, method, seed): values}; noisy: {(instance, method): values (noise0, first seed)}."""
    rng = np.random.default_rng(seed)
    rows = []
    keys = sorted({(i, m) for i, m, _ in explanations})
    for lv in levels:
        sp, st, rb = [], [], []
        for i, m in keys:
            deg = [degrade(explanations[(i, m, s)], lv, rng) for s in seeds]
            sp.append(gini(deg[0]))
            if len(deg) >= 2:
                st.append(np.nanmean([spearman(deg[a], deg[b]) for a in range(len(deg)) for b in range(a + 1, len(deg))]))
            if (i, m) in noisy:
                rb.append(spearman(degrade(explanations[(i, m, seeds[0])], lv, rng), degrade(noisy[(i, m)], lv, rng)))
        rows += [{"criterion": "sparsity", "level": lv, "value": float(np.nanmean(sp))},
                 {"criterion": "stability", "level": lv, "value": float(np.nanmean(st)) if st else np.nan},
                 {"criterion": "robustness", "level": lv, "value": float(np.nanmean(rb)) if rb else np.nan}]
    return pd.DataFrame(rows)


def fidelity_degradation_for_instance(view, attributions: dict[str, np.ndarray], pred_is_target: bool, step: int,
                                      n_random: int, modes: list[str], levels=LEVELS, seed: int = 5) -> list[dict]:
    """Fidelity of degraded versions of each attribution of one instance (model calls, batched)."""
    from xaicomp.metrics.fidelity import fidelity_for_instance

    rng = np.random.default_rng(seed)
    deg = {f"{k}@{lv}": degrade(a, lv, rng) for k, a in attributions.items() for lv in levels}
    rows, _ = fidelity_for_instance(view, deg, pred_is_target, step, n_random, modes, 999)
    out = []
    for r in rows:
        k, lv = r["key"].rsplit("@", 1)
        out.append({"key": k, "level": float(lv), "mode": r["mode"], "fidelity": r["fidelity"]})
    return out


def summarise(df: pd.DataFrame, min_abs_rho: float) -> pd.DataFrame:
    """One row per criterion: Spearman(level, mean value) and whether it falls monotonically."""
    rows = []
    for crit, g in df.groupby("criterion"):
        g = g.groupby("level")["value"].mean().reset_index()
        rho = monotonic(g["level"], g["value"])
        rows.append({"criterion": crit, "rho_level_vs_value": rho,
                     "values_by_level": g["value"].round(4).tolist(), "levels": g["level"].tolist(),
                     "monotonic_decrease": bool(rho == rho and rho <= -min_abs_rho)})
    return pd.DataFrame(rows)
