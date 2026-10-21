"""Negative control (D4): model-parameter randomisation test (Adebayo et al., 2018).

The same windows are explained for the trained model and for a copy with RE-INITIALISED weights.
  * method sanity:   a method whose explanations barely change when the model is destroyed does not
                     explain the model (similarity = Spearman(trained, randomised) per window, seed 0;
                     passes when the mean |similarity| is below `max_similarity`)
  * metric validity: fidelity must collapse on the random model (it has no signal to be faithful to);
                     ratio = mean control fidelity / mean trained fidelity, collapse when < `max_ratio`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaicomp.metrics.stability import spearman


def method_sanity(expl_trained: dict, expl_control: dict, max_similarity: float = 0.5) -> pd.DataFrame:
    """expl_*: {(instance, method, seed): values}. Uses the shared instances and the first seed."""
    rows = []
    shared = sorted({(i, m, s) for (i, m, s) in expl_control if (i, m, s) in expl_trained})
    by_method: dict[str, list[float]] = {}
    for i, m, s in shared:
        by_method.setdefault(m, []).append(spearman(expl_trained[(i, m, s)], expl_control[(i, m, s)]))
    for m, v in sorted(by_method.items()):
        mean_abs = float(np.nanmean(np.abs(v)))
        rows.append({"method": m, "n_pairs": len(v), "mean_similarity": float(np.nanmean(v)),
                     "mean_abs_similarity": mean_abs, "passes_sanity_check": bool(mean_abs < max_similarity)})
    return pd.DataFrame(rows)


def metric_collapse(im_trained: pd.DataFrame, im_control: pd.DataFrame, criteria: list[str],
                    max_ratio: float = 0.5) -> pd.DataFrame:
    shared = set(im_control["instance_id"]) & set(im_trained["instance_id"])
    a = im_trained[im_trained["instance_id"].isin(shared)]
    b = im_control[im_control["instance_id"].isin(shared)]
    rows = []
    for c in criteria:
        if c not in a or c not in b:
            continue
        mt, mc = float(a[c].mean()), float(b[c].mean())
        ratio = mc / mt if mt not in (0.0,) and mt == mt else float("nan")
        rows.append({"criterion": c, "trained_mean": mt, "control_mean": mc, "ratio_control_to_trained": ratio,
                     "collapses": bool(ratio == ratio and mt > 0 and ratio < max_ratio), "n_instances": len(shared)})
    return pd.DataFrame(rows)
