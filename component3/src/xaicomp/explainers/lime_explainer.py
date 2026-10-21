"""LIME (Ribeiro et al., 2016) on the binary component space, via lime.lime_base.LimeBase.

Sampling follows LIME's text explainer: each perturbed sample removes a uniformly random number
(1..F) of randomly chosen components; the first sample is the unmodified instance. Removed
components are filled with donor components (same replacement as KernelSHAP), the model output is
averaged over the K donors, samples are weighted by an exponential kernel on the cosine distance
(x 100) to the instance, and a weighted ridge regression gives one weight per component.
The surrogate's weighted R^2 (`score`) is kept: it tells how well a straight line fits the model
locally and is used in the disagreement analysis (Step 10).
"""

from __future__ import annotations

import time

import numpy as np

from xaicomp.explainers.base import Explanation


def explain_lime(view, seed: int, budget: dict, cfg: dict, target: str = "high") -> Explanation:
    from lime.lime_base import LimeBase

    F = view.n_features
    n = int(budget["num_samples"])
    width = float(cfg.get("kernel_width", 25.0))
    rng = np.random.default_rng([seed, 22])
    n0 = view.n_evals
    t0 = time.perf_counter()

    masks = np.ones((n, F), dtype=bool)
    for i, k in enumerate(rng.integers(1, F + 1, size=n - 1), start=1):
        masks[i, rng.choice(F, size=k, replace=False)] = False
    y = view.value(masks, "donor", rng)

    frac_present = masks.sum(axis=1) / F
    distances = (1.0 - np.sqrt(frac_present)) * 100.0               # cosine distance to all-ones, x 100

    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / width ** 2))

    lb = LimeBase(kernel, random_state=seed)
    intercept, exp, score, local_pred = lb.explain_instance_with_data(
        masks.astype(float), np.c_[1 - y, y], distances, label=1, num_features=F, feature_selection="none")
    values = np.zeros(F)
    for idx, w in exp:
        values[idx] = w
    fx = view.fx()
    return Explanation(method="lime", instance_id=view.instance_id, seed=seed, values=values, fx=fx,
                       base_value=float(intercept), n_evals=view.n_evals - n0, seconds=time.perf_counter() - t0,
                       target=target,
                       extras={"surrogate_r2": float(score), "local_pred": float(np.ravel(local_pred)[0]),
                               "num_samples": n, "kernel_width": width, "k_donors": view.k_donors})
