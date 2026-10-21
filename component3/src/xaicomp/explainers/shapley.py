"""KernelSHAP and Conditional SHAP: the same estimator, two different value functions.

Both use shap.KernelExplainer on the BINARY coalition space: the instance is the all-ones vector
(every component present), the background is the all-zeros vector (every component absent), and
the "model" handed to shap is the value function v(S) of the adapter, which rebuilds the EEG window
with the absent components replaced. shap therefore only does coalition sampling + the weighted
least-squares fit (Lundberg & Lee, 2017); the meaning of "absent" is ours:

  KernelSHAP         v(S) = mean over K donors of f(x_S, donor_notS)                 (marginal)
  Conditional SHAP   v(S) = mean over K draws of f(x_S, donor waveform rescaled to a
                     sample of the absent log band powers | the present ones)     (Gaussian conditional, D2)

Because the estimator and the coalitions (same seed) are shared, any difference between the two is
the effect of the value function (analysed in Step 10).
"""

from __future__ import annotations

import time

import numpy as np

from xaicomp.explainers.base import Explanation


def kernel_shap_on(view, replacement: str, nsamples: int, seed: int, method: str, target: str = "high") -> Explanation:
    import shap

    F = view.n_features
    rng = np.random.default_rng([seed, 11])
    n0 = view.n_evals
    t0 = time.perf_counter()

    def v(M: np.ndarray) -> np.ndarray:
        return view.value(np.asarray(M) > 0.5, replacement, rng)

    np.random.seed(seed)                       # shap's coalition sampler uses NumPy's global RNG
    ex = shap.KernelExplainer(v, np.zeros((1, F)))
    phi = np.asarray(ex.shap_values(np.ones((1, F)), nsamples=nsamples, l1_reg=False, silent=True)).reshape(-1)
    base = float(np.ravel(ex.expected_value)[0])
    fx = view.fx()
    seconds = time.perf_counter() - t0
    eff = float(abs(phi.sum() - (fx - base)))
    return Explanation(method=method, instance_id=view.instance_id, seed=seed, values=phi, fx=fx, base_value=base,
                       n_evals=view.n_evals - n0, seconds=seconds, target=target,
                       extras={"efficiency_error": eff, "replacement": replacement, "nsamples": nsamples,
                               "k_donors": view.k_donors})


def explain_kernel_shap(view, seed: int, budget: dict, cfg: dict, target: str = "high") -> Explanation:
    return kernel_shap_on(view, "donor", int(budget["nsamples"]), seed, "kernel_shap", target)


def explain_conditional_shap(view, seed: int, budget: dict, cfg: dict, target: str = "high") -> Explanation:
    replacement = {"gaussian": "conditional", "empirical": "empirical"}[cfg.get("mode", "gaussian")]
    return kernel_shap_on(view, replacement, int(budget["nsamples"]), seed, "conditional_shap", target)
