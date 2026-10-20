"""Integrated Gradients (Sundararajan et al., 2017) with Captum, w.r.t. the band components.

Input = the stacked components (1, 14, 6, 2304); the model input is their sum + residual, so
summing IG over time gives one attribution per channel x band and completeness carries over:
    sum(attributions) = f(x) - f(baseline)            (up to the Riemann/Gauss-Legendre error)

Two baselines (D3):
  ig       MAIN: K donor windows of training subjects (expected-gradients style); attributions are
           averaged over the K donors. Stochastic through the donor draw (seed) -> stability and
           self-agreement rho(A, A') are meaningful.
  ig_zero  all components = 0 (only the residual remains): a deterministic, off-manifold reference.
           Reported separately and flagged: its stability is trivially 1 and is never rewarded.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from xaicomp.explainers.base import Explanation


def _ig(view, baselines: list[torch.Tensor], n_steps: int):
    from captum.attr import IntegratedGradients

    module, x, args = view.torch_parts()
    ig = IntegratedGradients(module)
    attrs, deltas, base_out = [], [], []
    for b in baselines:
        a, d = ig.attribute(x, baselines=b, additional_forward_args=args, n_steps=n_steps,
                            internal_batch_size=min(n_steps, 8), return_convergence_delta=True)
        attrs.append(a.detach().double().sum(dim=-1).reshape(-1).numpy())
        deltas.append(float(d.abs().item()))
        with torch.no_grad():
            base_out.append(float(module(b, *args).item()))
    with torch.no_grad():
        fx = float(module(x, *args).item())
    return np.mean(attrs, axis=0), fx, float(np.mean(base_out)), deltas


def explain_ig(view, seed: int, budget: dict, cfg: dict, target: str = "high") -> Explanation:
    baseline = cfg.get("baseline", "donor")
    n_steps = int(budget["n_steps"])
    t0 = time.perf_counter()
    if baseline == "zero":
        bases = [torch.zeros_like(torch.as_tensor(view.x[None]))]
        method = "ig_zero"
    else:
        bases = [torch.as_tensor(view.donors[k][None]) for k in range(view.k_donors)]
        method = "ig"
    values, fx, base, deltas = _ig(view, bases, n_steps)
    comp = float(abs(values.sum() - (fx - base)))
    # cost in forward-equivalents: n_steps forward + backward (~2x) per baseline, + 2 forwards
    n_evals = len(bases) * (3 * n_steps + 1) + 1
    view.n_evals += n_evals
    return Explanation(method=method, instance_id=view.instance_id, seed=seed, values=values, fx=fx, base_value=base,
                       n_evals=n_evals, seconds=time.perf_counter() - t0, target=target,
                       extras={"completeness_error": comp, "completeness_error_rel": comp / (abs(fx - base) + 1e-12),
                               "convergence_delta_max": max(deltas), "n_steps": n_steps, "baseline": baseline,
                               "n_baselines": len(bases), "deterministic": baseline == "zero"})
