"""Cost: wall-clock seconds and model evaluations per explanation.

Seconds are measured inside one single-threaded worker process, so methods are timed under the
same conditions. Model evaluations count forward passes (IG: forward + backward counted as ~3
forward-equivalents per step), which is machine-independent.
"""

from __future__ import annotations

import numpy as np


def cost_summary(seconds: list[float], n_evals: list[int]) -> dict:
    return {"cost_seconds": float(np.mean(seconds)), "cost_evals": float(np.mean(n_evals))}


def ms_per_eval(seconds: list[float], n_evals: list[int]) -> float:
    s, n = float(np.sum(seconds)), float(np.sum(n_evals))
    return 1000 * s / n if n else float("nan")
