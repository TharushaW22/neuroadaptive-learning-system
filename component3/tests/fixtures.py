"""Tiny shape-only TEST FIXTURES (used only inside tests/, never written to results/).

ToyView mimics an InstanceView with a KNOWN additive model, so metric code can be checked
against ground truth: output = sigmoid(sum_f w_f * present_f + bias). The true importance of
feature f is w_f.
"""

from __future__ import annotations

import numpy as np


class ToyView:
    def __init__(self, weights: np.ndarray, bias: float = 0.0, instance_id: str = "toy", seed: int = 0):
        self.w = np.asarray(weights, dtype=float)
        self.bias = bias
        self.n_features = len(self.w)
        self.instance_id = instance_id
        self.seed = seed
        self.n_evals = 0
        self.k_donors = 1

    def _f(self, masks):
        z = np.asarray(masks, float) @ self.w + self.bias
        return 1 / (1 + np.exp(-z))

    def value(self, masks, replacement="zero", rng=None, **kw):
        masks = np.atleast_2d(masks)
        self.n_evals += len(masks)
        return self._f(masks)

    def fx(self):
        return float(self._f(np.ones((1, self.n_features)))[0])


def toy_weights(n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)
