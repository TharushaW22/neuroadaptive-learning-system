"""Robustness: does the explanation stay the same when the EEG is slightly noisy?

Noise: Gaussian, sigma = noise_scale x C1's channel scale per channel (0.1 = the noise C1 used as
training augmentation, trainer.py:94-95 / deep.yaml augment.noise_std), added to the 18-s window.
Only perturbations that do NOT flip the predicted label are scored (a different prediction is
allowed to have a different explanation). The noisy copy is explained with the SAME explainer seed
as the original, so the comparison changes only the input.

robustness = Spearman(attribution(x), attribution(x + noise))      (1 = unchanged ranking)
Interpretation: robustness cannot be expected to exceed the method's stability by much, because
run-to-run randomness is still present; both are shown together.
"""

from __future__ import annotations

import numpy as np

from xaicomp.metrics.stability import spearman, topk_jaccard


def noise_for_window(channel_scale: np.ndarray, shape: tuple[int, int], noise_scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 51])
    return (rng.standard_normal(shape) * (noise_scale * channel_scale)[:, None]).astype(np.float32)


def robustness(a: np.ndarray, a_noisy: np.ndarray, k: int = 10) -> dict:
    return {"robustness": spearman(a, a_noisy), "robustness_topk_jaccard": topk_jaccard(a, a_noisy, k)}
