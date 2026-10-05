"""System Usability Scale scorer (10-item, 5-point)."""
from __future__ import annotations
import numpy as np


def score(responses):
    if len(responses) != 10:
        raise ValueError("SUS requires exactly 10 responses.")
    r = np.array(responses, float)
    odd = r[[0, 2, 4, 6, 8]] - 1
    even = 5 - r[[1, 3, 5, 7, 9]]
    return float((odd.sum() + even.sum()) * 2.5)