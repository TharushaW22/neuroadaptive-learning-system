"""Disjoint LinUCB for 7 strategies."""
from __future__ import annotations
import numpy as np
from algorithms.base import PolicyAlgorithm
from schemas import Strategy
from expert_rules import expert_prompt

STRATEGIES = [s.value for s in Strategy]


class LinUCB(PolicyAlgorithm):
    name = "linucb"

    def __init__(self, dim: int, alpha: float = 0.6, seed: int = 0):
        self.dim = dim
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.A = {s: np.eye(dim) for s in STRATEGIES}
        self.b = {s: np.zeros(dim) for s in STRATEGIES}
        self.t = 0

    def _ucb(self, context: np.ndarray) -> dict:
        self.t += 1
        scores = {}
        for s in STRATEGIES:
            A_inv = np.linalg.inv(self.A[s])
            theta = A_inv @ self.b[s]
            mean = float(theta @ context)
            bonus = self.alpha * float(np.sqrt(context @ A_inv @ context))
            scores[s] = mean + bonus
        return scores

    def select(self, context: np.ndarray):
        scores = self._ucb(context)
        best = max(scores.values())
        ties = [s for s, v in scores.items() if abs(v - best) < 1e-12]
        chosen = str(self.rng.choice(ties))
        return chosen, expert_prompt(Strategy(chosen))

    def update(self, context: np.ndarray, strategy: str, reward: float):
        self.A[strategy] += np.outer(context, context)
        self.b[strategy] += reward * context

    def scores(self, context: np.ndarray):
        return self._ucb(context)