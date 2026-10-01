"""Tabular Q-Learning with discretised state."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from algorithms.base import PolicyAlgorithm
from schemas import Strategy
from expert_rules import expert_prompt

STRATEGIES = [s.value for s in Strategy]
BINS = [0.33, 0.66]


def _discretise(ctx_cog: np.ndarray) -> tuple:
    return tuple(int(np.digitize(v, BINS)) for v in ctx_cog)


class QLearning(PolicyAlgorithm):
    name = "qlearning"

    def __init__(self, alpha=0.15, gamma=0.0, epsilon=0.10, seed=0):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: {s: 0.0 for s in STRATEGIES})
        self.last_state = None
        self.last_action = None

    def select(self, context: np.ndarray):
        s = _discretise(context[:5])
        self.last_state = s
        if self.rng.random() < self.epsilon:
            a = str(self.rng.choice(STRATEGIES))
        else:
            a = max(self.Q[s], key=self.Q[s].get)
        self.last_action = a
        return a, expert_prompt(Strategy(a))

    def update(self, context: np.ndarray, strategy: str, reward: float):
        s = _discretise(context[:5])
        if self.gamma == 0.0:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[s].values())
        old = self.Q[s][strategy]
        self.Q[s][strategy] = old + self.alpha * (target - old)

    def scores(self, context: np.ndarray):
        return dict(self.Q[_discretise(context[:5])])