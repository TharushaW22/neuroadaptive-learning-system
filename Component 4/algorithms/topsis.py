"""TOPSIS multi-criteria ranker (static — no learning)."""
from __future__ import annotations
import numpy as np
from algorithms.base import PolicyAlgorithm
from schemas import Strategy
from expert_rules import expert_prompt

STRATEGIES = [s.value for s in Strategy]
WEIGHTS = np.array([0.45, 0.35, 0.20])


def _cog_fit(strategy: str, cog: np.ndarray) -> float:
    a, f, c, r, w = cog
    return {
        "simplify":       max(c, f) * 0.9,
        "analogy":        c * 0.9 + 0.05,
        "worked_example": w * 0.9 + 0.05,
        "hint":           0.4 + 0.3 * (1 - c),
        "quiz":           r * a,
        "visual":         (1 - a) * 0.9 + 0.05,
        "break":          f * 0.95,
    }[strategy]


def _prompt_affordance(strategy: str) -> float:
    return {
        "simplify": 0.8, "analogy": 0.8, "worked_example": 0.9,
        "hint": 0.6, "quiz": 0.7, "visual": 0.85, "break": 0.5,
    }[strategy]


class TOPSIS(PolicyAlgorithm):
    name = "topsis"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._last_cc = {}

    def select(self, context: np.ndarray):
        cog = context[:5]
        prior = context[-2]
        style_visual = context[5] == 1.0

        scores = {}
        for s in STRATEGIES:
            c1 = _cog_fit(s, cog)
            c2 = 0.4
            if s == "visual" and style_visual:
                c2 = 0.95
            elif s == "quiz" and prior >= 0.99:
                c2 = 0.85
            elif s == "simplify" and prior <= 0.01:
                c2 = 0.85
            c3 = _prompt_affordance(s)
            scores[s] = np.array([c1, c2, c3])

        M = np.stack([scores[s] for s in STRATEGIES])
        norm = np.sqrt((M ** 2).sum(axis=0))
        R = M / np.maximum(norm, 1e-9)
        V = R * WEIGHTS
        ideal = V.max(axis=0)
        anti = V.min(axis=0)
        d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
        d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
        cc = d_neg / np.maximum(d_pos + d_neg, 1e-9)
        order = np.argsort(-cc)
        top = STRATEGIES[int(order[0])]
        self._last_cc = {STRATEGIES[i]: float(cc[i]) for i in range(len(STRATEGIES))}
        return top, expert_prompt(Strategy(top))

    def update(self, context: np.ndarray, strategy: str, reward: float) -> None:
        return None  # static ranker

    def scores(self, context: np.ndarray):
        self.select(context)
        return self._last_cc