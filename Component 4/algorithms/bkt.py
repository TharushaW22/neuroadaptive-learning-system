"""Bayesian Knowledge Tracing with documented strategy mapping."""
from __future__ import annotations
import numpy as np
from algorithms.base import PolicyAlgorithm
from schemas import Strategy
from expert_rules import expert_prompt

STRATEGIES = [s.value for s in Strategy]


class BKT(PolicyAlgorithm):
    name = "bkt"

    def __init__(self, p_init=0.25, p_learn=0.15, p_slip=0.10, p_guess=0.20, seed=0):
        self.p_init, self.p_learn = p_init, p_learn
        self.p_slip, self.p_guess = p_slip, p_guess
        self.mastery = p_init
        self.rng = np.random.default_rng(seed)

    def _update_mastery(self, correct: bool):
        pL = self.mastery
        if correct:
            num = pL * (1 - self.p_slip)
            den = num + (1 - pL) * self.p_guess
        else:
            num = pL * self.p_slip
            den = num + (1 - pL) * (1 - self.p_guess)
        pL_given_obs = num / max(den, 1e-9)
        self.mastery = pL_given_obs + (1 - pL_given_obs) * self.p_learn

    def select(self, context: np.ndarray):
        m = self.mastery
        fatigue, confusion = context[1], context[2]
        if fatigue >= 0.7:
            s = "break"
        elif m < 0.30 and confusion >= 0.65:
            s = "simplify"
        elif m < 0.45:
            s = "worked_example"
        elif m < 0.60:
            s = "analogy"
        elif m < 0.80:
            s = "hint"
        else:
            s = "quiz"
        return s, expert_prompt(Strategy(s))

    def update(self, context: np.ndarray, strategy: str, reward: float):
        self._update_mastery(reward >= 0.5)

    def scores(self, context: np.ndarray):
        return {s: float(self.mastery) for s in STRATEGIES}