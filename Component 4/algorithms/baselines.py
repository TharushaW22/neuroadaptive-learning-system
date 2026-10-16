"""Four real baselines (fixed rules, random, cognitive-only, personal-only)."""
from __future__ import annotations
import numpy as np
from algorithms.base import PolicyAlgorithm
from algorithms.linucb import LinUCB
from schemas import (Strategy, CognitiveState, StudentTraits, LearningStyle,
                     Interest, Goal, Prior, Motivation, Emotion)
from expert_rules import expert_strategy, expert_prompt

STRATEGIES = [s.value for s in Strategy]


class FixedRules(PolicyAlgorithm):
    name = "fixed_rules"

    def select(self, context: np.ndarray):
        state, traits = self._unpack(context)
        s = expert_strategy(state, traits).value
        return s, expert_prompt(Strategy(s))

    def update(self, *_, **__):
        return None

    @staticmethod
    def _unpack(ctx: np.ndarray):
        import student_profile as st_profile
        state = CognitiveState(
            attention=float(ctx[0]), fatigue=float(ctx[1]),
            confusion=float(ctx[2]), readiness=float(ctx[3]),
            workload=float(ctx[4]),
        )
        i = 5
        style = LearningStyle(st_profile.STYLE_ORDER[int(np.argmax(ctx[i:i+4]))]); i += 4
        interest = Interest(st_profile.INTEREST_ORDER[int(np.argmax(ctx[i:i+6]))]); i += 6
        emotion = Emotion(st_profile.EMOTION_ORDER[int(np.argmax(ctx[i:i+5]))]); i += 5
        goal_s, prior_s, motiv_s = float(ctx[i]), float(ctx[i+1]), float(ctx[i+2])
        goal = min(st_profile.GOAL_SCALAR, key=lambda k: abs(st_profile.GOAL_SCALAR[k] - goal_s))
        prior = min(st_profile.PRIOR_SCALAR, key=lambda k: abs(st_profile.PRIOR_SCALAR[k] - prior_s))
        motiv = min(st_profile.MOTIVATION_SCALAR, key=lambda k: abs(st_profile.MOTIVATION_SCALAR[k] - motiv_s))
        traits = StudentTraits(style=style, interest=interest,
                               goal=Goal(goal), prior=Prior(prior),
                               motivation=Motivation(motiv), emotion=emotion)
        return state, traits


class RandomPolicy(PolicyAlgorithm):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, context):
        s = str(self.rng.choice(STRATEGIES))
        return s, expert_prompt(Strategy(s))

    def update(self, *_, **__):
        return None


class CognitiveOnlyBandit(LinUCB):
    name = "cognitive_only"

    def __init__(self, alpha=0.6, seed=0):
        super().__init__(dim=8, alpha=alpha, seed=seed)  # 5 cog + 3 dummy


class PersonalOnlyBandit(LinUCB):
    name = "personal_only"

    def __init__(self, alpha=0.6, seed=0):
        super().__init__(dim=18, alpha=alpha, seed=seed)  # 18 traits