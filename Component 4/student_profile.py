"""Student Profile Module — builds the multi-dimensional context vector.

Encoding (SO1):
  Cognitive states (5) : scalars in [0,1]
  style      (nominal) : one-hot, 4 dims
  interest   (nominal) : one-hot, 6 dims
  emotion    (nominal) : one-hot, 5 dims
  goal       (ordinal) : scalar [0,1]
  prior      (ordinal) : scalar [0,1]
  motivation (ordinal) : scalar [0,1]
Total: 5 + 4 + 6 + 5 + 3 = 23
"""
from __future__ import annotations
import numpy as np
from schemas import CognitiveState, StudentTraits

COGNITIVE_ORDER = ["attention", "fatigue", "confusion", "readiness", "workload"]
STYLE_ORDER = ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
INTEREST_ORDER = ["Sports", "Music", "Gaming", "Movies", "Technology", "Nature"]
EMOTION_ORDER = ["Confident", "Anxious", "Bored", "Tired", "Excited"]

GOAL_SCALAR = {"Exam Preparation": 0.3, "Deep Understanding": 1.0, "Quick Overview": 0.0}
PRIOR_SCALAR = {"Beginner": 0.0, "Intermediate": 0.5, "Advanced": 1.0}
MOTIVATION_SCALAR = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

CONTEXT_DIM = 5 + len(STYLE_ORDER) + len(INTEREST_ORDER) + len(EMOTION_ORDER) + 3


def _one_hot(value: str, order: list[str]) -> list[float]:
    if value not in order:
        raise ValueError(f"Unknown category '{value}'. Expected one of {order}.")
    return [1.0 if v == value else 0.0 for v in order]


def build_context(state: CognitiveState, traits: StudentTraits) -> np.ndarray:
    cog = [getattr(state, k) for k in COGNITIVE_ORDER]
    style_oh = _one_hot(traits.style.value, STYLE_ORDER)
    interest_oh = _one_hot(traits.interest.value, INTEREST_ORDER)
    emotion_oh = _one_hot(traits.emotion.value, EMOTION_ORDER)
    goal_s = GOAL_SCALAR[traits.goal.value]
    prior_s = PRIOR_SCALAR[traits.prior.value]
    motiv_s = MOTIVATION_SCALAR[traits.motivation.value]
    vec = np.array(cog + style_oh + interest_oh + emotion_oh + [goal_s, prior_s, motiv_s],
                   dtype=np.float32)
    assert vec.shape[0] == CONTEXT_DIM
    return vec


def build_cognitive_only(state: CognitiveState) -> np.ndarray:
    return np.array([getattr(state, k) for k in COGNITIVE_ORDER], dtype=np.float32)


def build_traits_only(traits: StudentTraits) -> np.ndarray:
    return np.array(
        _one_hot(traits.style.value, STYLE_ORDER)
        + _one_hot(traits.interest.value, INTEREST_ORDER)
        + _one_hot(traits.emotion.value, EMOTION_ORDER)
        + [GOAL_SCALAR[traits.goal.value],
           PRIOR_SCALAR[traits.prior.value],
           MOTIVATION_SCALAR[traits.motivation.value]],
        dtype=np.float32,
    )