import numpy as np
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)
from reward_simulator import simulate_learning_gain


def test_reward_depends_on_action():
    st_ = CognitiveState(attention=0.8, fatigue=0.2, confusion=0.2,
                         readiness=0.8, workload=0.3)
    tr = StudentTraits(style=LearningStyle.VISUAL, interest=Interest.SPORTS,
                       goal=Goal.DEEP, prior=Prior.ADVANCED,
                       motivation=Motivation.HIGH, emotion=Emotion.CONFIDENT)
    rng = np.random.default_rng(0)
    g_quiz  = np.mean([simulate_learning_gain("quiz",  st_, tr, rng) for _ in range(200)])
    g_break = np.mean([simulate_learning_gain("break", st_, tr, rng) for _ in range(200)])
    assert abs(g_quiz - g_break) > 0.05