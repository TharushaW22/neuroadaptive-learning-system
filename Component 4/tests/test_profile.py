import numpy as np
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)
import student_profile as st_profile


def test_dim():
    st_ = CognitiveState(attention=0.5, fatigue=0.5, confusion=0.5,
                         readiness=0.5, workload=0.5)
    tr = StudentTraits(style=LearningStyle.VISUAL, interest=Interest.SPORTS,
                       goal=Goal.EXAM, prior=Prior.BEGINNER,
                       motivation=Motivation.HIGH, emotion=Emotion.CONFIDENT)
    v = st_profile.build_context(st_, tr)
    assert v.shape == (st_profile.CONTEXT_DIM,)
    assert np.isclose(v[5:9].sum(), 1.0)
    assert np.isclose(v[9:15].sum(), 1.0)
    assert np.isclose(v[20:25].sum(), 1.0)