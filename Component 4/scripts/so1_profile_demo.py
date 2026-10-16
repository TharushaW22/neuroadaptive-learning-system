from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)
import student_profile as st_profile

st_ = CognitiveState(attention=0.65, fatigue=0.70, confusion=0.80,
                     readiness=0.40, workload=0.55)
tr = StudentTraits(style=LearningStyle.VISUAL, interest=Interest.SPORTS,
                   goal=Goal.EXAM, prior=Prior.BEGINNER,
                   motivation=Motivation.HIGH, emotion=Emotion.CONFIDENT)
v = st_profile.build_context(st_, tr)
print("context dim:", v.shape)
print(v)