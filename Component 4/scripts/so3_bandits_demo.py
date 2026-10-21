from policy_service import PolicyService
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)

svc = PolicyService()
st_ = CognitiveState(attention=0.65, fatigue=0.70, confusion=0.80,
                     readiness=0.40, workload=0.55)
tr = StudentTraits(style=LearningStyle.VISUAL, interest=Interest.SPORTS,
                   goal=Goal.EXAM, prior=Prior.BEGINNER,
                   motivation=Motivation.HIGH, emotion=Emotion.CONFIDENT)
print(svc.decide(st_, tr))