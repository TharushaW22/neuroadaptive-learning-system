"""FastAPI service exposing /decide, /feedback, /trace."""
from fastapi import FastAPI
from pydantic import BaseModel
from schemas import CognitiveState, StudentTraits
from policy_service import PolicyService

app = FastAPI(title="MD-AP2L")
svc = PolicyService()


class DecideIn(BaseModel):
    state: CognitiveState
    traits: StudentTraits


class FeedbackIn(BaseModel):
    state: CognitiveState
    traits: StudentTraits
    strategy: str
    prompt: str
    learning_gain: float


@app.post("/decide")
def decide(payload: DecideIn):
    return svc.decide(payload.state, payload.traits)


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    svc.feedback(payload.state, payload.traits, payload.strategy,
                 payload.prompt, payload.learning_gain)
    return {"ok": True}


@app.get("/trace")
def trace():
    return {"trace": svc.trace}