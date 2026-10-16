"""SIMULATED reward simulator. Reward depends on the *chosen* action."""
from __future__ import annotations
import numpy as np
from schemas import Strategy, CognitiveState, StudentTraits

STRATEGY_LIST = [s.value for s in Strategy]


def _latent_best(state: CognitiveState, traits: StudentTraits) -> str:
    if state.fatigue >= 0.70:
        return "break"
    if state.workload >= 0.65:
        return "worked_example"
    if state.confusion >= 0.65:
        return "simplify" if traits.prior.value == "Beginner" else "analogy"
    if state.attention <= 0.35:
        return "visual"
    if traits.style.value == "Visual":
        return "visual"
    if state.readiness >= 0.70 and traits.prior.value == "Advanced":
        return "quiz"
    return "hint"


def _mismatch_penalty(chosen: str, state: CognitiveState, traits: StudentTraits) -> float:
    p = 0.0
    if chosen == "quiz" and (state.confusion >= 0.65 or state.fatigue >= 0.70):
        p += 0.35
    if chosen != "break" and state.fatigue >= 0.80:
        p += 0.30
    if chosen == "worked_example" and state.attention <= 0.25:
        p += 0.15
    if chosen == "simplify" and traits.prior.value == "Advanced":
        p += 0.10
    return p


def simulate_learning_gain(chosen, state, traits, rng, noise_scale=0.06):
    if chosen not in STRATEGY_LIST:
        raise ValueError(f"Unknown strategy '{chosen}'")
    best = _latent_best(state, traits)
    base = 0.55
    if chosen == best:
        base += 0.30
    else:
        families = {
            "break": {"break"}, "simplify": {"simplify", "hint"},
            "analogy": {"analogy", "visual"}, "worked_example": {"worked_example"},
            "visual": {"visual", "analogy"}, "hint": {"hint", "simplify"},
            "quiz": {"quiz"},
        }
        if chosen in families.get(best, set()) - {best}:
            base += 0.10
        else:
            base -= 0.15
    base -= _mismatch_penalty(chosen, state, traits)
    return float(np.clip(base + rng.normal(0.0, noise_scale), 0.0, 1.0))