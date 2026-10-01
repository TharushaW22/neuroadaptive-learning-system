"""Ground-truth expert rules (single source of truth)."""
from schemas import Strategy

T_FATIGUE_HIGH = 0.70
T_CONFUSION_HIGH = 0.65
T_WORKLOAD_HIGH = 0.65
T_ATTENTION_LOW = 0.35
T_ATTENTION_HIGH = 0.70
T_READINESS_HIGH = 0.70


def expert_strategy(state, traits) -> Strategy:
    if state.fatigue >= T_FATIGUE_HIGH:
        return Strategy.BREAK
    if state.workload >= T_WORKLOAD_HIGH and state.fatigue < T_FATIGUE_HIGH:
        return Strategy.WORKED_EXAMPLE
    if state.confusion >= T_CONFUSION_HIGH:
        if traits.prior.value == "Beginner":
            return Strategy.SIMPLIFY
        return Strategy.ANALOGY
    if state.attention <= T_ATTENTION_LOW:
        return Strategy.VISUAL
    if traits.style.value == "Visual":
        return Strategy.VISUAL
    if (state.readiness >= T_READINESS_HIGH
            and state.attention >= T_ATTENTION_HIGH
            and traits.prior.value == "Advanced"):
        return Strategy.QUIZ
    return Strategy.HINT


def expert_prompt(strategy: Strategy) -> str:
    return {
        Strategy.SIMPLIFY: "Simplified",
        Strategy.ANALOGY: "Analogy-Based",
        Strategy.WORKED_EXAMPLE: "Step-by-Step",
        Strategy.HINT: "Question-Based",
        Strategy.QUIZ: "Question-Based",
        Strategy.VISUAL: "Visual Description",
        Strategy.BREAK: "Simplified",
    }[strategy]