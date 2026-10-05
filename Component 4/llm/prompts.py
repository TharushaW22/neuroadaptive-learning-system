"""System prompts built from strategy + prompt template + student context."""
SYSTEM = ("You are a biology tutor. Follow the requested teaching strategy "
          "strictly. Never reveal answers for a 'hint' strategy. Keep quizzes "
          "as questions. Keep breaks short.")

PER_STRATEGY_RULES = {
    "simplify":       "Use one short paragraph (<=90 words).",
    "analogy":        "Use a real-world analogy; the word 'like' must appear.",
    "worked_example": "Show numbered steps (Step 1, Step 2, ...).",
    "hint":           "Give only a hint; do NOT state the answer.",
    "quiz":           "Ask 2-3 questions; do not answer them.",
    "visual":         "Describe a picture the student can imagine.",
    "break":          "Suggest a short break; include the key recap word.",
}


def build_messages(state, traits, strategy, prompt_template, topic):
    sys = f"{SYSTEM}\nStrategy: {strategy}. {PER_STRATEGY_RULES[strategy]}"
    user = (f"Student context: attention={state.attention:.2f}, "
            f"fatigue={state.fatigue:.2f}, confusion={state.confusion:.2f}, "
            f"readiness={state.readiness:.2f}, workload={state.workload:.2f}. "
            f"Traits: style={traits.style.value}, interest={traits.interest.value}, "
            f"goal={traits.goal.value}, prior={traits.prior.value}, "
            f"motivation={traits.motivation.value}, emotion={traits.emotion.value}. "
            f"Prompt template: {prompt_template}. Topic: {topic}.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]