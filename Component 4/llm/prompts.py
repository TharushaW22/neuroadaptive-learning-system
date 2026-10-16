"""System prompts built from strategy + prompt template + student context.
Now supports passing conversation history so the LLM remembers prior turns."""
from __future__ import annotations

SYSTEM = ("You are a biology tutor. Follow the requested teaching strategy "
          "strictly. Never reveal answers for a 'hint' strategy. Keep quizzes "
          "as questions. Keep breaks short. Continue the conversation the "
          "student is already having — do NOT change the topic unless the "
          "student does.")

PER_STRATEGY_RULES = {
    "simplify":       "Use one short paragraph (<=90 words).",
    "analogy":        "Use a real-world analogy; the word 'like' must appear.",
    "worked_example": "Show numbered steps (Step 1, Step 2, ...).",
    "hint":           "Give only a hint; do NOT state the answer.",
    "quiz":           "Ask 2-3 questions; do not answer them.",
    "visual":         "Describe a picture the student can imagine.",
    "break":          "Suggest a short break; include the key recap word.",
}


def build_messages(state, traits, strategy, prompt_template, topic,
                   history=None, history_window=8):
    """
    Build the LLM message list.

    messages = [system] + history (last N user/assistant turns) + [user]

    history: list of dicts, each {"role": "user"|"assistant", "content": "..."}
    """
    sys = (
        f"{SYSTEM}\n"
        f"Strategy: {strategy}. {PER_STRATEGY_RULES[strategy]}\n"
        f"Prompt template: {prompt_template}.\n"
        f"Student cognitive state: attention={state.attention:.2f}, "
        f"fatigue={state.fatigue:.2f}, confusion={state.confusion:.2f}, "
        f"readiness={state.readiness:.2f}, workload={state.workload:.2f}.\n"
        f"Student traits: style={traits.style.value}, "
        f"interest={traits.interest.value}, goal={traits.goal.value}, "
        f"prior={traits.prior.value}, motivation={traits.motivation.value}, "
        f"emotion={traits.emotion.value}.\n"
        f"When referring to the student's cognitive state, only use it to "
        f"shape HOW you teach — never make the state itself the topic."
    )

    msgs = [{"role": "system", "content": sys}]

    if history:
        for m in history[-history_window:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})

    msgs.append({"role": "user", "content": topic})
    return msgs