"""Generate the 2000+ policy-learner dataset (SIMULATED)."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion, Strategy)
from expert_rules import expert_strategy, expert_prompt
from reward_simulator import simulate_learning_gain, STRATEGY_LIST

SEED = 42
N_SAMPLES = 2200
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "policy_dataset.csv")


def _rand_cog(rng):
    return CognitiveState(
        attention=float(np.clip(rng.beta(2, 2), 0, 1)),
        fatigue=float(np.clip(rng.beta(2, 3), 0, 1)),
        confusion=float(np.clip(rng.beta(2, 3), 0, 1)),
        readiness=float(np.clip(rng.beta(2, 2), 0, 1)),
        workload=float(np.clip(rng.beta(2, 2.5), 0, 1)),
    )


def _rand_traits(rng):
    return StudentTraits(
        style=LearningStyle(rng.choice([e.value for e in LearningStyle])),
        interest=Interest(rng.choice([e.value for e in Interest])),
        goal=Goal(rng.choice([e.value for e in Goal])),
        prior=Prior(rng.choice([e.value for e in Prior])),
        motivation=Motivation(rng.choice([e.value for e in Motivation])),
        emotion=Emotion(rng.choice([e.value for e in Emotion])),
    )


def generate(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        state = _rand_cog(rng)
        traits = _rand_traits(rng)
        gt = expert_strategy(state, traits)
        gt_prompt = expert_prompt(gt)
        noisy = rng.random() < 0.10
        if noisy:
            alt = list(set(STRATEGY_LIST) - {gt.value})
            gt = Strategy(rng.choice(alt))
            gt_prompt = expert_prompt(gt)

        rewards = {s: simulate_learning_gain(s, state, traits, rng) for s in STRATEGY_LIST}
        best_measured = max(rewards, key=rewards.get)

        row = {
            "example_id": i,
            "attention": state.attention, "fatigue": state.fatigue,
            "confusion": state.confusion, "readiness": state.readiness,
            "workload": state.workload,
            "style": traits.style.value, "interest": traits.interest.value,
            "goal": traits.goal.value, "prior": traits.prior.value,
            "motivation": traits.motivation.value, "emotion": traits.emotion.value,
            "expert_strategy": gt.value,
            "expert_prompt": gt_prompt,
            "noisy_label": noisy,
            "best_measured_strategy": best_measured,
        }
        for s in STRATEGY_LIST:
            row[f"reward_{s}"] = rewards[s]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    meta = {"n": len(df), "seed": SEED, "label_noise_rate": 0.10,
            "source": "SIMULATED"}
    with open(os.path.join(OUT_DIR, "policy_dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote {OUT_PATH}  n={len(df)}  (SIMULATED)")


if __name__ == "__main__":
    main()