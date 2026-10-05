"""Simulated benchmark (SIMULATED)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)
import student_profile as st_profile
from reward_simulator import simulate_learning_gain, STRATEGY_LIST
from comparison import _make_algorithms
from algorithms.baselines import (FixedRules, RandomPolicy,
                                  CognitiveOnlyBandit, PersonalOnlyBandit)

N_EPISODES = 250
SEEDS = list(range(10))
OUT_DIR = "results"


def _sample(rng):
    st = CognitiveState(attention=float(rng.beta(2, 2)), fatigue=float(rng.beta(2, 3)),
                        confusion=float(rng.beta(2, 3)), readiness=float(rng.beta(2, 2)),
                        workload=float(rng.beta(2, 2.5)))
    tr = StudentTraits(
        style=LearningStyle(rng.choice([e.value for e in LearningStyle])),
        interest=Interest(rng.choice([e.value for e in Interest])),
        goal=Goal(rng.choice([e.value for e in Goal])),
        prior=Prior(rng.choice([e.value for e in Prior])),
        motivation=Motivation(rng.choice([e.value for e in Motivation])),
        emotion=Emotion(rng.choice([e.value for e in Emotion])))
    return st, tr


def _pick(algo_name, seed):
    return {
        "linucb":         lambda: _make_algorithms(seed=seed)["linucb"],
        "fixed_rules":    lambda: FixedRules(),
        "cognitive_only": lambda: CognitiveOnlyBandit(seed=seed),
        "personal_only":  lambda: PersonalOnlyBandit(seed=seed),
        "random":         lambda: RandomPolicy(seed=seed),
    }[algo_name]()


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for algo_name in ["linucb", "fixed_rules", "cognitive_only", "personal_only", "random"]:
        for seed in SEEDS:
            algo = _pick(algo_name, seed)
            rng = np.random.default_rng(seed)
            cumulative, gains = 0.0, []
            for ep in range(N_EPISODES):
                st, tr = _sample(rng)
                if algo_name == "cognitive_only":
                    ctx = st_profile.build_cognitive_only(st)
                elif algo_name == "personal_only":
                    ctx = st_profile.build_traits_only(tr)
                else:
                    ctx = st_profile.build_context(st, tr)
                s_chosen, _ = algo.select(ctx)
                r = simulate_learning_gain(s_chosen, st, tr, rng)
                algo.update(ctx, s_chosen, r)
                cumulative += r
                gains.append(r)
            rows.append({"algorithm": algo_name, "seed": seed,
                         "mean_gain": float(np.mean(gains)),
                         "cumulative_reward": float(cumulative)})
    df = pd.DataFrame(rows)
    summary = df.groupby("algorithm").agg(
        gain_mean=("mean_gain", "mean"), gain_std=("mean_gain", "std"),
        cum_mean=("cumulative_reward", "mean"), cum_std=("cumulative_reward", "std"),
    ).reset_index()
    df.to_csv(f"{OUT_DIR}/benchmark_raw.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/benchmark_summary.csv", index=False)
    print("[SIMULATED benchmark]")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()