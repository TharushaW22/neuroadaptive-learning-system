"""Compare the 4 algorithms — TRAINED on the policy dataset, then tested."""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd
from algorithms.linucb import LinUCB
from algorithms.qlearning import QLearning
from algorithms.bkt import BKT
from algorithms.topsis import TOPSIS
import student_profile as st_profile
from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion)
from reward_simulator import STRATEGY_LIST

SEEDS = list(range(10))
N_EPOCHS = 5           # multiple passes over the training set
TRAIN_FRACTION = 0.8
DATA_PATH = "data/policy_dataset.csv"


def _make_algorithms(dim=23, seed=0):
    return {
        "linucb":    LinUCB(dim=dim, alpha=0.6, seed=seed),
        "qlearning": QLearning(seed=seed),
        "bkt":       BKT(seed=seed),
        "topsis":    TOPSIS(seed=seed),
    }


def _row_to_state_traits(row):
    state = CognitiveState(
        attention=float(row["attention"]), fatigue=float(row["fatigue"]),
        confusion=float(row["confusion"]), readiness=float(row["readiness"]),
        workload=float(row["workload"]))
    traits = StudentTraits(
        style=LearningStyle(row["style"]),
        interest=Interest(row["interest"]),
        goal=Goal(row["goal"]),
        prior=Prior(row["prior"]),
        motivation=Motivation(row["motivation"]),
        emotion=Emotion(row["emotion"]))
    return state, traits


def _train(algo, train_df, n_epochs=N_EPOCHS):
    """Train the algorithm on the policy dataset.
    Reward is looked up from the pre-computed reward table in the dataset."""
    for _ in range(n_epochs):
        for _, row in train_df.iterrows():
            st, tr = _row_to_state_traits(row)
            ctx = st_profile.build_context(st, tr)
            s_chosen, _ = algo.select(ctx)
            reward = float(row[f"reward_{s_chosen}"])
            algo.update(ctx, s_chosen, reward)


def _evaluate(algo, test_df, update_online=False):
    """Evaluate on held-out data. Returns accuracy, mean gain, mean latency."""
    acc_hits, gains, latencies = 0, [], []
    for _, row in test_df.iterrows():
        st, tr = _row_to_state_traits(row)
        ctx = st_profile.build_context(st, tr)
        t0 = time.perf_counter()
        s_chosen, _ = algo.select(ctx)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        gt = row["expert_strategy"]
        acc_hits += int(s_chosen == gt)
        reward = float(row[f"reward_{s_chosen}"])
        gains.append(reward)

        if update_online:
            algo.update(ctx, s_chosen, reward)

    n = len(test_df)
    return acc_hits / n, float(np.mean(gains)), float(np.mean(latencies))


def _weighted_score(acc, gain, latency_ms, w=(0.6, 0.3, 0.1), lat_cap_ms=50.0):
    lat_score = max(0.0, 1.0 - latency_ms / lat_cap_ms)
    return w[0] * acc + w[1] * gain + w[2] * lat_score


def run(save_dir="results"):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run policy_dataset first.")

    df = pd.read_csv(DATA_PATH)
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for seed in SEEDS:
        # Same split for all algorithms so comparison is fair
        train_df = df.sample(frac=TRAIN_FRACTION, random_state=seed)
        test_df = df.drop(train_df.index)

        for algo_name, algo in _make_algorithms(seed=seed).items():
            # 1. TRAIN
            _train(algo, train_df)
            # 2. EVALUATE (frozen policy, no online updates during test)
            acc, gain, lat = _evaluate(algo, test_df, update_online=False)

            rows.append({
                "seed": seed,
                "algorithm": algo_name,
                "accuracy": acc,
                "learning_gain": gain,
                "latency_ms": lat,
            })

    df_out = pd.DataFrame(rows)

    agg = df_out.groupby("algorithm").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        gain_mean=("learning_gain", "mean"),
        gain_std=("learning_gain", "std"),
        lat_mean=("latency_ms", "mean"),
        lat_std=("latency_ms", "std"),
    ).reset_index()

    agg["score"] = [
        _weighted_score(a, g, l)
        for a, g, l in zip(agg["accuracy_mean"], agg["gain_mean"], agg["lat_mean"])
    ]
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    winner = agg.loc[0, "algorithm"]

    df_out.to_csv(os.path.join(save_dir, "comparison_raw.csv"), index=False)
    agg.to_csv(os.path.join(save_dir, "comparison_summary.csv"), index=False)
    with open(os.path.join(save_dir, "comparison_winner.json"), "w") as f:
        json.dump({
            "winner": winner,
            "selection_rule": "0.6*accuracy + 0.3*gain + 0.1*(1-latency/50ms)",
            "seeds": SEEDS,
            "train_fraction": TRAIN_FRACTION,
            "epochs": N_EPOCHS,
        }, f, indent=2)

    print("[SIMULATED comparison — algorithms TRAINED before evaluation]")
    print(agg.to_string(index=False))
    print(f"\nSelected algorithm (computed from metrics): {winner}")
    return {"winner": winner, "summary": agg, "raw": df_out}


if __name__ == "__main__":
    run()