"""Bandit 1 (strategy) + Bandit 2 (prompt) + online learning + persistence."""
from __future__ import annotations
import os, pickle
import numpy as np
from schemas import Strategy, PromptTemplate, CognitiveState, StudentTraits
import student_profile as st_profile
from expert_rules import expert_prompt

STRATEGY_LIST = [s.value for s in Strategy]
PROMPT_LIST = [p.value for p in PromptTemplate]
PROMPT_CONTEXT_DIM = st_profile.CONTEXT_DIM + len(STRATEGY_LIST)


class Bandit:
    def __init__(self, arms, dim, alpha=0.6, seed=0):
        self.arms, self.dim, self.alpha = arms, dim, alpha
        self.rng = np.random.default_rng(seed)
        self.A = {a: np.eye(dim) for a in arms}
        self.b = {a: np.zeros(dim) for a in arms}

    def _ucb(self, x):
        out = {}
        for a in self.arms:
            A_inv = np.linalg.inv(self.A[a])
            mean = float((A_inv @ self.b[a]) @ x)
            bonus = self.alpha * float(np.sqrt(x @ A_inv @ x))
            out[a] = mean + bonus
        return out

    def select(self, x):
        s = self._ucb(x)
        best = max(s.values())
        ties = [a for a, v in s.items() if abs(v - best) < 1e-12]
        return str(self.rng.choice(ties))

    def update(self, x, arm, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class PolicyService:
    def __init__(self, seed: int = 0, state_path: str = "results/policy_state.pkl"):
        self.seed = seed
        self.state_path = state_path
        self.bandit1 = Bandit(STRATEGY_LIST, st_profile.CONTEXT_DIM, alpha=0.6, seed=seed)
        self.bandit2 = Bandit(PROMPT_LIST, PROMPT_CONTEXT_DIM, alpha=0.6, seed=seed + 1)
        self.trace = []
        self.load()

    def decide(self, state: CognitiveState, traits: StudentTraits):
        ctx = st_profile.build_context(state, traits)
        strategy = self.bandit1.select(ctx)
        ctx2 = np.concatenate([ctx, self._one_hot(strategy)])
        prompt = self.bandit2.select(ctx2)
        rec = {"strategy": strategy, "prompt": prompt,
               "context": ctx.tolist(),
               "strategy_scores": self.bandit1._ucb(ctx),
               "prompt_scores": self.bandit2._ucb(ctx2),
               "rationale": f"Bandit1 chose {strategy} (highest UCB); "
                            f"Bandit2 chose {prompt} given strategy."}
        self.trace.append({"type": "decide", **rec})
        return rec

    def feedback(self, state, traits, strategy, prompt, learning_gain):
        ctx = st_profile.build_context(state, traits)
        ctx2 = np.concatenate([ctx, self._one_hot(strategy)])
        self.bandit1.update(ctx, strategy, learning_gain)
        self.bandit2.update(ctx2, prompt, learning_gain)
        self.trace.append({"type": "feedback", "strategy": strategy,
                           "prompt": prompt, "reward": learning_gain})
        self.save()

    def _one_hot(self, s):
        return np.array([1.0 if a == s else 0.0 for a in STRATEGY_LIST], dtype=np.float32)

    def save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "wb") as f:
            pickle.dump({"b1": self.bandit1, "b2": self.bandit2,
                         "trace": self.trace}, f)

    def load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    d = pickle.load(f)
                self.bandit1, self.bandit2, self.trace = d["b1"], d["b2"], d["trace"]
            except Exception as e:
                print(f"[WARN] policy state load failed: {e}")