"""Bayesian signed-rank test with a ROPE (Benavoli et al., 2017; package baycomp), subjects as 'datasets'.

For two methods A and B and one criterion it returns three probabilities that sum to 1:
  P(A better)   the subject-level difference is beyond +ROPE in A's favour
  P(equivalent) the difference lies inside the ROPE: 'practically the same'
  P(B better)
ROPE widths are fixed per criterion in configs/stats.yaml (chosen before seeing results).
For cost (lower is better) the values are negated first so 'better' keeps its meaning.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def signed_rank(x: np.ndarray, y: np.ndarray, rope: float, n_samples: int, seed: int) -> tuple[float, float, float]:
    from baycomp.multiple import SignedRankTest

    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    np.random.seed(seed)                          # baycomp samples from NumPy's global RNG
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.all(np.abs(d) < 1e-15):
        return 0.0, 1.0, 0.0
    p = SignedRankTest.probs(np.asarray(x, float), np.asarray(y, float), rope=rope, nsamples=n_samples)
    return float(p[0]), float(p[1]), float(p[2])


def pairwise_bayes(W: pd.DataFrame, direction: str, rope: float, n_samples: int, seed: int) -> pd.DataFrame:
    sign = 1.0 if direction == "+" else -1.0
    rows = []
    for a, b in combinations(W.columns, 2):
        pa, pr, pb = signed_rank(sign * W[a].to_numpy(), sign * W[b].to_numpy(), rope, n_samples, seed)
        verdict = ("A better" if pa >= 0.95 else "B better" if pb >= 0.95 else
                   "practically the same" if pr >= 0.95 else "undecided")
        rows.append({"method_a": a, "method_b": b, "p_a_better": pa, "p_rope": pr, "p_b_better": pb,
                     "rope": rope, "verdict": verdict, "n_subjects": len(W)})
    return pd.DataFrame(rows)
