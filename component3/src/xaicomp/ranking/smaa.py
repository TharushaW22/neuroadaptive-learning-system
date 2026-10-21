"""Monte-Carlo SMAA (Lahdelma & Salminen) over TOPSIS: how often is each method ranked first?

Nobody has to choose the criterion weights. In every draw:
  * weights ~ Dirichlet(alpha, ..., alpha) (alpha = 1: uniform over all possible weightings),
    multiplied by each criterion's reliability factor, renormalised
  * criterion values = one cluster-bootstrap resample of the subjects (measurement uncertainty)
  * TOPSIS ranks the methods
Outputs: rank acceptability b[m, r] = P(method m gets rank r); P(rank 1) = b[m, 0];
central weights = mean weight vector of the draws in which m ranks first.
Repeated with several Monte-Carlo seeds; the max difference in P(rank 1) shows the MC error.
"""

from __future__ import annotations

import numpy as np

from xaicomp.ranking.topsis import topsis


def smaa(V_boot: np.ndarray, factors: np.ndarray, benefit: np.ndarray, n_draws: int, alpha: float,
         seed: int) -> dict:
    """V_boot: (n_boot, M, C) scaled criterion values; factors: (C,) reliability factors."""
    rng = np.random.default_rng(seed)
    M, C = V_boot.shape[1], V_boot.shape[2]
    w = rng.dirichlet(np.full(C, alpha), size=n_draws) * factors[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    V = V_boot[rng.integers(0, len(V_boot), size=n_draws)]
    V = np.where(np.isnan(V), np.nanmean(V, axis=0, keepdims=True), V)
    cl = topsis(V, w, benefit)                                       # (D, M)
    jitter = rng.random(cl.shape) * 1e-12                            # random tie-breaking
    order = np.argsort(-(cl + jitter), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n_draws)[:, None], order] = np.arange(M)[None, :]
    acc = np.stack([(ranks == r).mean(axis=0) for r in range(M)], axis=1)   # (M, ranks)
    central = np.array([w[ranks[:, m] == 0].mean(axis=0) if (ranks[:, m] == 0).any() else np.full(C, np.nan)
                        for m in range(M)])
    return {"rank_acceptability": acc, "p_rank1": acc[:, 0], "central_weights": central,
            "mean_closeness": cl.mean(axis=0), "expected_rank": (ranks + 1).mean(axis=0)}


def smaa_multi_seed(V_boot, factors, benefit, n_draws, alpha, seeds: list[int]) -> dict:
    runs = [smaa(V_boot, factors, benefit, n_draws, alpha, s) for s in seeds]
    p1 = np.stack([r["p_rank1"] for r in runs])
    out = runs[0]
    out["p_rank1_by_seed"] = p1
    out["p_rank1"] = p1.mean(axis=0)
    out["rank_acceptability"] = np.mean([r["rank_acceptability"] for r in runs], axis=0)
    out["mc_max_abs_diff"] = float((p1.max(axis=0) - p1.min(axis=0)).max())
    return out
