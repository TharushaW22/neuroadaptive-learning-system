"""Agreement between two methods on the same instance (Krishna et al., 2022, 'The disagreement problem').

  rank_corr        Spearman correlation of the full attribution vectors
  feature_agree    share of the top-k features (by |attribution|) that both methods name
  sign_agree       among the shared top-k features, share with the same sign
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from xaicomp.metrics.stability import spearman, topk_set


def agreement(a: np.ndarray, b: np.ndarray, k: int = 10) -> dict:
    A, B = topk_set(a, k), topk_set(b, k)
    common = sorted(A & B)
    sign = float(np.mean(np.sign(a[common]) == np.sign(b[common]))) if common else float("nan")
    return {"rank_corr": spearman(a, b), "feature_agree": len(common) / k, "sign_agree": sign}


def pairwise_agreement(values: dict[str, np.ndarray], k: int = 10) -> list[dict]:
    return [{"method_a": m1, "method_b": m2, **agreement(values[m1], values[m2], k)}
            for m1, m2 in combinations(sorted(values), 2)]
