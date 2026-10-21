"""Are two metrics measuring the same thing? (redundancy matrix)

Spearman correlation between criteria across all (window, method) rows. Criteria whose |rho|
exceeds the threshold form one group (connected components) and are MERGED in the ranking, so
the same property is not counted twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def redundancy_matrix(instance_method: pd.DataFrame, criteria: list[str]) -> pd.DataFrame:
    cols = [c for c in criteria if c in instance_method and instance_method[c].notna().sum() > 2
            and instance_method[c].nunique() > 1]
    return instance_method[cols].corr(method="spearman")


def redundant_groups(R: pd.DataFrame, threshold: float) -> list[list[str]]:
    names = list(R.columns)
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = R.loc[a, b]
            if v == v and abs(v) > threshold:
                parent[find(b)] = find(a)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [g for g in groups.values()]
