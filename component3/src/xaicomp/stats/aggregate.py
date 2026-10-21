"""Aggregate to the SUBJECT level before any test (windows of one subject are not independent)."""

from __future__ import annotations

import pandas as pd


def subject_level(instance_method: pd.DataFrame, criteria: list[str], methods: list[str]) -> pd.DataFrame:
    d = instance_method[instance_method["method"].isin(methods)]
    cols = [c for c in criteria if c in d]
    return d.groupby(["subject_id", "method"])[cols].mean().reset_index()


def wide(subject_df: pd.DataFrame, criterion: str, methods: list[str]) -> pd.DataFrame:
    """subjects x methods for one criterion; subjects with any missing method are dropped (paired tests)."""
    W = subject_df.pivot(index="subject_id", columns="method", values=criterion)
    return W[[m for m in methods if m in W]].dropna()
