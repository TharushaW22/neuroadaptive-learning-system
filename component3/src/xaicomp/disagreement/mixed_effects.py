"""Why do methods disagree? A linear mixed model with a random intercept per subject.

Unit: one (window, method pair). Outcome: disagreement = 1 - Spearman(A, B) (averaged over seeds).
Predictors (z-scored, so coefficients are comparable: change in disagreement per 1 SD):
  confidence     the model's confidence for the window (max class probability)
  lime_r2        how well a straight line fits the model locally (LIME surrogate R^2)
  local_bp_corr  how correlated the window's band powers are (feature dependence)
  frac_bad       share of the window's epochs flagged 'bad' by C1's quality checks
  + method pair as a fixed effect; subject as a random intercept (windows of one subject are not independent)

  disagreement ~ confidence + lime_r2 + local_bp_corr + frac_bad + C(pair) + (1 | subject)

Also: the KernelSHAP vs Conditional SHAP pair alone (same estimator and coalitions, different value
function) to isolate the effect of the value function, and how it relates to feature dependence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDICTORS = ["confidence", "lime_r2", "local_bp_corr", "frac_bad"]


def disagreement_frame(agreement: pd.DataFrame, instances: pd.DataFrame, exclude: set[str] = frozenset()) -> pd.DataFrame:
    ag = agreement[~agreement["method_a"].isin(exclude) & ~agreement["method_b"].isin(exclude)]
    ag = ag.groupby(["instance_id", "method_a", "method_b"])["rank_corr"].mean().reset_index()
    ag["pair"] = ag["method_a"] + " vs " + ag["method_b"]
    ag["disagreement"] = 1 - ag["rank_corr"]
    return ag.merge(instances[["instance_id", "subject_id"] + PREDICTORS], on="instance_id")


def fit_mixed(df: pd.DataFrame, with_pair: bool = True) -> dict:
    import statsmodels.formula.api as smf

    d = df.dropna(subset=["disagreement"] + PREDICTORS).copy()
    used = []
    for p in PREDICTORS:
        sd = d[p].std()
        if sd and sd > 0:
            d[f"z_{p}"] = (d[p] - d[p].mean()) / sd
            used.append(f"z_{p}")
    if d["subject_id"].nunique() < 3 or len(d) < len(used) + 5:
        return {"converged": False, "error": "too few subjects/rows for a mixed model", "n": len(d)}
    formula = "disagreement ~ " + " + ".join(used or ["1"]) + (" + C(pair)" if with_pair and d["pair"].nunique() > 1 else "")
    last_err = None
    for method in ("lbfgs", "powell", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, d, groups=d["subject_id"]).fit(reml=True, method=method)
            ci = res.conf_int()
            coefs = [{"term": t, "coef": float(res.params[t]), "ci_low": float(ci.loc[t, 0]), "ci_high": float(ci.loc[t, 1]),
                      "p_value": float(res.pvalues[t])} for t in res.params.index if t != "Group Var"]
            return {"converged": bool(res.converged), "optimizer": method, "formula": formula + " + (1 | subject)",
                    "n": int(len(d)), "n_subjects": int(d["subject_id"].nunique()),
                    "subject_variance": float(res.cov_re.iloc[0, 0]), "residual_variance": float(res.scale),
                    "coefficients": coefs}
        except Exception as e:                       # try the next optimizer
            last_err = str(e)
    return {"converged": False, "error": last_err, "n": len(d)}


def value_function_effect(df: pd.DataFrame, a: str = "conditional_shap", b: str = "kernel_shap") -> dict:
    """KernelSHAP vs Conditional SHAP: how different are they, and does it grow with feature dependence?"""
    pair = df[((df["method_a"] == a) & (df["method_b"] == b)) | ((df["method_a"] == b) & (df["method_b"] == a))]
    others = df[~df.index.isin(pair.index)]
    if pair.empty:
        return {"available": False}
    rho_bp = spearmanr(pair["local_bp_corr"], pair["disagreement"]).statistic if len(pair) > 3 else np.nan
    rho_bad = spearmanr(pair["frac_bad"], pair["disagreement"]).statistic if len(pair) > 3 and pair["frac_bad"].nunique() > 1 else np.nan
    return {"available": True, "n": int(len(pair)), "mean_disagreement": float(pair["disagreement"].mean()),
            "mean_disagreement_other_pairs": float(others["disagreement"].mean()) if len(others) else np.nan,
            "spearman_with_local_bp_corr": float(rho_bp), "spearman_with_frac_bad": float(rho_bad),
            "mixed_model": fit_mixed(pair, with_pair=False)}
