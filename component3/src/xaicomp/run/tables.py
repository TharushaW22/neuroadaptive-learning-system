"""Collect the per-instance metric files of one context into tidy tables (no recomputation).

Tables (saved to <context>/tables/*.parquet):
  instances        one row per explained window (subject, condition, p, confidence, frac_bad,
                   local band-power correlation, mean LIME surrogate R^2)
  explanations     one row per (window, method, seed): cost, sparsity, efficiency/completeness checks
  fidelity         one row per (window, method, seed, removal mode)
  stability        one row per (window, method)
  robustness       one row per (window, method, noise draw)
  agreement        one row per (window, seed, method pair)
  instance_method  one row per (window, method) with every ranking criterion (seed-averaged)
  by_seed          long table (window, method, seed, criterion, value) for test-retest ICC
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xaicomp.paths import safe_path

INSTANCE_KEYS = ["instance_id", "subject_id", "condition", "fold", "y"]


def load_records(ctx_root: Path) -> list[dict]:
    return [json.load(open(p, encoding="utf-8")) for p in sorted((ctx_root / "metrics").glob("*.json"))]


def build_tables(ctx_root: Path, save: bool = True) -> dict[str, pd.DataFrame]:
    recs = load_records(ctx_root)
    if not recs:
        return {}
    T: dict[str, list] = {k: [] for k in ("instances", "explanations", "fidelity", "stability", "robustness", "agreement")}
    for r in recs:
        key = {k: r[k] for k in INSTANCE_KEYS}
        lime_r2 = [e.get("surrogate_r2") for e in r["explanations"] if e["method"] == "lime"]
        T["instances"].append({**key, "frac_bad": r["frac_bad"], "p": r["p"], "confidence": r["confidence"],
                               "pred_is_target": r["pred_is_target"], "local_bp_corr": r["local_bp_corr"],
                               "lime_r2": float(np.nanmean(lime_r2)) if lime_r2 else np.nan,
                               "noise_flipped": any(n["flipped"] for n in r["noise"]),
                               "seconds_total": r["seconds_total"]})
        T["explanations"] += [{**key, **e} for e in r["explanations"]]
        T["fidelity"] += [{**key, **{k: v for k, v in f.items() if k != "key"}} for f in r["fidelity"]]
        T["stability"] += [{**key, **s} for s in r["stability"]]
        T["robustness"] += [{**key, **s} for s in r["robustness"]]
        T["agreement"] += [{**key, **a} for a in r["agreement"]]
    out = {k: pd.DataFrame(v) for k, v in T.items()}

    # ---- criteria per (instance, method)
    fid = out["fidelity"].pivot_table(index=["instance_id", "method", "seed"], columns="mode", values="fidelity").reset_index()
    fid = fid.rename(columns={"on_manifold": "fidelity_on", "off_manifold": "fidelity_off"})
    ex = out["explanations"][["instance_id", "method", "seed", "seconds", "n_evals", "sparsity"]]
    per_seed = ex.merge(fid, on=["instance_id", "method", "seed"], how="left")
    im = per_seed.groupby(["instance_id", "method"]).agg(
        fidelity_on=("fidelity_on", "mean"), fidelity_off=("fidelity_off", "mean"), sparsity=("sparsity", "mean"),
        cost_seconds=("seconds", "mean"), cost_evals=("n_evals", "mean")).reset_index()
    im = im.merge(out["stability"][["instance_id", "method", "stability", "stability_topk_jaccard"]],
                  on=["instance_id", "method"], how="left")
    if not out["robustness"].empty:
        rob = out["robustness"].groupby(["instance_id", "method"])["robustness"].mean().reset_index()
        im = im.merge(rob, on=["instance_id", "method"], how="left")
    else:
        im["robustness"] = np.nan
    im = out["instances"][INSTANCE_KEYS].merge(im, on="instance_id")
    out["instance_method"] = im

    long = per_seed.rename(columns={"seconds": "cost_seconds"}).melt(
        id_vars=["instance_id", "method", "seed"], value_vars=["fidelity_on", "fidelity_off", "sparsity", "cost_seconds"],
        var_name="criterion", value_name="value")
    out["by_seed"] = out["instances"][["instance_id", "subject_id"]].merge(long, on="instance_id")

    if save:
        for k, df in out.items():
            df.to_parquet(safe_path(ctx_root / "tables" / f"{k}.parquet"), index=False)
    return out


def load_tables(ctx_root: Path) -> dict[str, pd.DataFrame]:
    d = ctx_root / "tables"
    return {p.stem: pd.read_parquet(p) for p in d.glob("*.parquet")} if d.exists() else {}
