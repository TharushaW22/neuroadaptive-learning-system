"""D6 budget study: how do run-to-run stability and cost depend on the budget (coalitions, donors, IG steps)?

Explains a few real windows twice (seeds 0 and 1) under several budgets and records the Spearman
correlation between the two runs and the measured cost. Output: results/<run>/budget_study.json.
Usage: .venv/bin/python scripts/budget_study.py [--instances s02_hi_e10 s14_lo_e20] [--run-name pilot]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import _bootstrap  # noqa: F401

GRID = {
    "ig": [{"n_steps": 16, "k_donors": 2}, {"n_steps": 32, "k_donors": 4}, {"n_steps": 32, "k_donors": 8},
           {"n_steps": 32, "k_donors": 16}],
    "kernel_shap": [{"nsamples": 300, "k_donors": 2}, {"nsamples": 1000, "k_donors": 2}, {"nsamples": 500, "k_donors": 8},
                    {"nsamples": 1000, "k_donors": 4}],
    "lime": [{"num_samples": 300, "k_donors": 2}, {"num_samples": 1000, "k_donors": 4}],
}


def study(iid: str) -> list[dict]:
    import torch
    torch.set_num_threads(1)
    from xaicomp.config import load_c3_config, load_context
    from xaicomp.adapters.eeg_component1 import EEGComponent1Adapter
    from xaicomp.explainers.registry import explain
    from xaicomp.metrics.stability import stability

    ad = EEGComponent1Adapter(load_context("eeg_T1_A8"), threads=1)
    cfg = load_c3_config("explainers.yaml")["methods"]
    rows = []
    for m, grid in GRID.items():
        for b in grid:
            vals, secs, evals = [], [], []
            for seed in (0, 1):
                v = ad.view(iid, b["k_donors"], seed)
                e = explain(m, v, seed, b, cfg[m])
                vals.append(e.values); secs.append(e.seconds); evals.append(e.n_evals)
            st = stability(vals, 10)
            row = {"instance_id": iid, "method": m, **b, "stability": st["stability"],
                   "stability_topk_jaccard": st["stability_topk_jaccard"],
                   "seconds": sum(secs) / 2, "n_evals": sum(evals) / 2,
                   "ms_per_eval": 1000 * sum(secs) / sum(evals)}
            rows.append(row)
            print(json.dumps(row, default=float), flush=True)
    return rows


def main():
    from xaicomp.paths import RESULTS_DIR, safe_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=["s02_hi_e10", "s14_lo_e20", "s03_lo_e40"])
    ap.add_argument("--run-name", default="pilot")
    args = ap.parse_args()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(len(args.instances), mp_context=get_context("spawn")) as pool:
        rows = [r for rs in pool.map(study, args.instances) for r in rs]
    out = safe_path(RESULTS_DIR / args.run_name / "budget_study.json")
    json.dump({"rows": rows, "elapsed_s": time.perf_counter() - t0, "parallel_processes": len(args.instances)},
              open(out, "w"), indent=1, default=float)
    import pandas as pd
    df = pd.DataFrame(rows)
    keys = [c for c in ("nsamples", "num_samples", "n_steps", "k_donors") if c in df]
    print(df.groupby(["method"] + keys, dropna=False)[["stability", "stability_topk_jaccard", "seconds", "ms_per_eval"]].mean().round(3))
    print(f"{(time.perf_counter() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
