"""Phase 4 feature-family ablation (A1-A5 + all) with the best classical model, plus Wilcoxon tests.

The best model is chosen from the baseline runs WITHOUT rest baseline: best mean rank of pooled
macro-F1 over T1 and T3 (the two primary tasks). Ablations run for every task and both
normalisation settings.

Usage:  python scripts/run_ablation.py [--model xgb] [--skip-existing]
Writes: results/cls_<T>_<model>_<A..>_<norm>_loso/ and results/stats_classical.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from eegengine.config import RESULTS_DIR, load_config
from eegengine.data import load_features
from eegengine.evaluation.ablation import holm, paired_wilcoxon
from eegengine.evaluation.reports import load_results
from eegengine.labels import TASKS
from eegengine.models.classical import MODELS
from eegengine.training.classical_runner import FEATURE_SETS, experiment_id, run_classical


def pick_best_model(results: list[dict]) -> tuple[str, dict]:
    ranks = {m: [] for m in MODELS}
    table = {}
    for task in ("workload_binary", "workload_within_task"):
        rows = [r for r in results if r["task"] == task and r["feature_set"] == "all"
                and r["normalisation"] == "none" and r["cv"] == "loso"]
        scores = {r["model"]: r["pooled"]["macro_f1"] for r in rows}
        table[task] = scores
        for rank, m in enumerate(sorted(scores, key=scores.get, reverse=True), start=1):
            ranks[m].append(rank)
    mean_rank = {m: float(np.mean(v)) for m, v in ranks.items() if v}
    best = min(mean_rank, key=lambda m: (mean_rank[m], -table["workload_within_task"].get(m, 0)))
    return best, {"mean_rank": mean_rank, "pooled_macro_f1": table}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/classical.yaml")
    ap.add_argument("--model")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    base = [r for r in load_results("classical") if r["cv"] == "loso"]
    best, selection_info = (args.model, {"chosen_manually": True}) if args.model else pick_best_model(base)
    print(f"Best classical model: {best}  {selection_info}")

    F, meta, _ = load_features(cfg["features_config"])
    with open(RESULTS_DIR / "features" / "feature_summary.json", encoding="utf-8") as f:
        feat_lat = json.load(f)["single_epoch_latency_ms_median"]

    for task in cfg["tasks"]:
        for norm in cfg["normalisations"]:
            for fset in FEATURE_SETS:
                eid = experiment_id(task, best, fset, norm, "loso")
                out = RESULTS_DIR / eid
                if (out / "metrics.json").exists() and (args.skip_existing or fset == "all"):
                    continue
                r = run_classical(F, meta, task=task, model=best, feature_set=fset, normalisation=norm, cv="loso",
                                  out_dir=out, seed=cfg.get("seed", 42), feature_latency_ms=feat_lat,
                                  n_jobs=cfg.get("n_jobs", 8))
                print(f"{eid:60s} pooled F1 {r['pooled']['macro_f1']:.3f}  AUROC {r['pooled']['auroc']:.3f}  "
                      f"rec F1 {r['recording_level']['macro_f1']:.3f}", flush=True)

    # ---------------------------------------------------------------- statistics
    res = {r["experiment_id"]: r for r in load_results("classical") if r["cv"] == "loso"}
    tests = []

    def add(group, a, b):
        if a in res and b in res:
            tests.append({"group": group, **paired_wilcoxon(res[a], res[b])})

    for task in cfg["tasks"]:
        for norm in cfg["normalisations"]:
            ref = experiment_id(task, best, "A1_spectral", norm, "loso")
            for fset in FEATURE_SETS:
                if fset != "A1_spectral":
                    add(f"ablation_vs_A1/{TASKS[task]['short']}/{norm}", experiment_id(task, best, fset, norm, "loso"), ref)
        for m in MODELS:
            add(f"rest_baseline_vs_none/{TASKS[task]['short']}",
                experiment_id(task, m, "all", "rest_baseline", "loso"), experiment_id(task, m, "all", "none", "loso"))
            if m != best:
                add(f"best_vs_other_model/{TASKS[task]['short']}/none",
                    experiment_id(task, best, "all", "none", "loso"), experiment_id(task, m, "all", "none", "loso"))

    # Holm correction within each group family
    for fam in sorted({t["group"] for t in tests}):
        idx = [i for i, t in enumerate(tests) if t["group"] == fam]
        for i, p in zip(idx, holm([tests[i]["p_value"] for i in idx])):
            tests[i]["p_holm"] = p

    with open(RESULTS_DIR / "stats_classical.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best, "selection": selection_info, "tests": tests}, f, indent=2)
    print(f"Wrote {len(tests)} paired tests to results/stats_classical.json")


if __name__ == "__main__":
    main()
