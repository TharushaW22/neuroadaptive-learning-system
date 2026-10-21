"""Paired Wilcoxon tests for the deep models (same 5 folds, same 18-s windows).

* architecture ablation: full hybrid (A10) vs each other deep model, per task
* classical vs deep: best classical model on the 5 folds (sequence view) vs A10 and vs the best deep model
Per-subject score = macro-F1 if every subject has both classes (T1), else accuracy (T3).
Writes results/stats_deep.json.
"""

from __future__ import annotations

import json
import sys

from eegengine.config import RESULTS_DIR
from eegengine.evaluation.ablation import holm, paired_wilcoxon
from eegengine.evaluation.reports import load_results


def seq_view(r: dict) -> dict:
    """Classical 5-fold result scored on the deep models' windows."""
    sl = r["sequence_level"]
    return {"experiment_id": r["experiment_id"] + "[windows]", "per_subject": sl["per_subject"],
            "per_subject_metric": sl["per_subject_metric"], "pooled": sl["pooled"], "per_fold": sl["per_fold"]}


def score(r: dict) -> float:
    pf = (r.get("per_fold") or {}).get("macro_f1") or {}
    return pf.get("mean") if pf.get("mean") is not None else r["pooled"]["macro_f1"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    deep = load_results("deep")
    k5 = [r for r in load_results("classical") if r["cv"] == "kfold5" and r.get("sequence_level")]
    tests = []
    for task in sorted({r["task_short"] for r in deep}):
        d = {r["model"]: r for r in deep if r["task_short"] == task}
        full = d.get("A10_hybrid")
        if full:
            for name, r in d.items():
                if name != "A10_hybrid":
                    tests.append({"group": f"deep_ablation/{task}", **paired_wilcoxon(full, r)})
        cls = [seq_view(r) for r in k5 if r["task_short"] == task]
        if cls and d:
            best_c = max(cls, key=score)
            best_d = max(d.values(), key=score)
            for target in {id(full): full, id(best_d): best_d}.values():
                if target:
                    tests.append({"group": f"classical_vs_deep/{task}", **paired_wilcoxon(target, best_c)})
    for fam in sorted({t["group"] for t in tests}):
        idx = [i for i, t in enumerate(tests) if t["group"] == fam]
        for i, p in zip(idx, holm([tests[i]["p_value"] for i in idx])):
            tests[i]["p_holm"] = p
    with open(RESULTS_DIR / "stats_deep.json", "w", encoding="utf-8") as f:
        json.dump({"tests": tests}, f, indent=2)
    for t in tests:
        print(f"{t['group']:28s} {t['a']:40s} vs {t['b']:45s} {t['mean_a']:.3f} vs {t['mean_b']:.3f}  p={t['p_value']:.3g}")


if __name__ == "__main__":
    main()
