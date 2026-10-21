"""Cross-context comparison: is the same XAI method suitable in both PREDICTION CONTEXTS? (D-V4)

The two contexts are EEG workload (Component 1, STEW, rest vs task) and vision engagement (Component 2,
DAiSEE, low vs high). They differ in people, task, labels and model, so the comparison is ONLY about how
the explanation methods behave and rank in each context. It never says that one modality explains
workload (or anything else) better. Reads finished results only (ranking.json, statistics.json, ICC,
screening); recomputes nothing but a rank correlation.

  compare_contexts(eeg_dir, vision_dir) -> dict    (available=False + what is missing, until both exist)
  find_context_dir(run_root, modality)  -> the primary (non-control) context folder of a run
  save_comparison(...)                  -> results/cross_context/<eeg run>__<vision run>/comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from xaicomp.paths import RESULTS_DIR, safe_path

FRAMING = ("This page compares how suitable each explanation method is in two different prediction contexts: "
           "EEG workload (STEW, rest vs task) and vision engagement (DAiSEE, low vs high). The contexts use "
           "different people, tasks, labels and models, so it does not show that one modality explains workload "
           "(or any cognitive state) better than the other.")

CAVEATS = [
    "Different datasets and people: STEW (48 adults, EEG) vs DAiSEE (students on video); no shared subjects.",
    "Different constructs and label sources: workload = rest vs SIMKAP task; engagement = crowd-annotated levels "
    "(binary: 0-1 low, 2-3 high), clip-level labels copied to every window.",
    "Different interpretable spaces: 84 EEG channel x band features vs 60 facial feature x time-segment features.",
    "Same explainers, budgets, metrics, reliability rules, statistics and ranking procedure in both contexts.",
]


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def find_context_dir(run_root: Path, modality: str) -> Path | None:
    for p in sorted(Path(run_root).glob("*/run_info.json")):
        info = _read(p)
        if not info.get("negative_control") and info.get("modality", "eeg") == modality:
            return p.parent
    return None


def _context_summary(d: Path) -> dict:
    info = _read(d / "run_info.json")
    rk = _read(d / "analysis" / "ranking.json")
    st = _read(d / "analysis" / "statistics.json") or {}
    p1 = rk["p_rank1"]
    order = sorted(p1, key=lambda m: -p1[m])
    means = {c: {r["method"]: r["mean"] for r in s.get("bootstrap_means", [])} for c, s in st.items()}
    screening = None
    sc = d.parent / "screening.parquet"
    if sc.exists():
        import pandas as pd
        df = pd.read_parquet(sc)
        df = df[df["context_id"] == info["context_id"]]
        screening = {r["method"]: r["verdict"] for _, r in df.iterrows()}
    return {"context_id": info["context_id"], "title": info.get("title"), "modality": info.get("modality", "eeg"),
            "run": d.parent.name, "run_type": info.get("run_type"), "in_sample": bool(info.get("in_sample")),
            "pilot": bool(info.get("pilot")), "n_subjects": rk.get("n_subjects"), "n_instances": info.get("n_instances"),
            "recommended": order[0], "runner_up": order[1] if len(order) > 1 else None, "p_rank1": p1,
            "expected_rank": rk.get("expected_rank", {}), "equivalence_top2": rk.get("equivalence_top2", {}),
            "criteria_decisions": {x["criterion"]: {"decision": x["decision"], "weight_factor": x.get("weight_factor")}
                                   for x in rk.get("decisions", [])},
            "criteria_used": rk.get("criteria_used", []), "criterion_means": means, "screening": screening}


def compare_contexts(eeg_dir: Path | None, vision_dir: Path | None) -> dict:
    missing = []
    for name, d in (("EEG", eeg_dir), ("vision", vision_dir)):
        if d is None or not (Path(d) / "analysis" / "ranking.json").exists():
            missing.append(f"{name} ranking (run the {name} experiment and its analysis)")
    if missing:
        return {"available": False, "missing": missing, "framing": FRAMING}
    a, b = _context_summary(Path(eeg_dir)), _context_summary(Path(vision_dir))
    methods = [m for m in a["p_rank1"] if m in b["p_rank1"]]
    tau = p = float("nan")
    if len(methods) >= 3 and a["expected_rank"] and b["expected_rank"]:
        res = kendalltau([a["expected_rank"][m] for m in methods], [b["expected_rank"][m] for m in methods])
        tau, p = float(res.statistic), float(res.pvalue)
    rows = []
    for m in methods:
        rows.append({"method": m, "p_rank1_eeg": a["p_rank1"][m], "p_rank1_vision": b["p_rank1"][m],
                     "expected_rank_eeg": a["expected_rank"].get(m), "expected_rank_vision": b["expected_rank"].get(m),
                     **{f"{c}_eeg": a["criterion_means"].get(c, {}).get(m) for c in a["criterion_means"]},
                     **{f"{c}_vision": b["criterion_means"].get(c, {}).get(m) for c in b["criterion_means"]}})
    crits = sorted(set(a["criteria_decisions"]) | set(b["criteria_decisions"]))
    reliability = [{"criterion": c, "eeg": a["criteria_decisions"].get(c, {}).get("decision"),
                    "vision": b["criteria_decisions"].get(c, {}).get("decision")} for c in crits]
    caveats = list(CAVEATS)
    for s in (a, b):
        if s["in_sample"]:
            caveats.append(f"{s['title']}: IN-SAMPLE results ({s['run_type']}) — not for final conclusions.")
        if s["pilot"]:
            caveats.append(f"{s['title']}: PILOT budget.")
    return {"available": True, "framing": FRAMING, "eeg": a, "vision": b, "methods": rows,
            "same_recommendation": a["recommended"] == b["recommended"],
            "ranking_agreement": {"kendall_tau_expected_rank": tau, "p_value": p, "n_methods": len(methods),
                                  "note": "with 4 methods a rank correlation is descriptive only"},
            "criteria_reliability": reliability, "caveats": caveats}


def save_comparison(eeg_run: str, vision_run: str, results_dir: Path = RESULTS_DIR) -> tuple[dict, Path]:
    doc = compare_contexts(find_context_dir(results_dir / eeg_run, "eeg"), find_context_dir(results_dir / vision_run, "vision"))
    out = results_dir / "cross_context" / f"{eeg_run}__{vision_run}" / "comparison.json"

    def conv(o):
        return o.item() if isinstance(o, (np.floating, np.integer)) else str(o)
    with open(safe_path(out), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=conv)
    return doc, out
