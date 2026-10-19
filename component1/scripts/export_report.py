"""Build every summary table FROM the result files (no number is typed by hand).

Usage:  python scripts/export_report.py
Writes: results/summary.json          (served by the API at /results/summary, read by the dashboard)
        results/summary_classical.md  baselines, feature ablation A1-A5, normalisation, Wilcoxon tests
        results/summary_deep.md       deep models, architecture ablation A6-A10, calibration, latency
        results/summary_all.md        classical vs deep on the same 5 folds and the same 18-s windows
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from eegengine.config import RESULTS_DIR
from eegengine.evaluation.reports import fmt, load_results, num

TASK_ORDER = {"T1": 0, "T2": 1, "T3": 2}
FS_ORDER = ["A1_spectral", "A2_spectral_entropy", "A3_spectral_wavelet", "A4_spectral_connectivity",
            "A5_spectral_connectivity_graph", "all"]
DEEP_ORDER = ["eegnet", "A6_cnn", "A7_cnn_lstm", "A8_cnn_lstm_trans", "A9_cnn_lstm_gnn", "A10_hybrid"]
METRICS = ("macro_f1", "balanced_accuracy", "precision_macro", "recall_macro", "auroc")


def row(r: dict, view: str = "epoch") -> dict:
    """Flatten one metrics.json into a table row. view = epoch | sequence."""
    if view == "sequence" and r["kind"] == "classical":
        sl = r.get("sequence_level") or {}
        per_fold, pooled = sl.get("per_fold", {}), sl.get("pooled", {})
        unit = sl.get("unit")
    else:
        per_fold, pooled = r.get("per_fold", {}), r.get("pooled", {})
        unit = r.get("unit", "epoch (4 s)")
    lat = r.get("latency", {})
    return {
        "experiment_id": r["experiment_id"], "kind": r["kind"], "task": r["task"], "task_short": r["task_short"],
        "model": r["model"], "feature_set": r.get("feature_set"), "normalisation": r.get("normalisation"),
        "ablation_id": r.get("ablation_id"), "cv": r["cv"], "n_folds": r.get("n_folds"), "unit": unit,
        "per_fold": {k: per_fold.get(k) for k in METRICS},
        "pooled": {k: pooled.get(k) for k in METRICS},
        "recording_level": {k: (r.get("recording_level") or {}).get(k) for k in METRICS},
        "confusion_pooled": r.get("confusion_pooled"), "confusion_recording": r.get("confusion_recording"),
        "classes": r["classes"],
        "latency_ms": lat.get("end_to_end_ms_single_window") or lat.get("model_ms_single_window_median"),
        "model_latency_ms": lat.get("predict_ms_single_window_median") or lat.get("model_ms_single_window_median"),
        "calibration": {k: (r.get("calibration") or {}).get(k) for k in
                        ("ece_before", "ece_after", "reliability_before", "reliability_after", "temperatures")}
        if r.get("calibration") else None,
        "n_params": r.get("n_params"),
    }


def sort_key(r):
    return (TASK_ORDER.get(r["task_short"], 9), r["model"], r.get("normalisation") or "")


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pooled(r, k="macro_f1"):
    return num(r["pooled"].get(k))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cls = load_results("classical")
    deep = load_results("deep")
    stats_c = json.load(open(RESULTS_DIR / "stats_classical.json", encoding="utf-8")) \
        if (RESULTS_DIR / "stats_classical.json").exists() else None
    stats_d = json.load(open(RESULTS_DIR / "stats_deep.json", encoding="utf-8")) \
        if (RESULTS_DIR / "stats_deep.json").exists() else None

    loso = [r for r in cls if r["cv"] == "loso"]
    base = sorted([r for r in loso if r["feature_set"] == "all"], key=sort_key)
    best = stats_c["best_model"] if stats_c else None
    abl = sorted([r for r in loso if best and r["model"] == best],
                 key=lambda r: (TASK_ORDER[r["task_short"]], r["normalisation"], FS_ORDER.index(r["feature_set"])))
    k5_cls = sorted([r for r in cls if r["cv"] == "kfold5"], key=sort_key)
    deep = sorted(deep, key=lambda r: (TASK_ORDER[r["task_short"]], DEEP_ORDER.index(r["model"])
                                       if r["model"] in DEEP_ORDER else 99))

    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "notes": {
            "per_fold": "mean ± std over CV folds (LOSO: one fold per subject; kfold5: 5 subject folds)",
            "pooled": "metrics over all out-of-fold predictions together",
            "T3": "each subject has one T3 class, so LOSO per-fold metrics are undefined; use pooled / recording",
            "units": "classical LOSO rows are scored per 4-s epoch; 5-fold comparison rows per 18-s sequence",
        },
        "best_classical_model": best,
        "classical_baselines": [row(r) for r in base],
        "feature_ablation": [row(r) for r in abl],
        "classical_kfold5": [row(r, "sequence") for r in k5_cls],
        "deep": [row(r) for r in deep],
        "comparison_5fold": [row(r, "sequence") for r in k5_cls] + [row(r) for r in deep],
        "stats_classical": stats_c,
        "stats_deep": stats_d,
    }
    for name in ("eda/dataset_summary.json", "preprocessing/preprocess_summary.json", "preprocessing/label_counts.json",
                 "features/feature_summary.json", "preprocessing/quality_raw_vs_filtered.json"):
        p = RESULTS_DIR / name
        if p.exists():
            summary[name.split("/")[1].replace(".json", "")] = json.load(open(p, encoding="utf-8"))
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_classical_md(base, abl, best, stats_c)
    write_deep_md(deep, stats_d)
    write_all_md(k5_cls, deep, base)
    print(f"summary.json: {len(base)} baselines, {len(abl)} ablation rows, {len(k5_cls)} classical 5-fold, "
          f"{len(deep)} deep")


def write_classical_md(base, abl, best, stats) -> None:
    L = ["# Classical models — summary (generated by scripts/export_report.py from results/)", "",
         "Protocol: leave-one-subject-out (48 folds; 45 for tasks that need a rating). Unit = one 4-s epoch.",
         "Fold metrics = mean ± std over test subjects. **T3 per-fold metrics are undefined** (each subject has "
         "one T3 class), so T3 is read from the pooled and per-recording columns.",
         "`rest_baseline` = features z-scored with the subject's own rest epochs (can inflate T1, see D9).", "",
         "## 1. Baselines (all 1499 features)", ""]
    L.append(md_table(
        ["Task", "Model", "Norm", "Fold macro-F1", "Fold bal. acc", "Pooled macro-F1", "Pooled AUROC",
         "Recording macro-F1", "Latency ms/window"],
        [[r["task_short"], r["model"], r["normalisation"], fmt(r["per_fold"]), fmt(r["per_fold"], "balanced_accuracy"),
          pooled(r), pooled(r, "auroc"), num(r["recording_level"]["macro_f1"]),
          num((r.get("latency") or {}).get("end_to_end_ms_single_window"))] for r in base]))
    L += ["", f"Best model (mean rank of pooled macro-F1 over T1 and T3, no rest baseline): **{best}**", "",
          f"## 2. Feature-family ablation A1–A5 ({best})", ""]
    L.append(md_table(
        ["Task", "Norm", "Feature set", "#features", "Fold macro-F1", "Pooled macro-F1", "Pooled AUROC", "Recording macro-F1"],
        [[r["task_short"], r["normalisation"], r["feature_set"], r["n_features"], fmt(r["per_fold"]), pooled(r),
          pooled(r, "auroc"), num(r["recording_level"]["macro_f1"])] for r in abl]))
    if stats:
        L += ["", "## 3. Paired Wilcoxon signed-rank tests (per-subject scores)", "",
              "Per-subject score = macro-F1 (T1, T2) or accuracy (T3). p_holm = Holm-corrected within each group.", ""]
        L.append(md_table(["Group", "A", "B", "n", "mean A", "mean B", "A better / B better", "p", "p_holm"],
                          [[t["group"], t["a"].replace("cls_", ""), t["b"].replace("cls_", ""), t["n_subjects"],
                            f"{t['mean_a']:.3f}", f"{t['mean_b']:.3f}", f"{t['n_a_better']}/{t['n_b_better']}",
                            f"{t['p_value']:.2g}", f"{t.get('p_holm', float('nan')):.2g}"] for t in stats["tests"]]))
    (RESULTS_DIR / "summary_classical.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_deep_md(deep, stats) -> None:
    L = ["# Deep models — summary (generated by scripts/export_report.py from results/)", "",
         "Protocol: 5 fixed subject folds (results/folds_5fold.json), inner subject-wise validation split for "
         "early stopping and temperature scaling. Unit = one sequence of 8 epochs (18 s), stride 2 epochs.", "",
         "## 1. Models and architecture ablation (A6–A10)", ""]
    L.append(md_table(
        ["Task", "Model", "Ablation", "Params", "Fold macro-F1", "Fold bal. acc", "Pooled macro-F1", "Pooled AUROC",
         "Recording macro-F1", "ECE before → after", "CPU ms/window"],
        [[r["task_short"], r["model"], r.get("ablation_id") or "baseline", r.get("n_params"), fmt(r["per_fold"]),
          fmt(r["per_fold"], "balanced_accuracy"), pooled(r), pooled(r, "auroc"), num(r["recording_level"]["macro_f1"]),
          f"{r['calibration']['ece_before']:.3f} → {r['calibration']['ece_after']:.3f}",
          num(r["latency"]["model_ms_single_window_median"])] for r in deep]))
    L += ["", "Latency = one forward pass on 1 CPU thread for one window (EEGNet: one epoch; sequence models: "
          "8 epochs), excluding preprocessing and feature extraction.", "",
          "Reliability diagrams: results/deep_*/reliability.png."]
    if stats:
        L += ["", "## 2. Paired Wilcoxon tests (per-subject scores, same folds)", ""]
        L.append(md_table(["A", "B", "n", "mean A", "mean B", "A better / B better", "p", "p_holm"],
                          [[t["a"], t["b"], t["n_subjects"], f"{t['mean_a']:.3f}", f"{t['mean_b']:.3f}",
                            f"{t['n_a_better']}/{t['n_b_better']}", f"{t['p_value']:.2g}",
                            f"{t.get('p_holm', float('nan')):.2g}"] for t in stats["tests"]]))
    (RESULTS_DIR / "summary_deep.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_all_md(k5_cls, deep, base) -> None:
    L = ["# All models on the same folds and windows (generated by scripts/export_report.py)", "",
         "Every row uses the same 5 subject folds and is scored on the same 18-s windows (8 epochs, stride 2): "
         "classical models average their epoch scores over each window; deep models predict the window directly.",
         "No rest baseline anywhere in this table.", ""]
    rows = []
    for r in k5_cls:
        rr = row(r, "sequence")
        rows.append([rr["task_short"], "classical", rr["model"], fmt(rr["per_fold"]), fmt(rr["per_fold"], "balanced_accuracy"),
                     num(rr["pooled"]["macro_f1"]), num(rr["pooled"]["auroc"]), num(rr["recording_level"]["macro_f1"]),
                     num(rr["latency_ms"])])
    for r in deep:
        rows.append([r["task_short"], "deep", r["model"], fmt(r["per_fold"]), fmt(r["per_fold"], "balanced_accuracy"),
                     pooled(r), pooled(r, "auroc"), num(r["recording_level"]["macro_f1"]),
                     num(r["latency"]["model_ms_single_window_median"])])
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    L.append(md_table(["Task", "Kind", "Model", "Fold macro-F1", "Fold bal. acc", "Pooled macro-F1", "Pooled AUROC",
                       "Recording macro-F1", "Latency ms"], rows))
    L += ["", "Latency: classical = feature extraction + prediction for one 4-s window; deep = model forward pass only "
          "(1 thread). Recording macro-F1 averages each recording's window probabilities into one prediction "
          "(T3: one prediction per subject)."]
    (RESULTS_DIR / "summary_all.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
