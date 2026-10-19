"""Build the technical source report docs/report/ from hand-written section templates + result files.

Templates live in docs/report/_src/NN_name.md. They contain markers such as <!-- TABLE:master -->
that are replaced by tables generated HERE, directly from results/, configs/, git and the code.
Every generated table names its source file(s), so no table number is typed by hand.

Prose numbers in the templates use the citation form
    0.865 [`results/cls_T1_xgb_all_none_loso/metrics.json` → `pooled.macro_f1`]
and `--check` verifies every such citation against the cited JSON/YAML file, and verifies that
every generated table in the report equals a freshly generated one.

Usage:
  python scripts/build_report.py            # build sections, FULL_REPORT.md, figures, zip, then check
  python scripts/build_report.py --check    # only run the cross-check on the existing FULL_REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORT = ROOT / "docs" / "report"
SRC = REPORT / "_src"
FIG = REPORT / "figures"
GENFIG = RES / "figures"            # summary figures generated from result files

SECTIONS = ["00_index", "01_overview", "02_dataset", "03_preprocessing", "04_features", "05_classical_models",
            "06_deep_models", "07_evaluation_and_results", "08_system_architecture", "09_proposal_alignment",
            "10_limitations_future", "11_viva_guide", "12_appendix"]

MODEL_NAME = {"lr": "Logistic Regression", "svm": "SVM (RBF)", "rf": "Random Forest", "xgb": "XGBoost",
              "eegnet": "EEGNet", "A6_cnn": "A6 CNN", "A7_cnn_lstm": "A7 CNN-LSTM",
              "A8_cnn_lstm_trans": "A8 CNN-LSTM-Transformer", "A9_cnn_lstm_gnn": "A9 CNN-LSTM-GAT",
              "A10_hybrid": "A10 full hybrid (CNN+LSTM+Transformer+GAT)"}
FS_ORDER = ["A1_spectral", "A2_spectral_entropy", "A3_spectral_wavelet", "A4_spectral_connectivity",
            "A5_spectral_connectivity_graph", "all"]
DEEP_ORDER = ["eegnet", "A6_cnn", "A7_cnn_lstm", "A8_cnn_lstm_trans", "A9_cnn_lstm_gnn", "A10_hybrid"]


# ============================================================================================ helpers
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(eid: str) -> dict:
    return jload(RES / eid / "metrics.json")


def all_results() -> dict[str, dict]:
    out = {}
    for p in sorted(RES.glob("*/metrics.json")):
        if p.parent.name.startswith("_"):
            continue
        out[p.parent.name] = jload(p)
    return out


def f3(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ms(d, k="macro_f1"):
    m = (d or {}).get(k)
    if not m or m.get("mean") is None:
        return "—"
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


def pfmt(p):
    return "—" if p is None else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def src(path) -> str:
    return f"`{Path(path).as_posix().replace(ROOT.as_posix() + '/', '')}`"


# ============================================================================================ tables
def t_labels():
    lc = jload(RES / "preprocessing" / "label_counts.json")
    rows = []
    for unit in ("recordings", "epochs"):
        for task, c in lc[unit].items():
            bc = "; ".join(f"{k}: " + "/".join(str(v) for v in c["by_condition"][k].values()) for k in ("lo", "hi"))
            rows.append([task, unit, c["n"], c["n_subjects"], ", ".join(f"{k} {v}" for k, v in c["classes"].items()), bc])
    return table(["Task", "Unit", "N labelled", "Subjects", "Class counts", "By condition (lo / hi)"], rows) + \
        f"\n\nSource: {src(RES / 'preprocessing/label_counts.json')}"


def t_ratings():
    d = jload(RES / "eda" / "dataset_summary.json")
    rows = [[r, d["rating_distribution"]["lo"][str(r)], d["rating_distribution"]["hi"][str(r)]] for r in range(1, 10)]
    return table(["Rating (1–9)", "Rest (lo) recordings", "Task (hi) recordings"], rows) + \
        f"\n\nSource: {src(RES / 'eda/dataset_summary.json')} → `rating_distribution`"


def t_quality_rates():
    q = jload(RES / "preprocessing" / "quality_raw_vs_filtered.json")
    rows = [[name, f3(q[k]["good"], 4), f3(q[k]["warn"], 4), f3(q[k]["bad"], 4)] for name, k in
            [("Raw (mean removal + 0.5 Hz Butterworth HP)", "raw_dc_removed"), ("After 0.5–45 Hz FIR + 50 Hz notch", "filtered"),
             ("After interpolation of persistent bad channels (FINAL)", "filtered_interpolated_final")]]
    return table(["Stage (same rules, 4-s windows, 2-s step)", "good", "warn", "bad"], rows) + \
        f"\n\nn_windows = {q['n_windows']}. Source: {src(RES / 'preprocessing/quality_raw_vs_filtered.json')}"


def t_sweep():
    q = jload(RES / "preprocessing" / "quality_raw_vs_filtered.json")["amplitude_threshold_sweep_filtered"]
    rows = [[f"{t} µV", f3(v["good"], 4), f3(v["warn"], 4), f3(v["bad"], 4)] for t, v in q.items()]
    return table(["Amplitude threshold (filtered signal)", "good", "warn", "bad"], rows) + \
        f"\n\nSource: {src(RES / 'preprocessing/quality_raw_vs_filtered.json')} → `amplitude_threshold_sweep_filtered`"


def t_phase1_quality():
    q = jload(RES / "quality" / "quality_summary.json")
    rows = [[k, v] for k, v in q["windows_failing_check"].items()]
    worst_ch = ", ".join(f"{c} {v:.3f}" for c, v in list(q["mean_channel_bad_fraction"].items())[:6])
    worst = sorted(q["recordings"], key=lambda r: -r["n_bad"])[:6]
    return (table(["Check", "Windows with ≥ 1 channel failing it"], rows) +
            f"\n\nWindow flags: good {q['flag_fractions']['good']}, warn {q['flag_fractions']['warn']}, bad {q['flag_fractions']['bad']} "
            f"(n = {q['n_windows']}). Most often bad channels (mean fraction of windows): {worst_ch}.\n\n"
            "Worst recordings (bad windows / total): " + ", ".join(f"sub{r['subject_id']:02d}_{r['condition']} {r['n_bad']}/{r['n_windows']}" for r in worst) +
            f"\n\nSource: {src(RES / 'quality/quality_summary.json')} (raw signal: mean removal + 0.5 Hz high-pass; current quality rules)")


def t_interp():
    d = pd.read_csv(RES / "preprocessing" / "interpolation_log.csv")
    rows = [[r.recording, r.channel, r.reasons, r.bad_fraction, r.std_ratio, "yes" if r.interpolated else "no"] for r in d.itertuples()]
    return table(["Recording", "Channel", "Reason(s)", "Fraction of windows hard-bad", "Median-window std ratio", "Interpolated"], rows) + \
        f"\n\n{len(d)} channels in {d.recording.nunique()} recordings. Source: {src(RES / 'preprocessing/interpolation_log.csv')}"


def t_usable():
    d = pd.read_csv(RES / "preprocessing" / "usable_epochs_per_subject.csv").sort_values("usable_all").head(8)
    return table(["Subject", "Usable rest", "Usable task", "Usable all"],
                 [[int(r.subject_id), r.usable_lo, r.usable_hi, r.usable_all] for r in d.itertuples()]) + \
        f"\n\n(8 lowest of 48; usable = quality flag ≠ bad.) Source: {src(RES / 'preprocessing/usable_epochs_per_subject.csv')}"


def t_ica():
    d = jload(RES / "preprocessing" / "ica_demo_log.json")
    rows = [[k, v["n_components"], v["removed_components"] or "none", v["eog_scores_max_abs"] or "—"] for k, v in d.items()]
    return table(["Recording", "ICA components", "Removed (EOG-like)", "max |correlation score|"], rows) + \
        f"\n\nSource: {src(RES / 'preprocessing/ica_demo_log.json')} (demo run only — ICA is OFF in the main pipeline)"


def t_feature_counts():
    f = jload(RES / "features" / "feature_summary.json")
    desc = {"spectral": "abs + rel band power (14 ch × 5 bands × 2), 3 ratios × 14 ch, 3 asymmetries",
            "entropy": "6 measures × 14 ch (spectral entropy, Hjorth mobility, Hjorth complexity, log variance, skewness, kurtosis)",
            "wavelet": "6 sub-bands × 14 ch × 3 (log energy, relative energy, std) + 14 wavelet entropies",
            "plv": "91 channel pairs × 4 bands", "coh": "91 channel pairs × 4 bands",
            "graph": "4 bands × (4 node metrics × 14 ch + 3 global metrics)"}
    rows = [[k, v, desc[k]] for k, v in f["per_family"].items()]
    rows.append(["**total**", f["n_features"], ""])
    return table(["Family (column prefix)", "Features", "Composition"], rows) + \
        f"\n\nSource: {src(RES / 'features/feature_summary.json')}"


def t_feature_timing():
    f = jload(RES / "features" / "feature_summary.json")
    rows = [[k, v] for k, v in f["cpu_seconds_by_family"].items()]
    return table(["Step", "CPU seconds (all 7104 epochs, summed over workers)"], rows) + \
        f"\n\nWall time {f['extraction_s_total']} s; single-window extraction latency (median of 20) {f['single_epoch_latency_ms_median']} ms. " \
        f"Source: {src(RES / 'features/feature_summary.json')}"


def _cls_row(r, cols):
    return [r["experiment_id"]] + cols


def t_cls_baselines(R):
    rows = []
    for task in ("T1", "T2", "T3"):
        for norm in ("none", "rest_baseline"):
            for m in ("lr", "svm", "rf", "xgb"):
                eid = f"cls_{task}_{m}_all_{norm}_loso"
                if eid not in R:
                    continue
                r = R[eid]
                rows.append([task, MODEL_NAME[m], norm, ms(r["per_fold"]), ms(r["per_fold"], "balanced_accuracy"),
                             f3(r["pooled"]["macro_f1"]), f3(r["pooled"]["balanced_accuracy"]), f3(r["pooled"]["auroc"]),
                             f3(r["recording_level"]["macro_f1"]), f"`{eid}`"])
    return table(["Task", "Model", "Norm", "Fold macro-F1 (mean ± std)", "Fold bal. acc", "Pooled macro-F1",
                  "Pooled bal. acc", "Pooled AUROC", "Recording macro-F1", "Source: results/<id>/metrics.json"], rows)


def t_ablation(R, task):
    rows = []
    for norm in ("none", "rest_baseline"):
        for fs in FS_ORDER:
            eid = f"cls_{task}_xgb_{fs}_{norm}_loso"
            if eid not in R:
                continue
            r = R[eid]
            rows.append([fs, norm, r["n_features"], ms(r["per_fold"]), f3(r["pooled"]["macro_f1"]), f3(r["pooled"]["auroc"]),
                         f3(r["recording_level"]["macro_f1"]), f"`{eid}`"])
    return table(["Feature set", "Norm", "#features", "Fold macro-F1", "Pooled macro-F1", "Pooled AUROC",
                  "Recording macro-F1", "Source"], rows)


def t_stats(path, groups=None):
    d = jload(path)
    rows = []
    for i, t in enumerate(d["tests"]):
        if groups and not any(t["group"].startswith(g) for g in groups):
            continue
        rows.append([i, t["group"], t["a"], t["b"], t["metric"], t["n_subjects"], f3(t["mean_a"], 4), f3(t["mean_b"], 4),
                     f"{t['n_a_better']}/{t['n_b_better']}/{t['n_ties']}", pfmt(t["p_value"]), pfmt(t.get("p_holm"))])
    return table(["#", "Group", "A", "B", "Per-subject metric", "n", "mean A", "mean B", "A better / B better / ties",
                  "p (Wilcoxon)", "p (Holm)"], rows) + f"\n\nSource: {src(path)} (row # = index in `tests`)"


def t_cls_kfold5(R):
    rows = []
    for task in ("T1", "T3"):
        for m in ("lr", "svm", "rf", "xgb"):
            eid = f"cls_{task}_{m}_all_none_kfold5"
            if eid not in R:
                continue
            r = R[eid]
            sl = r["sequence_level"]
            rows.append([task, MODEL_NAME[m], ms(r["per_fold"]), f3(r["pooled"]["auroc"]), ms(sl["per_fold"]),
                         ms(sl["per_fold"], "balanced_accuracy"), f3(sl["pooled"]["auroc"]), f"`{eid}`"])
    return table(["Task", "Model", "Epoch-level fold macro-F1", "Epoch-level pooled AUROC", "18-s window fold macro-F1",
                  "18-s window fold bal. acc", "18-s window pooled AUROC", "Source"], rows)


def t_deep(R):
    rows = []
    for task in ("T1", "T3"):
        for m in DEEP_ORDER:
            eid = f"deep_{task}_{m}_kfold5"
            if eid not in R:
                continue
            r = R[eid]
            folds = pd.read_csv(RES / eid / "folds.csv")
            c = r["calibration"]
            rows.append([task, MODEL_NAME[m], r["n_params"], ms(r["per_fold"]), ms(r["per_fold"], "balanced_accuracy"),
                         f3(r["pooled"]["auroc"]), f3(r["recording_level"]["macro_f1"]),
                         f"{c['ece_before']:.3f} → {c['ece_after']:.3f}", f3(r["latency"]["model_ms_single_window_median"], 1),
                         str(folds.best_epoch.tolist()), round(r["train_time_s_total"]), f"`{eid}`"])
    return table(["Task", "Model", "Params", "Fold macro-F1", "Fold bal. acc", "Pooled AUROC", "Recording macro-F1",
                  "ECE before → after", "CPU ms / window (1 thread)", "Best epoch per fold", "Train+eval s", "Source"], rows)


def t_temps(R):
    rows = []
    for task in ("T1", "T3"):
        for m in DEEP_ORDER:
            eid = f"deep_{task}_{m}_kfold5"
            if eid in R:
                rows.append([task, MODEL_NAME[m], ", ".join(f"{t:.2f}" for t in R[eid]["calibration"]["temperatures"]), f"`{eid}`"])
    return table(["Task", "Model", "Temperature T per fold (bounds 0.05–20)", "Source (calibration.temperatures)"], rows)


def t_master(R):
    rows = []
    for eid, r in sorted(R.items()):
        if r.get("kind") not in ("classical", "deep"):
            continue
        sl = r.get("sequence_level") or {}
        rows.append([eid, r["task_short"], r["kind"], r["model"], r.get("feature_set"), r.get("normalisation"), r["cv"],
                     r.get("n_folds"), ms(r["per_fold"]), ms(r["per_fold"], "balanced_accuracy"), f3(r["pooled"]["macro_f1"]),
                     f3(r["pooled"]["precision_macro"]), f3(r["pooled"]["recall_macro"]), f3(r["pooled"]["auroc"]),
                     f3(r["recording_level"]["macro_f1"]), ms(sl.get("per_fold"))])
    return table(["Experiment id (results/<id>/metrics.json)", "Task", "Kind", "Model", "Features", "Norm", "CV", "Folds",
                  "Fold macro-F1", "Fold bal. acc", "Pooled macro-F1", "Pooled precision", "Pooled recall", "Pooled AUROC",
                  "Recording macro-F1", "18-s window fold macro-F1 (kfold5 classical)"], rows)


def t_rest_inflation(R):
    rows = []
    for task in ("T1", "T2", "T3"):
        for m in ("lr", "svm", "rf", "xgb"):
            a, b = f"cls_{task}_{m}_all_none_loso", f"cls_{task}_{m}_all_rest_baseline_loso"
            if a in R and b in R:
                rows.append([task, MODEL_NAME[m], f3(R[a]["pooled"]["balanced_accuracy"]), f3(R[b]["pooled"]["balanced_accuracy"]),
                             f"{R[b]['pooled']['balanced_accuracy'] - R[a]['pooled']['balanced_accuracy']:+.3f}",
                             f3(R[a]["recording_level"]["balanced_accuracy"]), f3(R[b]["recording_level"]["balanced_accuracy"]),
                             f"`{a}` / `{b}`"])
    return table(["Task", "Model", "Pooled bal. acc — none", "— rest baseline", "Δ", "Recording bal. acc — none",
                  "— rest baseline", "Sources"], rows)


def t_confusions(R):
    out = []
    for eid in ["cls_T1_xgb_all_none_loso", "cls_T1_xgb_all_none_kfold5", "deep_T1_A8_cnn_lstm_trans_kfold5",
                "deep_T1_A10_hybrid_kfold5", "cls_T3_xgb_all_none_loso", "deep_T3_A10_hybrid_kfold5", "cls_T2_xgb_all_none_loso"]:
        r = R[eid]
        cl = r["classes"]
        for view, key in (("pooled", "confusion_pooled"), ("per recording", "confusion_recording")):
            M = r[key]
            body = table(["true \\ predicted"] + cl, [[cl[i]] + M[i] for i in range(len(cl))])
            out.append(f"**`{eid}` — {view}** (source: `results/{eid}/metrics.json` → `{key}`)\n\n{body}\n")
    return "\n".join(out)


def t_leakage():
    a = jload(RES / "leakage_audit.json")
    p = a["permutation_test"]
    rows = [["Fold integrity", f"{a['fold_integrity']['experiments_checked']} experiments, problems: {len(a['fold_integrity']['problems'])}",
             "OK" if a["fold_integrity"]["ok"] else "FAIL"],
            ["Feature locality (5 epochs re-extracted alone)", f"max abs difference {a['feature_locality']['max_abs_difference']}",
             "OK" if a["feature_locality"]["ok"] else "FAIL"],
            ["Label-permutation test (3 runs, LR LOSO T1)", "balanced accuracy " + ", ".join(f"{r['pooled_balanced_accuracy']:.3f}" for r in p["runs"]) +
             f"; mean {p['mean_balanced_accuracy']:.3f}", "≈ chance → OK"],
            ["Subject-identity probe (random epoch split)", f"accuracy {a['subject_identity_probe']['random_split_subject_id_accuracy']:.3f} "
             f"(chance {a['subject_identity_probe']['chance']:.3f})", "shows why subject-wise CV is required"],
            ["Results with balanced accuracy > 0.97", "; ".join(f"{h['experiment']} ({h['view']}, {h['balanced_accuracy']:.3f}, {h['normalisation']})"
                                                              for h in a["results_above_97pct"]) or "none", "audited (rest-baseline effect)"]]
    return table(["Check", "Result", "Verdict"], rows) + f"\n\nSource: {src(RES / 'leakage_audit.json')}"


def t_latency(R):
    rows = []
    for task in ("T1",):
        for m in ("lr", "svm", "rf", "xgb"):
            eid = f"cls_{task}_{m}_all_none_kfold5"
            l = R[eid]["latency"]
            rows.append([MODEL_NAME[m], "classical", f3(l["feature_extraction_ms_single_window"], 2), f3(l["predict_ms_single_window_median"], 2),
                         f3(l["end_to_end_ms_single_window"], 2), f"`{eid}`"])
        for m in DEEP_ORDER:
            eid = f"deep_{task}_{m}_kfold5"
            l = R[eid]["latency"]
            rows.append([MODEL_NAME[m], f"deep ({l['window_epochs']} epoch(s) per forward pass)", "—",
                         f3(l["model_ms_single_window_median"], 2) + f" (p95 {l['model_ms_p95']:.2f})", "—", f"`{eid}`"])
    eng = engine_latency()
    note = (f"\n\nEnd-to-end engine latency per window (filtering 20 s context + quality + features + model + JSON), from the "
            f"offline demo streams produced by `scripts/export_demo.py` (model `{eng['model']}`, {eng['n_windows']} windows): median "
            f"{eng['median_ms']:.1f} ms, p95 {eng['p95_ms']:.1f} ms — saved to {src(RES / 'engine_latency.json')}.") if eng else ""
    return table(["Model", "Kind", "Feature extraction ms", "Model ms (single window, median)", "Classical end-to-end ms", "Source"], rows) + note


def engine_latency():
    """Summarise per-window latency_ms recorded in the demo stream files -> results/engine_latency.json."""
    files = sorted((ROOT / "frontend" / "public" / "demo").glob("stream_sub*.json"))
    if not files:
        return None
    lat, model = [], None
    for f in files:
        for m in jload(f):
            if m["type"] == "meta":
                model = m["model"]
            if m["type"] == "state":
                lat.append(m["output"]["latency_ms"])
    d = {"source": "frontend/public/demo/stream_sub*.json (scripts/export_demo.py)", "model": model, "n_windows": len(lat),
         "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "max_ms": float(np.max(lat)),
         "note": "measured on the shared development laptop while other applications were running"}
    with open(RES / "engine_latency.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return d


def t_timeline():
    log = subprocess.run(["git", "log", "--date=format:%Y-%m-%d %H:%M", "--pretty=%h|%ad|%s", "--", "."], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()
    rows = [l.split("|", 2) for l in reversed(log)]
    return table(["Commit", "Date (+0530)", "Message"], rows) + "\n\nSource: `git log -- component1`"


def t_versions():
    import importlib.metadata as md
    pkgs = ["numpy", "scipy", "pandas", "pyarrow", "matplotlib", "PyYAML", "mne", "PyWavelets", "networkx", "scikit-learn",
            "xgboost", "torch", "joblib", "fastapi", "uvicorn", "pydantic", "python-multipart", "httpx", "websockets", "pytest"]
    req = {l.split(">=")[0].split("[")[0].strip(): l.split("#")[0].strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")}
    rows = [["Python", "≥ 3.11 (pyproject)", sys.version.split()[0]]]
    for p in pkgs:
        try:
            v = md.version(p)
        except md.PackageNotFoundError:
            v = "not installed"
        rows.append([p, req.get(p, "(dependency)"), v])
    lock = jload(ROOT / "frontend" / "package-lock.json")["packages"]
    pk = jload(ROOT / "frontend" / "package.json")
    for n, spec in {**pk["dependencies"], **pk["devDependencies"]}.items():
        if n in ("react", "react-dom", "react-router-dom", "recharts", "vite", "tailwindcss", "@tailwindcss/vite", "typescript", "@vitejs/plugin-react"):
            rows.append([n, spec, lock.get(f"node_modules/{n}", {}).get("version", "?")])
    return table(["Package", "Required (requirements.txt / package.json)", "Installed in the dev environment"], rows)


def t_code_index():
    rows = []
    files = sorted([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("api").rglob("*.py"), *ROOT.joinpath("scripts").rglob("*.py"),
                    *ROOT.joinpath("tests").rglob("*.py")])
    for f in files:
        if f.name == "__init__.py":
            continue
        text = f.read_text(encoding="utf-8")
        tree = ast.parse(text)
        doc = (ast.get_docstring(tree) or "").splitlines()
        funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
        rows.append([f"`{f.relative_to(ROOT).as_posix()}`", doc[0] if doc else "", len(text.splitlines()), ", ".join(funcs[:12])])
    for f in sorted(ROOT.joinpath("frontend", "src").rglob("*.ts*")):
        text = f.read_text(encoding="utf-8")
        first = next((l[2:].strip() for l in text.splitlines() if l.startswith("//")), "")
        exports = re.findall(r"export (?:default )?(?:function|const|interface|type) (\w+)", text)
        rows.append([f"`{f.relative_to(ROOT).as_posix()}`", first, len(text.splitlines()), ", ".join(exports[:12])])
    return table(["File", "Purpose (first docstring / comment line)", "Lines", "Public functions / classes / exports"], rows)


def t_tests():
    rows, total = [], 0
    for f in sorted(ROOT.joinpath("tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        n = sum(1 for x in tree.body if isinstance(x, ast.FunctionDef) and x.name.startswith("test_"))
        total += n
        rows.append([f"`tests/{f.name}`", (ast.get_docstring(tree) or "").splitlines()[0], n])
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8").stdout
    per_file = [int(x) for x in re.findall(r"^tests/\S+\.py: (\d+)$", res, re.M)]   # `-q` twice -> per-file counts
    collected = sum(per_file) if per_file else "NOT AVAILABLE"
    return table(["Test file", "What it covers", "Test functions"], rows) + \
        f"\n\nTest functions: {total}; collected test cases (after `@pytest.mark.parametrize` expansion): {collected}."


def t_configs():
    out = []
    for p in sorted(ROOT.joinpath("configs").rglob("*.yaml")):
        out.append(f"### `{p.relative_to(ROOT).as_posix()}`\n\n```yaml\n{p.read_text(encoding='utf-8').strip()}\n```\n")
    return "\n".join(out)


def t_decisions():
    text = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    parts = re.split(r"\n## ", text)[1:]
    rows = []
    for p in parts:
        title, body = p.split("\n", 1)
        first = " ".join(body.strip().split("\n")[:3]).strip()
        rows.append([title.split(" — ")[0], title.split(" — ", 1)[1] if " — " in title else title, first[:260] + ("…" if len(first) > 260 else "")])
    return table(["ID", "Decision", "Summary (first lines of docs/decisions.md)"], rows)


def t_api_endpoints():
    text = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    eps = re.findall(r'@app\.(get|post|websocket)\("([^"]+)"', text)
    return table(["Method", "Path"], [[m.upper() if m != "websocket" else "WS", f"`{p}`"] for m, p in eps]) + \
        "\n\nSource: decorators in `api/main.py`"


def t_figures(figs):
    return table(["File (docs/report/figures/)", "Original location", "Caption"], [[f"`{n}`", f"`{o}`", c] for n, o, c in figs])


def t_models_deployed():
    rows = []
    for j in sorted((ROOT / "models").glob("*.json")):
        d = jload(j)
        cv = d.get("cv_summary") or {}
        f1 = cv.get("per_fold_macro_f1") or {}
        rows.append([f"`{j.stem}`", d.get("task"), d.get("version"), f"{d.get('temperature'):.2f}" if d.get("temperature") else "—",
                     cv.get("experiment_id", "—"), f"{f1.get('mean'):.3f}" if f1.get("mean") is not None else "—"])
    return table(["Model id (models/)", "Task", "Version", "Temperature", "CV experiment it represents", "That experiment's fold macro-F1"], rows) + \
        "\n\nSource: `models/*.json` (written by `scripts/train_final_classical.py` and `scripts/run_deep.py --final`)"


# ============================================================================================ figures
def make_summary_figures(R) -> list[tuple[str, str, str]]:
    GENFIG.mkdir(parents=True, exist_ok=True)
    out = []
    blue, orange, grey = "#2a78d6", "#eb6834", "#7a7974"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

    # 1. feature ablation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, task in zip(axes, ("T1", "T3")):
        for k, (norm, col) in enumerate((("none", blue), ("rest_baseline", orange))):
            vals = [R.get(f"cls_{task}_xgb_{fs}_{norm}_loso", {}).get("pooled", {}).get("macro_f1", np.nan) for fs in FS_ORDER]
            ax.barh(np.arange(len(FS_ORDER)) + (k - 0.5) * 0.38, vals, height=0.36, color=col, label=norm)
        ax.set_yticks(range(len(FS_ORDER)), FS_ORDER)
        ax.axvline(0.5, color=grey, ls="--", lw=0.8)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("pooled macro-F1 (LOSO, XGBoost)")
        ax.set_title(f"{task} feature ablation", loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")   # T3 panel has free space on the right
    fig.tight_layout()
    p = GENFIG / "feature_ablation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(("summary_feature_ablation.png", p, "Feature-family ablation A1–A5 + all (XGBoost, LOSO), pooled macro-F1, with/without rest baseline; dashed = chance."))

    # 2. classical vs deep on the same folds/windows
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ("T1", "T3")):
        names, means, stds, cols = [], [], [], []
        for m in ("lr", "svm", "rf", "xgb"):
            sl = R[f"cls_{task}_{m}_all_none_kfold5"]["sequence_level"]["per_fold"]["macro_f1"]
            names.append(m)
            means.append(sl["mean"])
            stds.append(sl["std"])
            cols.append(blue)
        for m in DEEP_ORDER:
            pf = R[f"deep_{task}_{m}_kfold5"]["per_fold"]["macro_f1"]
            names.append(m)
            means.append(pf["mean"])
            stds.append(pf["std"])
            cols.append(orange)
        y = np.arange(len(names))
        ax.barh(y, means, xerr=stds, color=cols, height=0.6, error_kw={"elinewidth": 0.8})
        ax.set_yticks(y, names)
        ax.axvline(0.5, color=grey, ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fold macro-F1 (5 subject folds, 18-s windows), mean ± std")
        ax.set_title(f"{task}: classical (blue) vs deep (orange)", loc="left")
    fig.tight_layout()
    p = GENFIG / "classical_vs_deep.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(("summary_classical_vs_deep.png", p, "All models on the same 5 subject folds and the same 18-s windows (fold macro-F1 ± std)."))

    # 3. rest-baseline inflation
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ms_ = ("lr", "svm", "rf", "xgb")
    a = [R[f"cls_T1_{m}_all_none_loso"]["pooled"]["balanced_accuracy"] for m in ms_]
    b = [R[f"cls_T1_{m}_all_rest_baseline_loso"]["pooled"]["balanced_accuracy"] for m in ms_]
    x = np.arange(len(ms_))
    ax.bar(x - 0.2, a, 0.38, color=blue, label="no baseline")
    ax.bar(x + 0.2, b, 0.38, color=orange, label="subject rest baseline")
    ax.set_xticks(x, ms_)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("pooled balanced accuracy (T1, LOSO)")
    ax.set_title("Rest-baseline normalisation inflates T1", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = GENFIG / "rest_baseline_inflation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(("summary_rest_baseline_inflation.png", p, "T1 pooled balanced accuracy with vs without the subject rest baseline (LOSO, all features)."))

    # 4. latency
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    names = [*("lr", "svm", "rf", "xgb"), *DEEP_ORDER]
    vals = [R[f"cls_T1_{m}_all_none_kfold5"]["latency"]["end_to_end_ms_single_window"] for m in ("lr", "svm", "rf", "xgb")] + \
           [R[f"deep_T1_{m}_kfold5"]["latency"]["model_ms_single_window_median"] for m in DEEP_ORDER]
    ax.barh(range(len(names)), vals, color=[blue] * 4 + [orange] * len(DEEP_ORDER))
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("ms per window (classical: features + model; deep: forward pass, 1 thread)")
    ax.set_title("CPU latency per window", loc="left")
    fig.tight_layout()
    p = GENFIG / "latency.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(("summary_latency.png", p, "CPU latency per window (T1 models)."))
    return out


def collect_figures(R):
    figs = [
        ("eda_rating_distribution.png", RES / "eda/rating_distribution.png", "Self-rated workload (1–9) for rest vs SIMKAP task; dashed lines = T2 level boundaries."),
        ("eda_raw_sub01.png", RES / "eda/raw_sub01.png", "First 10 s of subject 01 (rest vs task), DC removed + 0.5 Hz HP; shows the broken T7 electrode at rest."),
        ("eda_raw_sub24.png", RES / "eda/raw_sub24.png", "First 10 s of subject 24 (an unrated subject), rest vs task."),
        ("eda_psd_grand_average.png", RES / "eda/psd_grand_average.png", "Grand-average PSD (mean of 14 channels) rest vs task, band = ±1 SD across subjects."),
        ("eda_psd_subjects.png", RES / "eda/psd_subjects.png", "Per-subject PSD rest vs task for subjects 01, 12, 24, 36."),
        ("prep_psd_before_after_sub01.png", RES / "preprocessing/psd_before_after_sub01.png", "PSD before vs after preprocessing, subject 01."),
        ("prep_psd_before_after_sub38.png", RES / "preprocessing/psd_before_after_sub38.png", "PSD before vs after preprocessing, subject 38 (F3 interpolated in the task recording)."),
    ]
    for eid, cap in [("cls_T1_xgb_all_none_loso", "Confusion matrices, XGBoost T1 LOSO (pooled epochs and per recording)."),
                     ("cls_T1_xgb_all_none_kfold5", "Confusion matrices, XGBoost T1 5-fold."),
                     ("cls_T1_svm_all_rest_baseline_loso", "Confusion matrices, SVM T1 LOSO WITH rest baseline (the 0.990 per-recording case)."),
                     ("cls_T3_xgb_all_none_loso", "Confusion matrices, XGBoost T3 LOSO (≈ chance)."),
                     ("cls_T2_xgb_all_none_loso", "Confusion matrices, XGBoost T2 LOSO (rest ≈ low)."),
                     ("deep_T1_eegnet_kfold5", "Confusion matrices, EEGNet T1 5-fold (18-s windows)."),
                     ("deep_T1_A8_cnn_lstm_trans_kfold5", "Confusion matrices, A8 CNN-LSTM-Transformer T1."),
                     ("deep_T1_A10_hybrid_kfold5", "Confusion matrices, A10 full hybrid T1."),
                     ("deep_T3_A10_hybrid_kfold5", "Confusion matrices, A10 full hybrid T3.")]:
        figs.append((f"cm_{eid}.png", RES / eid / "confusion_matrix.png", cap))
    for eid in ("deep_T1_A10_hybrid_kfold5", "deep_T1_A8_cnn_lstm_trans_kfold5", "deep_T1_eegnet_kfold5", "deep_T3_A10_hybrid_kfold5"):
        figs.append((f"reliability_{eid}.png", RES / eid / "reliability.png", f"Reliability diagram before/after temperature scaling, {eid}."))
    figs += make_summary_figures(R)
    FIG.mkdir(parents=True, exist_ok=True)
    for name, path, _ in figs:
        shutil.copy(path, FIG / name)
    return [(n, Path(p).relative_to(ROOT).as_posix(), c) for n, p, c in figs]


# ============================================================================================ build / check
def generated_tables(R, figs) -> dict[str, str]:
    return {
        "labels": t_labels(), "ratings": t_ratings(), "quality_rates": t_quality_rates(), "sweep": t_sweep(),
        "phase1_quality": t_phase1_quality(), "interp": t_interp(), "usable": t_usable(), "ica": t_ica(),
        "feature_counts": t_feature_counts(), "feature_timing": t_feature_timing(),
        "cls_baselines": t_cls_baselines(R), "ablation_T1": t_ablation(R, "T1"), "ablation_T3": t_ablation(R, "T3"),
        "ablation_T2": t_ablation(R, "T2"),
        "stats_classical": t_stats(RES / "stats_classical.json"), "stats_deep": t_stats(RES / "stats_deep.json"),
        "cls_kfold5": t_cls_kfold5(R), "deep": t_deep(R), "temps": t_temps(R), "master": t_master(R),
        "rest_inflation": t_rest_inflation(R), "confusions": t_confusions(R), "leakage": t_leakage(),
        "latency": t_latency(R), "timeline": t_timeline(), "versions": t_versions(), "code_index": t_code_index(),
        "tests": t_tests(), "configs": t_configs(), "decisions": t_decisions(), "api_endpoints": t_api_endpoints(),
        "figures": t_figures(figs), "models_deployed": t_models_deployed(),
        "output_example": "```json\n" + (ROOT / "docs" / "output_example.json").read_text(encoding="utf-8").strip() + "\n```",
    }


def render(text: str, tables: dict[str, str]) -> str:
    def rep(m):
        key = m.group(1)
        if key not in tables:
            raise KeyError(f"unknown table marker {key}")
        return f"<!-- BEGIN TABLE:{key} -->\n{tables[key]}\n<!-- END TABLE:{key} -->"
    return re.sub(r"<!-- TABLE:(\w+) -->", rep, text)


def build() -> None:
    R = all_results()
    figs = collect_figures(R)
    tables = generated_tables(R, figs)
    parts = []
    for s in SECTIONS:
        text = (SRC / f"{s}.md").read_text(encoding="utf-8")
        out = render(text, tables)
        (REPORT / f"{s}.md").write_text(out, encoding="utf-8")
        parts.append(out)
    (REPORT / "FULL_REPORT.md").write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"built {len(SECTIONS)} sections + FULL_REPORT.md, {len(figs)} figures in {FIG}")


def make_zip() -> None:
    """Zip docs/report/ (without the _src templates) into docs/report_bundle.zip — after the check, so
    verification.json is included."""
    bundle = ROOT / "docs" / "report_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REPORT.rglob("*")):
            if p.is_file() and "_src" not in p.parts:
                z.write(p, Path("report") / p.relative_to(REPORT))
    print(f"zipped -> {bundle}")


CITE = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\**\s*\[`((?:results|configs|models|docs)/[^`]+)`\s*→\s*`([^`]+)`\]")


def lookup(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def matches(written: str, actual) -> bool:
    w = float(written)
    a = float(actual)
    if "e" in written.lower():
        return abs(a - w) <= 0.05 * abs(w) + 1e-300
    d = len(written.split(".")[1]) if "." in written else 0
    return round(a, d) == round(w, d) or abs(a - w) <= 0.5 * 10 ** (-d) + 1e-12


def check() -> int:
    R = all_results()
    full = (REPORT / "FULL_REPORT.md").read_text(encoding="utf-8")
    problems, n_cite, n_tab = [], 0, 0
    # 1. prose citations
    cache = {}
    for m in CITE.finditer(full):
        n_cite += 1
        val, path, key = m.groups()
        p = ROOT / path
        if p not in cache:
            cache[p] = jload(p) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
        try:
            actual = lookup(cache[p], key)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            problems.append(f"citation {val} → {path}:{key}: key not found ({e})")
            continue
        if not matches(val, actual):
            problems.append(f"citation MISMATCH: wrote {val}, file has {actual} ({path} → {key})")
    # 2. generated tables equal a fresh generation
    fig_rows = re.search(r"<!-- BEGIN TABLE:figures -->\n(.*?)\n<!-- END TABLE:figures -->", full, re.S)
    fresh = generated_tables(R, figs=[])  # figures table compared separately below
    for key, body in re.findall(r"<!-- BEGIN TABLE:(\w+) -->\n(.*?)\n<!-- END TABLE:\1 -->", full, re.S):
        n_tab += 1
        if key in ("figures", "timeline", "tests", "versions"):
            continue   # environment-dependent (git / installed packages); not result numbers
        if fresh[key].strip() != body.strip():
            problems.append(f"table {key} differs from a fresh generation from results/")
    if fig_rows:
        for line in fig_rows.group(1).splitlines()[2:]:
            name = line.split("|")[1].strip().strip("`")
            if not (FIG / name).exists():
                problems.append(f"figure listed but missing: {name}")
    report = {"citations_checked": n_cite, "tables_checked": n_tab, "problems": problems}
    with open(REPORT / "verification.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"checked {n_cite} prose citations and {n_tab} generated tables: {len(problems)} problem(s)")
    for p in problems:
        print("  -", p)
    return len(problems)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        build()
    n = check()
    if not args.check:
        make_zip()
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
