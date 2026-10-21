"""End-to-end evaluation on held-out TEST windows and report writing.

Real reports  -> outputs/evaluation/<timestamp>/   (only from real data + real models)
Demo reports  -> data/demo/evaluation/<timestamp>/ (synthetic; every artifact labelled DEMO ONLY)
Nothing is hardcoded: every number in a report is computed from model predictions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.feature_schema import FEATURE_GROUPS, validate_feature_frame
from src.data.validate_dataset import RealDataUnavailableError
from src.evaluation.metrics import classification_metrics, measure_latency, video_level
from src.evaluation.plots import plot_confusion_matrix, plot_model_comparison, plot_training_curves
from src.models.inference import predict_proba
from src.models.labels import get_label_scheme
from src.models.registry import latest_model, load_model
from src.models.train import (apply_normalizer, assign_validation_split, build_split_sequences, load_demo_features,
                              load_real_features, prepare_data, train_baseline)
from src.processing.preprocessing import Normalizer
from src.utils.config import resolve_path
from src.utils.logging_config import get_logger

log = get_logger("evaluate")

NOT_EVALUATED = "NOT EVALUATED: REAL DATASET NOT AVAILABLE"
DEMO_BANNER = "DEMO ONLY - SYNTHETIC DATA - NOT RESEARCH RESULTS"


def _test_sequences(cfg, df, meta, scheme):
    seqs = build_split_sequences(cfg, df, scheme, meta["feature_columns"], window=meta["window"], stride=meta["stride"])
    return apply_normalizer(seqs["test"], Normalizer.from_dict(meta["normalizer"]))


def run_evaluation(cfg: dict, demo: bool = False) -> dict:
    source = "synthetic" if demo else "real"
    scheme = get_label_scheme(cfg)
    df = load_demo_features(cfg) if demo else load_real_features(cfg)
    df = validate_feature_frame(df, allow_synthetic=demo)
    df = df[df[scheme.column].notna()]
    df = assign_validation_split(df, cfg["split"]["validation_fraction"], cfg["project"]["seed"])
    model_root = resolve_path(cfg, "demo_dir") / "models" if demo else resolve_path(cfg, "models_dir")
    metas = [m for m in (latest_model(model_root, "baseline", source), latest_model(model_root, "temporal", source)) if m]
    if not metas:
        raise RealDataUnavailableError(f"No trained {source} models found in {model_root}.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (resolve_path(cfg, "demo_dir") / "evaluation" if demo else resolve_path(cfg, "evaluation_dir")) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = DEMO_BANNER if demo else None

    report = {"title": DEMO_BANNER if demo else "Research evaluation (held-out test split)",
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "data_source": source, "dataset": "SYNTHETIC" if demo else "DAiSEE",
              "label_scheme": scheme.to_dict(),
              "label_policy": cfg["labels"]["video_label_policy"],
              "split": {"source": "official split files" if not demo else "synthetic participant-level split",
                        "participants": df.groupby("split")["participant_id"].nunique().to_dict(),
                        "videos": df.groupby("split")["video_id"].nunique().to_dict()},
              "models": [], "comparison": [], "ablation": [], "artifacts": []}
    preds_rows, comp = [], []
    for meta in metas:
        model, meta_loaded = load_model(Path(meta["path"]), allow_demo=demo,
                                        expected_classes=scheme.class_names)
        test = _test_sequences(cfg, df, meta_loaded, scheme)
        if len(test) == 0:
            raise RealDataUnavailableError("No test windows available.")
        proba = predict_proba(model, meta_loaded, test.X, test.mask)
        y_pred = proba.argmax(1)
        m = classification_metrics(test.y, y_pred, scheme.class_names)
        vy, vp = video_level(test.y, proba, test.video_ids)
        mv = classification_metrics(vy, vp, scheme.class_names)
        lat = measure_latency(lambda i: predict_proba(model, meta_loaded, test.X[i:i + 1], test.mask[i:i + 1]), 0,
                              cfg["evaluation"]["latency_repeats"])
        name = meta_loaded["model_name"]
        entry = {"model_name": name, "model_kind": meta_loaded["model_kind"], "window_level": m,
                 "video_level": mv, "latency_per_window": lat,
                 "test_participants": int(len(set(test.participant_ids))), "test_videos": int(len(set(test.video_ids)))}
        report["models"].append(entry)
        comp.append({"model": meta_loaded["model_kind"], "model_name": name, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"],
                     "macro_recall": m["macro_recall"], "video_macro_f1": mv["macro_f1"],
                     "latency_mean_ms": lat["mean_ms"]})
        cm_path = plot_confusion_matrix(m["confusion_matrix"], scheme.class_names,
                                        f"{meta_loaded['model_kind']} - window level", out_dir / f"confusion_{meta_loaded['model_kind']}.png", banner)
        report["artifacts"].append(cm_path.name)
        hist = meta_loaded.get("history", {}).get("epochs")
        if hist:
            cp = plot_training_curves(hist, f"{name} training curves", out_dir / f"curves_{meta_loaded['model_kind']}.png", banner)
            report["artifacts"].append(cp.name)
        for i in range(len(test)):
            preds_rows.append({"model": name, "video_id": test.video_ids[i], "participant_id": test.participant_ids[i],
                               "window_start_s": test.start_ts[i], "y_true": int(test.y[i]), "y_pred": int(y_pred[i]),
                               **{f"p_{c}": float(proba[i, k]) for k, c in enumerate(scheme.class_names)}})
    report["comparison"] = comp
    report["artifacts"].append(plot_model_comparison(comp, out_dir / "model_comparison.png", banner).name)

    if cfg["evaluation"].get("ablation", True):
        data = prepare_data(cfg, load_demo_features(cfg) if demo else load_real_features(cfg), source)
        groups = {"all_features": data.feature_columns,
                  **{f"without_{g}": [c for c in data.feature_columns if c not in cols] for g, cols in FEATURE_GROUPS.items()}}
        for gname, cols in groups.items():
            model, _ = train_baseline(cfg, data, Path("."), feature_subset=cols, save=False)
            idx = [data.feature_columns.index(c) for c in cols]
            p = predict_proba(model, {"model_kind": "baseline", "label_scheme": data.scheme.to_dict()},
                              data.test.X[:, :, idx], data.test.mask)
            mm = classification_metrics(data.test.y, p.argmax(1), data.scheme.class_names)
            report["ablation"].append({"feature_set": gname, "n_features": len(cols),
                                       "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]})
        report["ablation_note"] = "Baseline retrained on the training split for each feature set; evaluated on test."

    _write(out_dir, report, comp, preds_rows)
    report["output_dir"] = str(out_dir)
    log.info("Evaluation written to %s", out_dir)
    return report


def _write(out_dir: Path, report: dict, comp: list[dict], preds_rows: list[dict]) -> None:
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    pd.DataFrame(comp).to_csv(out_dir / "metrics_summary.csv", index=False)
    pc = [{"model": m["model_name"], **row} for m in report["models"] for row in m["window_level"]["per_class"]]
    pd.DataFrame(pc).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(preds_rows).to_csv(out_dir / "window_predictions.csv", index=False)
    if report["ablation"]:
        pd.DataFrame(report["ablation"]).to_csv(out_dir / "ablation.csv", index=False)
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(r: dict) -> str:
    L = [f"# {r['title']}", "", f"- Generated (UTC): {r['generated_utc']}", f"- Dataset: {r['dataset']}",
         f"- Data source: {r['data_source']}", f"- Label scheme: {r['label_scheme']['name']} "
         f"(target `{r['label_scheme']['target']}`, classes {r['label_scheme']['class_names']})",
         f"- Video-label policy: {r['label_policy']} (video labels broadcast to windows; not frame-verified)",
         f"- Split: {r['split']['source']}; participants per split {r['split']['participants']}; "
         f"videos per split {r['split']['videos']}", ""]
    if r["data_source"] != "real":
        L += [f"> **{DEMO_BANNER}.** These numbers verify that the software runs. They are not findings.", ""]
    L += ["## Model comparison (test windows)", "", "| Model | Accuracy | Macro P | Macro R | Macro F1 | Video macro F1 | Latency ms |",
          "|---|---|---|---|---|---|---|"]
    for c in r["comparison"]:
        L.append(f"| {c['model_name']} | {c['accuracy']:.3f} | {c['macro_precision']:.3f} | {c['macro_recall']:.3f} | "
                 f"{c['macro_f1']:.3f} | {c['video_macro_f1']:.3f} | {c['latency_mean_ms']:.2f} |")
    for m in r["models"]:
        L += ["", f"## {m['model_name']} - per class", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for pc in m["window_level"]["per_class"]:
            L.append(f"| {pc['class']} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")
        L.append(f"\nMajority-class accuracy on this test set: {m['window_level']['majority_class_accuracy']:.3f}")
    if r.get("ablation"):
        L += ["", "## Feature-group ablation (baseline)", "", "| Feature set | #features | Accuracy | Macro F1 |", "|---|---|---|---|"]
        for a in r["ablation"]:
            L.append(f"| {a['feature_set']} | {a['n_features']} | {a['accuracy']:.3f} | {a['macro_f1']:.3f} |")
    return "\n".join(L) + "\n"


def list_reports(cfg: dict, demo: bool = False) -> list[Path]:
    root = resolve_path(cfg, "demo_dir") / "evaluation" if demo else resolve_path(cfg, "evaluation_dir")
    if not root.exists():
        return []
    return sorted((p.parent for p in root.glob("*/report.json")), reverse=True)


def load_report(path: Path) -> dict:
    with open(Path(path) / "report.json", encoding="utf-8") as fh:
        return json.load(fh)
