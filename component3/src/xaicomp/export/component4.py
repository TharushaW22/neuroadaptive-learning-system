"""Component 4 interface (D7): per context, which explainer to trust; per window, what drove the prediction.

  recommendation(run_root, context_id)  -> recommended explainer, P(rank 1) from SMAA, runner-up,
                                           practically equivalent? (Bayesian ROPE on per-subject TOPSIS closeness)
  window_explanation(run_root, context_id, instance_id, k=3)
                                        -> prediction + confidence, top-k region / band / channel x band
                                           attributions of the recommended method, trust indicator
  build_export(run_root)                -> the whole JSON document (validated against schema.EXPORT_SCHEMA)
  modality_status(run_root)             -> is each prediction context (EEG, vision) available? (vision: "not
                                           available" until Component 2's trained model has been explained)

Vision windows (top-k face-region group / time segment / feature x segment) are exported under the
optional key `vision_windows`, so the EEG `windows` format is unchanged.

Trust indicator (0..1, shown as high / medium / low) = mean of three parts, each in [0, 1]:
  model confidence   (max probability - 0.5) / 0.5
  stability          the recommended method's run-to-run agreement on this window (Spearman, clipped at 0)
  agreement          mean Spearman of the recommended method with the other ranked methods (clipped at 0)
In-sample windows (deployed model trained on all subjects) are flagged in `notes` and capped at 'low'.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xaicomp.paths import safe_path

TRUST_LEVELS = {"high": 0.6, "medium": 0.35}


def _ctx_dirs(run_root: Path) -> list[Path]:
    return sorted(p.parent for p in run_root.glob("*/run_info.json"))


def _find_ctx(run_root: Path, context_id: str) -> Path:
    for d in _ctx_dirs(run_root):
        if json.loads((d / "run_info.json").read_text())["context_id"] == context_id:
            return d
    raise FileNotFoundError(f"context {context_id} not in {run_root}")


def recommendation(run_root: Path, context_id: str) -> dict:
    d = _find_ctx(run_root, context_id)
    info = json.loads((d / "run_info.json").read_text())
    rk = json.loads((d / "analysis" / "ranking.json").read_text())
    p1 = rk["p_rank1"]
    order = sorted(p1, key=lambda m: -p1[m])
    eq = rk.get("equivalence_top2", {})
    labels = rk.get("labels", {})
    rec, ru = order[0], order[1] if len(order) > 1 else order[0]
    equivalent = bool(eq.get("practically_equivalent", False))
    summary = (f"For {info['title']}, {labels.get(rec, rec)} was ranked first in {p1[rec]:.0%} of the "
               f"Monte-Carlo weightings; runner-up {labels.get(ru, ru)} ({p1[ru]:.0%}). "
               + ("The two are practically the same (inside the ROPE)." if equivalent else
                  "They are not shown to be practically the same."))
    return {"context_id": context_id, "title": info["title"], "modality": info.get("modality", "eeg"),
            "run_type": info["run_type"],
            "in_sample": bool(info["in_sample"]), "recommended_method": rec, "recommended_label": labels.get(rec, rec),
            "p_rank1": {k: float(v) for k, v in p1.items()}, "runner_up": ru,
            "practically_equivalent_to_runner_up": equivalent,
            "equivalence": {"p_recommended_better": eq.get("p_a_better"), "p_rope": eq.get("p_rope"),
                            "p_runner_up_better": eq.get("p_b_better"), "rope": float(eq.get("rope", 0.0)),
                            "measure": "per-subject TOPSIS closeness (reliability-weighted, equal base weights)"},
            "criteria_used": rk.get("criteria_used", []), "plain_summary": summary}


def _top(values: np.ndarray, names: list[str], k: int) -> list[dict]:
    idx = np.argsort(-np.abs(values), kind="stable")[:k]
    return [{"name": names[i], "attribution": float(values[i]),
             "direction": "towards_target" if values[i] > 0 else "away_from_target" if values[i] < 0 else "neutral"}
            for i in idx]


def window_explanation(run_root: Path, context_id: str, instance_id: str, k: int = 3, method: str | None = None,
                       seed: int | None = None) -> dict:
    from xaicomp.explainers.base import Explanation, explanation_path
    from xaicomp.representation.groups import InterpretableSpace

    d = _find_ctx(run_root, context_id)
    info = json.loads((d / "run_info.json").read_text())
    rec = recommendation(run_root, context_id)
    method = method or rec["recommended_method"]
    seed = info["seeds"][0] if seed is None else seed
    path = explanation_path(d, instance_id, method, seed)
    if not path.exists():
        raise FileNotFoundError(f"{instance_id} was not explained in this run ({path.name} missing)")
    e = Explanation.load(path)
    vision = info.get("modality") == "vision"
    if vision:
        from xaicomp.representation.vision_space import space_from_info
        space = space_from_info(info)
        levels = ("group", "segment", "feature", "feature_x_segment")
    else:
        space = InterpretableSpace(info["channels"], info["bands"], info["regions"])
        levels = ("region", "band", "channel_x_band")
    top = {lvl: _top(*space.aggregate(e.values, lvl), k) for lvl in levels}

    sel = pd.read_parquet(d / "selected.parquet").set_index("instance_id").loc[instance_id]
    p = float(e.fx)
    classes = info["class_names"]
    tgt = info["target_class"]
    pred = tgt if p >= 0.5 else [c for c in classes if c != tgt][0]
    conf = max(p, 1 - p)
    tables = d / "tables"
    stab = agree = np.nan
    if (tables / "stability.parquet").exists():
        st = pd.read_parquet(tables / "stability.parquet")
        r = st[(st["instance_id"] == instance_id) & (st["method"] == method)]
        stab = float(r["stability"].iloc[0]) if len(r) else np.nan
        ag = pd.read_parquet(tables / "agreement.parquet")
        ranked = set(info.get("ranked_methods", info["methods"])) - {"ig_zero"}
        a = ag[(ag["instance_id"] == instance_id) & ((ag["method_a"] == method) | (ag["method_b"] == method))]
        a = a[a["method_a"].isin(ranked) & a["method_b"].isin(ranked)]
        agree = float(a["rank_corr"].mean()) if len(a) else np.nan
    parts = {"model_confidence": (conf - 0.5) / 0.5,
             "stability": float(np.clip(stab, 0, 1)) if stab == stab else None,
             "agreement": float(np.clip(agree, 0, 1)) if agree == agree else None}
    vals = [v for v in parts.values() if v is not None]
    score = float(np.mean(vals)) if vals else 0.0
    level = "high" if score >= TRUST_LEVELS["high"] else "medium" if score >= TRUST_LEVELS["medium"] else "low"
    notes = []
    if info["in_sample"]:
        level = "low"
        notes.append(("IN-SAMPLE: " + info["label"].split("— ", 1)[-1]) if vision and info.get("label") else
                     "IN-SAMPLE: deployed model trained on all subjects; not for final results")
    if info["pilot"]:
        notes.append("PILOT run: reduced budget")
    out = {"context_id": context_id, "instance_id": instance_id, "subject_id": int(sel["subject_id"]),
           "t_start_s": float(sel["t_start"]), "t_end_s": float(sel["t_end"]), "method": method,
           "prediction": {"target_class": tgt, "p_target": p, "predicted_class": pred, "confidence": conf},
           "top_k": top, "trust": {"level": level, "score": score, "components": parts, "notes": notes},
           "in_sample": bool(info["in_sample"])}
    if vision:
        out.update(participant_id=str(sel["participant_id"]), video_id=str(sel["video_id"]))
    return out


VISION_WAITING = ("Waiting for Component 2's trained model — run scripts/check_vision_ready.py to see what is missing.")


def _ranked_contexts(run_root: Path) -> list[dict]:
    out = []
    for d in _ctx_dirs(run_root):
        info = json.loads((d / "run_info.json").read_text())
        if not info.get("negative_control") and (d / "analysis" / "ranking.json").exists():
            out.append(info)
    return out


def modality_status(run_root: Path, results_dir: Path | None = None) -> dict:
    """Availability of each prediction context for Component 4. A modality missing from this run is looked
    up in the other runs of results/ (EEG and vision are run separately)."""
    results_dir = results_dir or run_root.parent
    out = {}
    for mod in ("eeg", "vision"):
        here = [i["context_id"] for i in _ranked_contexts(run_root) if i.get("modality", "eeg") == mod]
        if here:
            out[mod] = {"status": "available", "run": run_root.name, "context_ids": here}
            continue
        other = None
        for r in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")) if results_dir.exists() else []:
            ids = [i["context_id"] for i in _ranked_contexts(r) if i.get("modality", "eeg") == mod]
            if ids and r != run_root:
                other = {"status": "available_in_other_run", "run": r.name, "context_ids": ids}
                break
        out[mod] = other or {"status": "not_available", "run": None, "context_ids": [],
                             "reason": VISION_WAITING if mod == "vision" else "no EEG results run yet"}
    return out


def build_export(run_root: Path, k: int = 3, save: bool = True) -> dict:
    from xaicomp.export.schema import validate

    contexts, windows, vision_windows, run_type, pilot = [], [], [], "", False
    for d in _ctx_dirs(run_root):
        info = json.loads((d / "run_info.json").read_text())
        if info.get("negative_control") or not (d / "analysis" / "ranking.json").exists():
            continue
        rec = recommendation(run_root, info["context_id"])
        contexts.append(rec)
        run_type, pilot = info["run_type"], info["pilot"]
        target = vision_windows if info.get("modality") == "vision" else windows
        for iid in pd.read_parquet(d / "selected.parquet")["instance_id"]:
            try:
                target.append(window_explanation(run_root, info["context_id"], iid, k))
            except FileNotFoundError:
                continue
    doc = {"schema_version": "1.0", "generated": dt.datetime.now().isoformat(timespec="seconds"),
           "run": {"name": run_root.name, "run_type": run_type, "pilot": pilot}, "contexts": contexts, "windows": windows,
           "modalities": modality_status(run_root)}
    if vision_windows:
        doc["vision_windows"] = vision_windows
    validate(doc)
    if save:
        with open(safe_path(run_root / "component4_export.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=float)
    return doc
