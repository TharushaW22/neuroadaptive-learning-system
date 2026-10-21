"""Is everything in place to run the vision context? (READ-ONLY: writes nothing, anywhere.)

Used by scripts/check_vision_ready.py and by run_experiment.py --vision, which refuses to start
unless every required check passes. Each check names who must act: the Component 2 owner (trained
model, normaliser, extracted features — produced in C2's own Python 3.11 environment) or the
Component 3 student (environment problems on this side).

Checks
  1 c2_project        component2/vision_cbu exists
  2 trained_model     a REAL (not demo) temporal GRU checkpoint with label scheme daisee_binary
  3 normaliser        the training-split normaliser is stored in that checkpoint's metadata.json
  4 model_loads       the weights load into Component 3's copy of the GRU in this venv (versions reported)
  5 features          per-clip feature files exist, follow schema 1.0.0 and contain no synthetic rows
  6 splits_labels     train and test splits exist and have engagement labels 0-3
  7 subject_ids       every test clip has a real participant ID
  8 subject_disjoint  no test participant is in train/validation (else results are labelled IN-SAMPLE: warning)
  9 class_balance     both classes have windows in train and test (warning if the minority share < 20 %)
 10 model_vs_data     the checkpoint's training-data identity matches the features; test balanced accuracy (info)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

C2_OWNER = "Component 2 owner"
C3_OWNER = "Component 3 (you)"


@dataclass
class Check:
    name: str
    status: str            # pass | fail | warn | skipped
    detail: str
    owner: str = ""
    required: bool = True


def _versions_note(meta: dict) -> str:
    import numpy
    import torch
    lv = meta.get("library_versions") or {}
    here = {"torch": torch.__version__.split("+")[0], "numpy": numpy.__version__}
    parts = [f"{k} C2 {lv.get(k, '?')} / here {v}" for k, v in here.items()]
    return "; ".join(parts)


def check_vision_ready(ctx: dict, compute_accuracy: bool = True) -> dict:
    """Run every check for one vision context. Returns {"ready": bool, "checks": [...], "model_dir": ...}."""
    from xaicomp.data.vision_data import (LABEL_COLUMN, VisionDataError, c2_max_interpolate_gap, c2_paths,
                                          feature_files, label_index, normalise, read_feature_table, split_check,
                                          windows_from_table)
    from xaicomp.models.vision_c2 import (C2_FEATURE_COLUMNS, load_vision_model, read_metadata, select_model_dir)

    checks: list[Check] = []
    add = checks.append
    paths = c2_paths(ctx)
    expect = ctx["expect"]

    # 1 project
    if not paths["root"].exists():
        add(Check("c2_project", "fail", f"{paths['root']} not found", C2_OWNER))
        return _result(checks, None)
    add(Check("c2_project", "pass", str(paths["root"])))

    # 2 model, 3 normaliser
    model_dir, report = select_model_dir(paths["models_dir"], ctx["component2"].get("model_name", "auto"), expect)
    meta = None
    if model_dir is None:
        if report:
            why = "; ".join(f"{name}: {', '.join(p)}" for name, p in report.items())
        else:
            why = f"no model folders in {paths['models_dir']}"
        add(Check("trained_model", "fail",
                  f"no usable checkpoint ({why}). Needed: a GRU trained on REAL DAiSEE features with "
                  "labels.scheme = daisee_binary (binary_threshold 2), saved by C2's scripts/train_temporal.py "
                  "as models/<name>/model.pt + metadata.json", C2_OWNER))
        add(Check("normaliser", "skipped", "needs the trained model", C2_OWNER))
        add(Check("model_loads", "skipped", "needs the trained model", C2_OWNER))
    else:
        meta = read_metadata(model_dir)
        add(Check("trained_model", "pass", f"{model_dir.name} (label scheme {meta['label_scheme']['name']}, "
                                           f"cell {meta['model_params']['cell']}, created {meta.get('created_utc', '?')})"))
        add(Check("normaliser", "pass", f"{meta['normalizer']['method']} normaliser fitted on the training split, "
                                        "stored in metadata.json"))
        # 4 loads here
        try:
            import torch
            lm = load_vision_model(model_dir)
            with torch.inference_mode():
                out = lm.model(torch.zeros(1, int(meta["window"]), len(C2_FEATURE_COLUMNS)))
            ok = out.shape == (1, len(lm.class_names)) and bool(torch.isfinite(out).all())
            add(Check("model_loads", "pass" if ok else "fail",
                      f"loaded into Component 3's GRU copy; {_versions_note(meta)}", C3_OWNER))
        except Exception as e:  # report the exact problem; never hide it
            add(Check("model_loads", "fail", f"{type(e).__name__}: {e} ({_versions_note(meta)})", C3_OWNER))

    # 5 features
    files = feature_files(paths["features_dir"])
    if not files:
        add(Check("features", "fail", f"no per-clip feature files (*.csv.gz) in {paths['features_dir']}. Needed: "
                                      "C2's scripts/extract_features.py run on DAiSEE (all splits)", C2_OWNER))
        for n, o in (("splits_labels", C2_OWNER), ("subject_ids", C2_OWNER), ("subject_disjoint", C2_OWNER),
                     ("class_balance", C2_OWNER), ("model_vs_data", C2_OWNER)):
            add(Check(n, "skipped", "needs the extracted features", o))
        return _result(checks, model_dir)
    try:
        df = read_feature_table(paths["features_dir"])
    except VisionDataError as e:
        add(Check("features", "fail", str(e), C2_OWNER))
        return _result(checks, model_dir)
    add(Check("features", "pass", f"{len(files)} clip files, {len(df):,} frames, no synthetic rows"))

    # 6 splits + labels
    lab = df[df[LABEL_COLUMN].notna()]
    splits = set(lab["split"].astype(str))
    bad_levels = sorted(set(lab[LABEL_COLUMN].unique()) - {0.0, 1.0, 2.0, 3.0})
    if not {"train", "test"} <= splits or bad_levels:
        add(Check("splits_labels", "fail", f"labelled splits {sorted(splits)} (need train and test); "
                                           f"unexpected label values {bad_levels}", C2_OWNER))
    else:
        add(Check("splits_labels", "pass", f"labelled splits: {sorted(splits)}"))

    # 7 subject IDs, 8 disjointness
    test_p = set(lab.loc[lab["split"] == "test", "participant_id"].astype(str))
    if not test_p or "unknown" in test_p:
        add(Check("subject_ids", "fail", "test clips without a participant ID ('unknown'): keep DAiSEE's "
                                         "<split>/<subject>/<clip>/<clip>.avi layout when extracting", C2_OWNER))
    else:
        add(Check("subject_ids", "pass", f"{len(test_p)} test participants"))
    sc = split_check(lab[["participant_id", "split"]])
    if sc["subject_disjoint"]:
        add(Check("subject_disjoint", "pass", f"no test participant appears in train/validation "
                                              f"({sc['participants_per_split']}) -> test windows are OUT-OF-SAMPLE"))
    else:
        add(Check("subject_disjoint", "warn", f"overlap {sc['overlap_participants'][:10]} "
                                              f"(unknown IDs in test: {sc['test_has_unknown_participants']}) -> every "
                                              "vision result will be labelled IN-SAMPLE", C2_OWNER, required=False))

    # 9 windows + class balance, 10 model vs data
    if meta is None:
        add(Check("class_balance", "skipped", "needs the checkpoint's window settings and label mapping", C2_OWNER))
        add(Check("model_vs_data", "skipped", "needs the trained model", C2_OWNER))
        return _result(checks, model_dir)
    ws = windows_from_table(df, int(meta["window"]), int(meta["stride"]), float(meta["min_valid_ratio"]),
                            c2_max_interpolate_gap(ctx))
    y = label_index(meta, ws.y_level) if len(ws) else np.zeros(0, int)
    counts = {s: np.bincount(y[ws.splits == s], minlength=len(meta["label_scheme"]["class_names"])).tolist()
              for s in ("train", "validation", "test")}
    missing = [s for s in ("train", "test") if min(counts[s]) == 0]
    share = min(counts["test"]) / max(1, sum(counts["test"]))
    if missing:
        add(Check("class_balance", "fail", f"a class has no windows in {missing}: {counts}", C2_OWNER))
    else:
        add(Check("class_balance", "pass" if share >= 0.2 else "warn",
                  f"windows per class [low, high]: {counts}; test minority share {share:.0%}"
                  + ("" if share >= 0.2 else " (imbalanced: per-subject selection balances explained windows "
                                             "where possible; report it as a limitation)"),
                  C2_OWNER if share < 0.2 else "", required=False))
    notes = []
    td = meta.get("training_data") or {}
    n_vid = int(lab["video_id"].nunique())
    if td.get("n_videos") not in (None, n_vid):
        notes.append(f"checkpoint was trained on {td['n_videos']} labelled clips, features have {n_vid}")
    status = "warn" if notes else "pass"
    if compute_accuracy and any(c.name == "model_loads" and c.status == "pass" for c in checks) and counts["test"] and min(counts["test"]):
        try:
            lm = load_vision_model(model_dir)
            import torch
            te = np.where(ws.splits == "test")[0]
            X = normalise(lm, ws.X[te], ws.mask[te])
            with torch.inference_mode():
                pred = lm.model(torch.from_numpy(X), torch.from_numpy(ws.mask[te])).argmax(1).numpy()
            yt = y[te]
            bacc = float(np.mean([np.mean(pred[yt == c] == c) for c in np.unique(yt)]))
            notes.append(f"test balanced accuracy {bacc:.3f} on {len(te)} windows (information only)")
            if bacc < 0.55:
                status = "warn"
                notes.append("close to chance: check that the features and checkpoint belong together")
        except Exception as e:
            status = "warn"
            notes.append(f"could not score the test windows: {type(e).__name__}: {e}")
    add(Check("model_vs_data", status, "; ".join(notes) or "consistent", C2_OWNER if status == "warn" else "",
              required=False))
    return _result(checks, model_dir)


def _result(checks: list[Check], model_dir) -> dict:
    ready = not any(c.status in ("fail", "skipped") and c.required for c in checks)
    return {"ready": bool(ready), "model_dir": str(model_dir) if model_dir else None,
            "checks": [asdict(c) for c in checks]}


def format_report(res: dict) -> str:
    icon = {"pass": "✓", "fail": "✗", "warn": "!", "skipped": "·"}
    lines = ["Vision context readiness (Component 2 -> Component 3)", ""]
    for c in res["checks"]:
        who = f"  [{c['owner']}]" if c["status"] != "pass" and c["owner"] else ""
        lines.append(f" {icon.get(c['status'], '?')} {c['name']:<17} {c['status']:<8} {c['detail']}{who}")
    missing = [c for c in res["checks"] if c["status"] in ("fail", "skipped") and c["required"]]
    lines.append("")
    if res["ready"]:
        lines.append("READY: vision can run. Next: .venv/bin/python scripts/run_experiment.py --vision --pilot")
    else:
        owners = sorted({c["owner"] for c in missing if c["owner"] and c["status"] == "fail"})
        lines.append(f"NOT READY: {len(missing)} required item(s) missing or blocked. Who must act: {', '.join(owners)}.")
        lines.append("Step-by-step: component3/docs/VISION_RUNBOOK.md")
    return "\n".join(lines)
