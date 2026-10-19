"""Phase 4: run a suite of classical experiments from a YAML config.

Usage:
  python scripts/run_baselines.py [--config experiments/classical.yaml] [--skip-existing]
  python scripts/run_baselines.py --smoke            # 4 subjects, all models, quick check
Each run writes results/<experiment_id>/ (see evaluation/reports.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from eegengine.config import RESULTS_DIR, load_config, set_seed
from eegengine.data import load_features
from eegengine.training.classical_runner import experiment_id, run_classical
from eegengine.training.cv import load_assignment


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/classical.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    F, meta, _ = load_features(cfg["features_config"])
    with open(RESULTS_DIR / "features" / "feature_summary.json", encoding="utf-8") as f:
        feat_lat = json.load(f)["single_epoch_latency_ms_median"]
    assignment = load_assignment(RESULTS_DIR / "folds_5fold.json") if cfg["cv"] == "kfold5" else None

    suffix = ""
    if args.smoke:
        keep = meta["subject_id"].isin([1, 2, 3, 4]).to_numpy()
        F, meta = F[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
        suffix = "_smoke"

    runs = [(t, m, fs, n) for t in cfg["tasks"] for fs in cfg["feature_sets"]
            for n in cfg["normalisations"] for m in cfg["models"]]
    for task, model, fset, norm in runs:
        eid = experiment_id(task, model, fset, norm, cfg["cv"], suffix)
        out = (RESULTS_DIR / "_smoke" / eid) if args.smoke else (RESULTS_DIR / eid)
        if args.skip_existing and (out / "metrics.json").exists():
            print(f"skip {eid}", flush=True)
            continue
        t = time.perf_counter()
        r = run_classical(F, meta, task=task, model=model, feature_set=fset, normalisation=norm, cv=cfg["cv"],
                          out_dir=out, seed=cfg.get("seed", 42), selection=cfg.get("selection"),
                          exclude_bad_windows=cfg.get("exclude_bad_windows", False), fold_assignment=assignment,
                          feature_latency_ms=feat_lat, n_jobs=cfg.get("n_jobs", 8))
        pf = r["per_fold"].get("macro_f1", {})
        print(f"{eid:55s} pooled F1 {r['pooled']['macro_f1']:.3f}  bal.acc {r['pooled']['balanced_accuracy']:.3f}  "
              f"AUROC {r['pooled']['auroc'] if r['pooled']['auroc'] is not None else float('nan'):.3f}  "
              f"fold F1 {pf.get('mean', float('nan')):.3f}±{pf.get('std', float('nan')):.3f}  "
              f"rec F1 {r['recording_level']['macro_f1']:.3f}  [{time.perf_counter() - t:.0f}s]", flush=True)
        if r["pooled"]["balanced_accuracy"] and r["pooled"]["balanced_accuracy"] > 0.97 and r["cv"] in ("loso", "kfold5"):
            print(f"  !! {eid}: cross-subject balanced accuracy > 0.97 — audit for leakage (docs/leakage_audit.md)")


if __name__ == "__main__":
    main()
