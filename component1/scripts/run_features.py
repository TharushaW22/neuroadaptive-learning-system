"""Phase 3: extract all feature families from the saved epochs (cached by config hash).

Usage:  python scripts/run_features.py [--config features.yaml] [--force] [--limit 200]
Writes: data/processed/features_<hash>.parquet         (n_epochs x n_features, family-prefixed)
        data/processed/features_<hash>_arrays.npz      bandpower (N,14,5), plv/coh (N,4,14,14)
        results/features/feature_summary.json          counts per family, timings, latency
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np

from eegengine.config import config_path, load_config, resolve
from eegengine.data import feature_hash, features_paths, load_epochs
from eegengine.features.extract import FAMILIES, extract_features, extract_features_parallel
from eegengine.io.stew import CHANNELS, FS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features.yaml")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--limit", type=int, help="smoke run on the first N epochs (not saved)")
    args = ap.parse_args()

    fcfg = load_config(args.config)
    pcfg = load_config(fcfg["preprocess_config"])
    fp, arp = features_paths(fcfg, pcfg)
    if fp.exists() and not args.force and not args.limit:
        print(f"Cached: {fp.name} (use --force to recompute)")
        return

    X, meta = load_epochs(pcfg)
    if args.limit:
        X = X[: args.limit]
    t0 = time.perf_counter()
    F, arrays, timings = extract_features_parallel(X, FS, CHANNELS, fcfg, n_jobs=fcfg.get("n_jobs", 4))
    elapsed = time.perf_counter() - t0
    assert np.isfinite(F.to_numpy()).all(), "non-finite feature values"

    if args.limit:
        print(f"[smoke] {len(F)} epochs x {F.shape[1]} features in {elapsed:.1f}s")
        return

    fp.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(fp, index=False)
    np.savez(arp, **arrays)

    # single-epoch latency (what the real-time engine pays per window), median of 20 runs
    lat = []
    for i in range(20):
        t = time.perf_counter()
        extract_features(X[i:i + 1], FS, CHANNELS, fcfg)
        lat.append((time.perf_counter() - t) * 1000)

    out = resolve(fcfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_hash": feature_hash(fcfg, pcfg),
        "features_file": fp.relative_to(resolve(".")).as_posix(),
        "n_epochs": int(len(F)),
        "n_features": int(F.shape[1]),
        "per_family": {f: int(sum(c.startswith(f + "_") for c in F.columns)) for f in FAMILIES},
        "extraction_s_total": round(elapsed, 1),
        "cpu_seconds_by_family": {k: round(v, 1) for k, v in timings.items()},
        "single_epoch_latency_ms_median": round(float(np.median(lat)), 2),
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        "graph_threshold": fcfg["graph"],
    }
    with open(out / "feature_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "feature_columns.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(F.columns))
    shutil.copy(config_path(args.config), out / "config.yaml")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
