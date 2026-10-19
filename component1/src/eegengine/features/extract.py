"""Feature extraction entry point: epochs -> feature table + connectivity matrices.

The same function serves batch extraction (scripts/run_features.py) and the real-time
inference engine (one epoch at a time), so training and inference features are identical.
Families are selected by prefix: spectral_, entropy_, wavelet_, plv_, coh_, graph_.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from eegengine.features.connectivity import connectivity_matrices, upper_triangle_columns
from eegengine.features.entropy import entropy_features
from eegengine.features.graph import graph_features
from eegengine.features.spectral import spectral_features, welch_psd
from eegengine.features.wavelet import wavelet_features

FAMILIES = ("spectral", "entropy", "wavelet", "plv", "coh", "graph")


def extract_features(X: np.ndarray, fs: float, ch_names: list[str], cfg: dict, timings: dict | None = None):
    """Return (features DataFrame, arrays) for epochs X (N, ch, T).

    arrays = {"bandpower": (N, ch, 5) uV^2, "plv": (N, bands, ch, ch), "coh": (N, bands, ch, ch)}
    """
    fams = cfg.get("families", list(FAMILIES))
    conn_bands = cfg["connectivity"]["bands"]
    cols: dict[str, np.ndarray] = {}
    timings = timings if timings is not None else {}

    t = time.perf_counter()
    w = cfg.get("welch", {})
    freqs, psd = welch_psd(X, fs, w.get("nperseg", 256), w.get("noverlap"))
    spec_cols, bp = spectral_features(freqs, psd, ch_names)
    timings["spectral"] = timings.get("spectral", 0) + time.perf_counter() - t
    if "spectral" in fams:
        cols.update(spec_cols)

    if "entropy" in fams:
        t = time.perf_counter()
        cols.update(entropy_features(X, freqs, psd, ch_names))
        timings["entropy"] = timings.get("entropy", 0) + time.perf_counter() - t

    if "wavelet" in fams:
        t = time.perf_counter()
        wc = cfg.get("wavelet", {})
        cols.update(wavelet_features(X, ch_names, wc.get("name", "db4"), wc.get("level", 5)))
        timings["wavelet"] = timings.get("wavelet", 0) + time.perf_counter() - t

    t = time.perf_counter()
    c = cfg["connectivity"]
    plv, coh = connectivity_matrices(X, fs, conn_bands, c.get("coh_nperseg", 128), c.get("filter_order", 4))
    timings["connectivity"] = timings.get("connectivity", 0) + time.perf_counter() - t
    if "plv" in fams:
        cols.update(upper_triangle_columns(plv, "plv", conn_bands, ch_names))
    if "coh" in fams:
        cols.update(upper_triangle_columns(coh, "coh", conn_bands, ch_names))

    if "graph" in fams:
        t = time.perf_counter()
        g = cfg["graph"]
        cols.update(graph_features(plv, conn_bands, ch_names, g.get("threshold", "proportional"),
                                   g.get("proportion", 0.3), g.get("absolute", 0.5)))
        timings["graph"] = timings.get("graph", 0) + time.perf_counter() - t

    df = pd.DataFrame(cols)
    return df, {"bandpower": bp.astype(np.float32), "plv": plv, "coh": coh}


def _worker(args):
    X, fs, ch_names, cfg = args
    timings: dict = {}
    df, arrays = extract_features(X, fs, ch_names, cfg, timings)
    return df, arrays, timings


def extract_features_parallel(X: np.ndarray, fs: float, ch_names: list[str], cfg: dict, n_jobs: int = 4,
                              chunk: int = 256):
    """Chunked, multi-process version of `extract_features` for the full dataset."""
    chunks = [X[i:i + chunk] for i in range(0, len(X), chunk)]
    jobs = [(c, fs, ch_names, cfg) for c in chunks]
    if n_jobs <= 1:
        results = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_worker, jobs))
    df = pd.concat([r[0] for r in results], ignore_index=True)
    arrays = {k: np.concatenate([r[1][k] for r in results]) for k in results[0][1]}
    timings: dict = {}
    for r in results:
        for k, v in r[2].items():
            timings[k] = timings.get(k, 0) + v
    return df, arrays, timings


def family_columns(columns, families) -> list[str]:
    """Columns belonging to the given families (by prefix)."""
    fams = [families] if isinstance(families, str) else list(families)
    return [c for c in columns if c.split("_", 1)[0] in fams]
