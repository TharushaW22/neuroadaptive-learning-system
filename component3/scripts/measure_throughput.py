"""Measure model-evaluation throughput with N parallel single-threaded workers (no other jobs running).

Each worker evaluates `n_evals` random donor coalitions of one real window and reports ms per evaluation.
Usage: .venv/bin/python scripts/measure_throughput.py --workers 4 --n-evals 800
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import _bootstrap  # noqa: F401


def work(args):
    iid, n = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from xaicomp.adapters.eeg_component1 import EEGComponent1Adapter
    from xaicomp.config import load_context
    ad = EEGComponent1Adapter(load_context("eeg_T1_A8"), threads=1)
    v = ad.view(iid, 4, 0)
    masks = np.random.default_rng(0).random((n // 4, 84)) < 0.5
    v.value(masks[:4], "donor")                                  # warm-up
    t = time.perf_counter()
    v.value(masks, "donor")
    return 1000 * (time.perf_counter() - t) / (len(masks) * 4)


def main():
    from xaicomp.paths import RESULTS_DIR, safe_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-evals", type=int, default=800)
    a = ap.parse_args()
    ids = ["s02_hi_e10", "s14_lo_e20", "s03_lo_e40", "s01_hi_e32", "s05_lo_e12", "s11_hi_e02"][: a.workers]
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
        ms = list(pool.map(work, [(i, a.n_evals) for i in ids]))
    out = {"workers": a.workers, "ms_per_eval_each": ms, "ms_per_eval_mean": sum(ms) / len(ms),
           "evals_per_second_total": sum(1000 / m for m in ms)}
    print(json.dumps(out))
    p = safe_path(RESULTS_DIR / "pilot" / "throughput.json")
    prev = json.loads(p.read_text()) if p.exists() else []
    json.dump(prev + [out], open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
