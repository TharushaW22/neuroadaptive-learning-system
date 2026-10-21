"""Resumable experiment runner (Step 8): explanations + per-instance metrics for every context.

Layout of one run (results/<run_name>/<context slug>/):
  run_info.json            run type (PILOT / IN-SAMPLE / FINAL), budgets, model source, versions
  manifest.parquet         every candidate window with the model's prediction
  selected.parquet         the explained windows (seeded, stratified selection)
  explanations/<instance>/<method>__s<seed>[__noise<i>].npz     one Explanation per file
  metrics/<instance>.json  fidelity, sparsity, stability, robustness, agreement, cost for that window
  progress.json, errors.log

Resumable: an explanation file or metrics file that exists is never recomputed, so an interrupted
run continues where it stopped. Instances run in parallel worker processes, each single-threaded.
Measured on the Apple M1 dev machine: one process ~17 ms per model evaluation; with 3 parallel
workers ~30 ms each, i.e. ~1.7x the total throughput (more threads inside one process helped less).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from xaicomp.config import load_context, run_settings
from xaicomp.paths import RESULTS_DIR, safe_path

EVAL_SEED = 999          # donors of the fidelity evaluation: fixed, identical for every method

_W: dict = {}            # per-worker state (adapter, settings)


def context_slug(context_id: str) -> str:
    return context_id.replace("/", "__")


def run_type_tags(pilot: bool, in_sample: bool) -> list[str]:
    tags = ["PILOT"] if pilot else []
    if in_sample:
        tags.append("IN-SAMPLE")
    return tags or ["FINAL"]


def make_adapter(ctx: dict, threads: int = 1):
    if ctx["adapter"] == "eeg_component1":
        from xaicomp.adapters.eeg_component1 import EEGComponent1Adapter
        return EEGComponent1Adapter(ctx, threads=threads)
    if ctx["adapter"] == "vision_component2":
        from xaicomp.adapters.vision_component2 import VisionComponent2Adapter
        return VisionComponent2Adapter(ctx, threads=threads)
    raise ValueError(f"no adapter for {ctx['adapter']!r} (screening-only contexts are not run)")


def methods_for(settings: dict) -> list[str]:
    return list(settings["explainers"]["methods"])


# ---------------------------------------------------------------------------------- per instance
def _init_worker(ctx_name: str, pilot: bool, ctx_root: str, run_file: str = "run.yaml") -> None:
    import torch
    torch.set_num_threads(1)
    ctx = load_context(ctx_name)
    _W.update(ctx=ctx, settings=run_settings(pilot, run_file), root=Path(ctx_root), adapter=make_adapter(ctx, threads=1))


def _explain_one(ad, iid, method, seed, settings, root, noise_seed=None):
    from xaicomp.explainers.base import Explanation, explanation_path
    from xaicomp.explainers.registry import NEEDS_DONORS, explain

    variant = f"noise{noise_seed}" if noise_seed is not None else ""
    path = explanation_path(root, iid, method, seed, variant)
    if path.exists():
        return Explanation.load(path)
    b = settings["budget"]["budgets"][method]
    k = int(b.get("k_donors", 0)) if method in NEEDS_DONORS else 0
    view = ad.view(iid, k, seed, noise_seed=noise_seed, noise_scale=settings["metrics"]["robustness"]["noise_scale"])
    e = explain(method, view, seed, b, settings["explainers"]["methods"][method], ad.target_class)
    e.variant = variant
    e.save(path)
    return e


def process_instance(iid: str) -> dict:
    """Explain one window with every method and seed, then compute its metrics (worker side)."""
    from xaicomp.metrics.agreement import pairwise_agreement
    from xaicomp.metrics.fidelity import fidelity_for_instance
    from xaicomp.metrics.robustness import robustness
    from xaicomp.metrics.sparsity import sparsity
    from xaicomp.metrics.stability import stability

    ad, settings, root = _W["adapter"], _W["settings"], _W["root"]
    out_path = root / "metrics" / f"{iid}.json"
    if out_path.exists():
        return {"instance_id": iid, "skipped": True}
    t0 = time.perf_counter()
    budget, mcfg = settings["budget"], settings["metrics"]
    methods, seeds = methods_for(settings), list(budget["seeds"])
    ex = {(m, s): _explain_one(ad, iid, m, s, settings, root) for m in methods for s in seeds}

    # robustness: noisy copies, explained with the first seed; label flips are recorded, not scored
    fx0 = ex[(methods[0], seeds[0])].fx
    pred_is_target = fx0 >= 0.5
    robust_rows, flips = [], []
    for i in range(int(budget.get("n_noise", 1))):
        nv = ad.view(iid, 0, seeds[0], noise_seed=i, noise_scale=mcfg["robustness"]["noise_scale"])
        flipped = (nv.fx() >= 0.5) != pred_is_target
        flips.append({"noise": i, "fx_noisy": nv.fx(), "flipped": bool(flipped)})
        if flipped and mcfg["robustness"].get("require_same_label", True):
            continue
        for m in methods:
            en = _explain_one(ad, iid, m, seeds[0], settings, root, noise_seed=i)
            robust_rows.append({"method": m, "noise": i, **robustness(ex[(m, seeds[0])].values, en.values,
                                                                      mcfg["agreement"]["top_k"])})

    # fidelity of every (method, seed) attribution, both removal modes, one batched evaluation per mode
    fcfg = mcfg["fidelity"]
    eval_view = ad.view(iid, int(budget["budgets"]["fidelity"]["k_donors"]), EVAL_SEED)
    attributions = {f"{m}|s{s}": ex[(m, s)].values for m in methods for s in seeds}
    fid_rows, curves = fidelity_for_instance(eval_view, attributions, pred_is_target, int(fcfg["step_features"]),
                                             int(fcfg["n_random_orders"]), list(fcfg["modes"]), EVAL_SEED)
    for r in fid_rows:
        r["method"], r["seed"] = r["key"].split("|s")[0], int(r["key"].split("|s")[1])

    expl_rows = [{"method": m, "seed": s, "seconds": e.seconds, "n_evals": e.n_evals, "fx": e.fx,
                  "base_value": e.base_value, **sparsity(e.values, mcfg["sparsity"]["mass_fraction"]),
                  **{k: v for k, v in e.extras.items() if isinstance(v, (int, float, bool, str))}}
                 for (m, s), e in ex.items()]
    stab = {m: stability([ex[(m, s)].values for s in seeds], mcfg["agreement"]["top_k"]) for m in methods}
    agree = []
    for s in seeds:
        for row in pairwise_agreement({m: ex[(m, s)].values for m in methods}, mcfg["agreement"]["top_k"]):
            agree.append({"seed": s, **row})
    self_agree = []
    for m in methods:           # rho(A, A') between seeds: needed for the noise-corrected agreement
        st = stab[m]
        self_agree.append({"method": m, "self_rank_corr": st["stability"]})

    w = ad._windows.loc[iid]
    rec = {
        "instance_id": iid, "subject_id": int(w["subject_id"]), "condition": w["condition"], "fold": int(w["fold"]),
        "y": int(w["y"]), "frac_bad": float(w["frac_bad"]), "p": fx0, "pred_is_target": bool(pred_is_target),
        "confidence": float(max(fx0, 1 - fx0)), "explanations": expl_rows, "fidelity": fid_rows, "curves": curves,
        "stability": [{"method": m, **v} for m, v in stab.items()], "robustness": robust_rows, "noise": flips,
        "agreement": agree, "self_agreement": self_agree, "local_bp_corr": local_feature_dependence(ad, iid),
        "seconds_total": time.perf_counter() - t0,
    }
    with open(safe_path(out_path), "w", encoding="utf-8") as f:
        json.dump(rec, f, default=float)
    return {"instance_id": iid, "skipped": False, "seconds": rec["seconds_total"]}


def local_feature_dependence(ad, iid: str) -> float:
    """Feature dependence inside one window (mixed-model covariate): the adapter's own measure if it has
    one (vision: correlation between the 15 features across frames), else EEG's band-power correlation."""
    if hasattr(ad, "local_feature_correlation"):
        return ad.local_feature_correlation(iid)
    return local_bandpower_correlation(ad, iid)


def local_bandpower_correlation(ad, iid: str) -> float:
    """Local dependence between features: mean |Pearson r| between the per-epoch log powers of the
    84 components across the 8 epochs of the window (high = correlated features)."""
    from xaicomp.adapters.eeg_component1 import EPOCH_LEN, EPOCH_STEP

    comps, _ = ad.window_components(iid)
    P = np.stack([np.log10(np.mean(comps[..., j * EPOCH_STEP: j * EPOCH_STEP + EPOCH_LEN] ** 2, axis=-1) + 1e-12)
                  for j in range(ad.L)], axis=-1).reshape(-1, ad.L)
    P = P[P.std(axis=1) > 1e-9]
    R = np.corrcoef(P)
    iu = np.triu_indices_from(R, k=1)
    return float(np.nanmean(np.abs(R[iu])))


# ---------------------------------------------------------------------------------- per context
def prepare_context(ctx_name: str, settings: dict, run_root: Path, primary_selected: pd.DataFrame | None = None):
    from xaicomp.data.instances import build_manifest, select_control_instances, select_instances

    ctx = load_context(ctx_name)
    root = run_root / context_slug(ctx["context_id"])
    ad = make_adapter(ctx, threads=int(settings["budget"].get("torch_threads", 4)))
    old = root / "run_info.json"
    if old.exists():                       # never resume into a run made with another model / budget
        prev = json.loads(old.read_text(encoding="utf-8"))
        ident = ad.model_identity() if hasattr(ad, "model_identity") else None    # vision: which C2 checkpoint
        if prev.get("model_source") != ad.bank.source or prev.get("budget") != json.loads(json.dumps(settings["budget"], default=str)) \
                or prev.get("model_identity") != ident:
            raise RuntimeError(f"{root} was created with model source {prev.get('model_source')!r} / another budget / "
                               "another checkpoint; resuming would mix results. Use --run-name <new name> or move the old folder.")
    manifest = build_manifest(ad)
    b = settings["budget"]
    if ctx.get("negative_control"):
        if primary_selected is None:
            raise ValueError("the negative control needs the primary context's selection")
        ids = select_control_instances(primary_selected, int(b["control"]["subjects_per_fold"]),
                                       int(b["control"]["instances_per_subject"]))
        selected = manifest.set_index("instance_id").loc[ids].reset_index()
    else:
        selected = select_instances(manifest, b["subjects"], int(b["instances_per_subject"]), int(b["selection_seed"]))
    manifest.to_parquet(safe_path(root / "manifest.parquet"), index=False)
    selected.to_parquet(safe_path(root / "selected.parquet"), index=False)
    tags = run_type_tags(settings["pilot"], ad.in_sample)
    info = {
        "context_id": ctx["context_id"], "context_file": ctx_name, "title": ctx.get("title", ctx["context_id"]),
        "run_type": " + ".join(tags), "run_tags": tags, "in_sample": bool(ad.in_sample), "pilot": settings["pilot"],
        "negative_control": bool(ctx.get("negative_control", False)), "control_of": ctx.get("control_of"),
        "model_source": ad.bank.source, "weights": ctx.get("weights", "trained"),
        "label": ("IN-SAMPLE — deployed model trained on all subjects; not for final results" if ad.in_sample else ""),
        "n_instances": int(len(selected)), "n_subjects": int(selected["subject_id"].nunique()),
        "methods": methods_for(settings), "ranked_methods": settings["ranking"]["ranked_methods"],
        "seeds": settings["budget"]["seeds"], "budget": settings["budget"], "run_file": settings.get("run_file", "run.yaml"),
        "modality": ctx.get("modality", "eeg"), **space_info(ad.space),
        "class_names": ad.class_names, "target_class": ad.target_class,
        "created": dt.datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if hasattr(ad, "run_info_extra"):          # vision: model, data source, split check, conditional check
        info.update(ad.run_info_extra(), model_identity=ad.model_identity())
    with open(safe_path(root / "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return ctx, root, selected, ad


def space_info(space) -> dict:
    """Interpretable-space fields of run_info.json (EEG keeps its original keys)."""
    if hasattr(space, "describe"):
        return space.describe()
    return {"feature_names": space.feature_names, "channels": space.channels, "bands": space.bands,
            "regions": space.regions}


def run_context(ctx_name: str, settings: dict, run_root: Path, workers: int, primary_selected=None,
                limit: int | None = None, log=print) -> tuple[Path, pd.DataFrame]:
    ctx, root, selected, ad = prepare_context(ctx_name, settings, run_root, primary_selected)
    del ad
    ids = selected["instance_id"].tolist()[: limit or None]
    todo = [i for i in ids if not (root / "metrics" / f"{i}.json").exists()]
    log(f"[{ctx['context_id']}] {len(ids)} instances, {len(todo)} to do, {workers} workers")
    t0 = time.perf_counter()
    done, times = len(ids) - len(todo), []
    if todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
                                 initargs=(ctx_name, settings["pilot"], str(root), settings.get("run_file", "run.yaml"))) as pool:
            futs = {pool.submit(process_instance, i): i for i in todo}
            for fut in as_completed(futs):
                iid = futs[fut]
                try:
                    r = fut.result()
                    times.append(r.get("seconds", 0))
                except Exception:                     # keep going; the error is logged and can be re-run
                    with open(safe_path(root / "errors.log"), "a", encoding="utf-8") as fh:
                        fh.write(f"--- {iid} {dt.datetime.now():%H:%M:%S}\n{traceback.format_exc()}\n")
                    log(f"  ERROR on {iid} (see errors.log)")
                    continue
                done += 1
                el = time.perf_counter() - t0
                eta = el / max(1, done - (len(ids) - len(todo))) * (len(ids) - done)
                log(f"  {done}/{len(ids)} {iid} [{r.get('seconds', 0):.0f}s]  elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min")
                with open(safe_path(root / "progress.json"), "w", encoding="utf-8") as fh:
                    json.dump({"done": done, "total": len(ids), "elapsed_s": el, "eta_s": eta}, fh)
    return root, selected


def default_run_name(settings: dict) -> str:
    """pilot / full for EEG (run.yaml); vision_pilot / vision_full for run_vision.yaml (`run_names`)."""
    names = settings["run"].get("run_names", {"pilot": "pilot", "full": "full"})
    return names["pilot" if settings["pilot"] else "full"]


def run_experiment(pilot: bool, run_name: str | None = None, contexts: list[str] | None = None,
                   workers: int | None = None, limit: int | None = None, log=print, run_file: str = "run.yaml") -> Path:
    settings = run_settings(pilot, run_file)
    run_root = RESULTS_DIR / (run_name or default_run_name(settings))
    contexts = contexts or settings["run"]["contexts"]
    workers = workers or int(settings["budget"].get("workers", max(1, min(4, (os.cpu_count() or 2) - 2))))
    primary = None
    for c in sorted(contexts, key=lambda n: bool(load_context(n).get("negative_control"))):
        root, sel = run_context(c, settings, run_root, workers, primary, limit, log)
        if not load_context(c).get("negative_control"):
            primary = sel
    return run_root
