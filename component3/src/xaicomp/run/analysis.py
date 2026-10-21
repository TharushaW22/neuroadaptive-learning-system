"""Analysis of a finished run (Steps 9-13), reading only the run's files (+ the model for the
fidelity-degradation check). Writes <context>/analysis/*.json|parquet and the C4 export.

  reliability   test-retest ICC, monotonic response to degradation, redundancy, randomised-weights control
  disagreement  noise-corrected agreement, mixed model, KernelSHAP vs Conditional SHAP value-function effect
  statistics    subject-level Friedman + Wilcoxon-Holm + rank-biserial, cluster-bootstrap CIs, Bayesian ROPE
  ranking       reliability-based criteria, TOPSIS, Monte-Carlo SMAA (several seeds)
  export        component4_export.json (validated)
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from xaicomp.config import load_context, run_settings
from xaicomp.paths import safe_path
from xaicomp.run import orchestrator as orch
from xaicomp.run.tables import build_tables


def _jdump(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return str(o)
    with open(safe_path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)


def load_explanations(ctx_root: Path) -> tuple[dict, dict]:
    from xaicomp.explainers.base import Explanation

    ex, noisy = {}, {}
    for p in (ctx_root / "explanations").glob("*/*.npz"):
        e = Explanation.load(p)
        if e.variant:
            if e.variant == "noise0":
                noisy[(e.instance_id, e.method)] = e.values
        else:
            ex[(e.instance_id, e.method, e.seed)] = e.values
    return ex, noisy


# ------------------------------------------------------------------ fidelity degradation (needs the model)
def _degradation_task(iid: str) -> list[dict]:
    from xaicomp.explainers.base import Explanation, explanation_path
    from xaicomp.reliability.degradation import fidelity_degradation_for_instance

    ad, settings, root = orch._W["adapter"], orch._W["settings"], orch._W["root"]
    seed = settings["budget"]["seeds"][0]
    methods = settings["ranking"]["ranked_methods"]
    attr = {m: Explanation.load(explanation_path(root, iid, m, seed)).values for m in methods}
    view = ad.view(iid, int(settings["budget"]["budgets"]["fidelity"]["k_donors"]), orch.EVAL_SEED)
    fcfg = settings["metrics"]["fidelity"]
    rows = fidelity_degradation_for_instance(view, attr, view.fx() >= 0.5, int(fcfg["step_features"]),
                                             int(fcfg["n_random_orders"]), list(fcfg["modes"]))
    return [{"instance_id": iid, **r} for r in rows]


def fidelity_degradation(ctx_name: str, ctx_root: Path, settings: dict, ids: list[str], workers: int) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=orch._init_worker,
                             initargs=(ctx_name, settings["pilot"], str(ctx_root), settings.get("run_file", "run.yaml"))) as pool:
        rows = [r for rs in pool.map(_degradation_task, ids) for r in rs]
    df = pd.DataFrame(rows)
    df["criterion"] = df["mode"].map({"on_manifold": "fidelity_on", "off_manifold": "fidelity_off"})
    return df.rename(columns={"fidelity": "value"})


# ------------------------------------------------------------------ main
def analyse_context(ctx_root: Path, settings: dict, control_root: Path | None, workers: int, log=print) -> dict:
    from xaicomp.disagreement.corrected_agreement import corrected_agreement
    from xaicomp.disagreement.mixed_effects import disagreement_frame, fit_mixed, value_function_effect
    from xaicomp.ranking.criteria import build_values, decide, final_criteria
    from xaicomp.ranking.smaa import smaa_multi_seed
    from xaicomp.ranking.topsis import topsis
    from xaicomp.reliability.degradation import attribution_only_degradation, summarise
    from xaicomp.reliability.randomisation import method_sanity, metric_collapse
    from xaicomp.reliability.redundancy import redundancy_matrix, redundant_groups
    from xaicomp.reliability.test_retest import test_retest
    from xaicomp.stats.aggregate import subject_level, wide
    from xaicomp.stats.bayesian import pairwise_bayes, signed_rank
    from xaicomp.stats.frequentist import bootstrap_matrix, cluster_bootstrap, friedman, pairwise_wilcoxon

    info = json.loads((ctx_root / "run_info.json").read_text())
    A = ctx_root / "analysis"
    T = build_tables(ctx_root)
    ranked = settings["ranking"]["ranked_methods"]
    seeds = list(info["seeds"])
    crit_cfg = settings["metrics"]["criteria"]
    criteria = list(crit_cfg)
    rules = settings["ranking"]["reliability_rules"]
    scfg = settings["stats"]
    ex, noisy = load_explanations(ctx_root)
    out: dict = {"context_id": info["context_id"]}

    # ---------------- Step 9 reliability
    log("  reliability ...")
    icc = test_retest(T, ex, seeds, ranked)
    deg_attr = attribution_only_degradation({k: v for k, v in ex.items() if k[1] in ranked},
                                            {k: v for k, v in noisy.items() if k[1] in ranked}, seeds)
    n_deg = int(settings["budget"].get("degradation_instances", 4))
    sel = pd.read_parquet(ctx_root / "selected.parquet")
    deg_ids = [i for i in sel.drop_duplicates("subject_id")["instance_id"] if (ctx_root / "metrics" / f"{i}.json").exists()][:n_deg]
    deg_fid = fidelity_degradation(info["context_file"], ctx_root, settings, deg_ids, workers)
    deg_all = pd.concat([deg_attr, deg_fid[["criterion", "level", "value"]] if len(deg_fid) else None])
    deg_sum = summarise(deg_all, rules["monotonic_min_abs_rho"])
    R = redundancy_matrix(T["instance_method"][T["instance_method"]["method"].isin(ranked)], criteria)
    groups = redundant_groups(R, rules["redundancy_rho"])
    sanity = collapse = None
    if control_root is not None and (control_root / "metrics").exists():
        Tc = build_tables(control_root)
        exc, _ = load_explanations(control_root)
        sanity = method_sanity(ex, exc, rules.get("sanity_max_similarity", 0.5))
        collapse = metric_collapse(T["instance_method"], Tc["instance_method"], criteria,
                                   rules.get("control_max_ratio", 0.5))
    icc.to_parquet(safe_path(A / "icc.parquet"), index=False)
    deg_all.to_parquet(safe_path(A / "degradation_values.parquet"), index=False)
    deg_sum.to_parquet(safe_path(A / "degradation.parquet"), index=False)
    R.to_parquet(safe_path(A / "redundancy.parquet"))
    if sanity is not None:
        sanity.to_parquet(safe_path(A / "control_sanity.parquet"), index=False)
        collapse.to_parquet(safe_path(A / "control_collapse.parquet"), index=False)
    out["reliability"] = {"redundant_groups": groups, "degradation_instances": deg_ids}

    # ---------------- Step 10 disagreement
    log("  disagreement ...")
    ca = corrected_agreement(T["agreement"], T["stability"])
    ca.to_parquet(safe_path(A / "corrected_agreement.parquet"), index=False)
    dfm = disagreement_frame(T["agreement"], T["instances"], exclude={"ig_zero"})
    dfm.to_parquet(safe_path(A / "disagreement_frame.parquet"), index=False)
    mixed = fit_mixed(dfm)
    vfe = value_function_effect(dfm)
    _jdump({"mixed_model": mixed, "value_function_effect": vfe}, A / "disagreement.json")

    # ---------------- Step 11 statistics (subject level)
    log("  statistics ...")
    subj = subject_level(T["instance_method"], criteria, ranked)
    subj.to_parquet(safe_path(A / "subject_level.parquet"), index=False)
    stats = {}
    for c in criteria:
        W = wide(subj, c, ranked)
        if W.empty:
            continue
        d = crit_cfg[c]["direction"]
        fr = friedman(W)
        wx = pairwise_wilcoxon(W, d, scfg["alpha"], scfg["multiple_testing"])
        bm, bd = cluster_bootstrap(W, scfg["bootstrap"]["n_boot"], scfg["bootstrap"]["ci"], scfg["bootstrap"]["seed"])
        by = pairwise_bayes(W, d, scfg["bayes"]["rope"][c], scfg["bayes"]["n_samples"], scfg["bayes"]["seed"])
        stats[c] = {"direction": d, "friedman": fr, "wilcoxon": wx, "bootstrap_means": bm, "bootstrap_diffs": bd,
                    "bayes": by, "n_subjects": len(W)}
    _jdump(stats, A / "statistics.json")

    # ---------------- Step 12 ranking
    log("  ranking ...")
    decisions = decide(criteria, icc, deg_sum, collapse, groups, rules)
    final = final_criteria(decisions, groups)
    raw_names = criteria
    methods = [m for m in ranked if m in set(subj["method"])]
    B = bootstrap_matrix(subj, raw_names, methods, scfg["bootstrap"]["n_boot"], scfg["bootstrap"]["seed"])
    V = build_values(B, raw_names, final)
    factors = np.array([f["factor"] for f in final])
    benefit = np.array([f["benefit"] for f in final])
    rcfg = settings["ranking"]["smaa"]
    sm = smaa_multi_seed(V, factors, benefit, int(rcfg["n_draws"]), float(rcfg["dirichlet_alpha"]), list(rcfg["seeds"]))
    point = subj.groupby("method")[raw_names].mean().loc[methods].to_numpy()
    Vp = build_values(point[None], raw_names, final)[0]
    cl_equal = topsis(Vp, factors / factors.sum(), benefit)
    # practical equivalence of the top two: Bayesian signed-rank on per-subject TOPSIS closeness
    order = list(np.argsort(-sm["p_rank1"]))
    eq = {}
    if len(order) > 1:
        per_subj = []
        for s, g in subj.groupby("subject_id"):
            g = g.set_index("method").reindex(methods)
            if g[raw_names].isna().all(axis=1).any():
                continue
            Vs = build_values(g[raw_names].to_numpy()[None], raw_names, final)[0]
            per_subj.append(topsis(np.nan_to_num(Vs, nan=float(np.nanmean(Vs))), factors / factors.sum(), benefit))
        per_subj = np.array(per_subj)
        rope = float(settings["ranking"].get("closeness_rope", 0.05))
        a, b = order[0], order[1]
        pa, pr, pb = signed_rank(per_subj[:, a], per_subj[:, b], rope, scfg["bayes"]["n_samples"], scfg["bayes"]["seed"])
        eq = {"method_a": methods[a], "method_b": methods[b], "p_a_better": pa, "p_rope": pr, "p_b_better": pb,
              "rope": rope, "practically_equivalent": bool(pr == pr and pr >= max(pa, pb)), "n_subjects": len(per_subj)}
    labels = {m: settings["explainers"]["methods"][m]["label"] for m in settings["explainers"]["methods"]}
    ranking = {"methods": methods, "labels": labels, "criteria_used": [f["name"] for f in final],
               "final_criteria": final, "decisions": decisions,
               "p_rank1": {m: float(p) for m, p in zip(methods, sm["p_rank1"])},
               "p_rank1_by_seed": {str(s): dict(zip(methods, map(float, row)))
                                   for s, row in zip(rcfg["seeds"], sm["p_rank1_by_seed"])},
               "mc_max_abs_diff": sm["mc_max_abs_diff"],
               "rank_acceptability": {m: list(map(float, sm["rank_acceptability"][i])) for i, m in enumerate(methods)},
               "central_weights": {m: list(map(float, sm["central_weights"][i])) for i, m in enumerate(methods)},
               "expected_rank": {m: float(sm["expected_rank"][i]) for i, m in enumerate(methods)},
               "topsis_equal_weights": {m: float(cl_equal[i]) for i, m in enumerate(methods)},
               "equivalence_top2": eq, "n_subjects": int(subj["subject_id"].nunique()),
               "n_draws": int(rcfg["n_draws"]), "seeds": rcfg["seeds"]}
    _jdump(ranking, A / "ranking.json")
    out["ranking"] = ranking
    return out


def context_capabilities(ctx: dict):
    """What a context offers to the screening, from its config only (no model or data is loaded)."""
    from xaicomp.adapters.base import Capabilities
    from xaicomp.adapters.tabular_component1 import tabular_capabilities
    from xaicomp.adapters.vision_baseline import vision_baseline_capabilities

    if ctx.get("adapter") == "vision_component2_baseline":
        return vision_baseline_capabilities(ctx)
    if ctx.get("screening_only"):
        return tabular_capabilities(ctx)
    if ctx.get("adapter") == "vision_component2":
        n = len([f for fs in ctx["groups"].values() for f in fs]) * int(ctx["representation"]["segments"])
        from xaicomp.adapters.vision_component2 import VISION_SCREENING_NOTES
        return Capabilities(differentiable=True, n_features=n, model_kind="deep (GRU)", supports_conditional=True,
                            notes=dict(VISION_SCREENING_NOTES))
    return Capabilities(differentiable=True, n_features=84, model_kind="deep", supports_conditional=True)


def screening_table(run_root: Path, settings: dict) -> pd.DataFrame:
    """Applicability screening for every configured context (+ the screening-only contexts:
    C1's XGBoost for EEG, C2's scikit-learn baseline for vision)."""
    from xaicomp.metrics.cost import ms_per_eval
    from xaicomp.screening.applicability import screen

    ms = None
    rows = []
    for d in sorted(p.parent for p in run_root.glob("*/run_info.json")):
        ex = d / "tables" / "explanations.parquet"
        if ex.exists():
            e = pd.read_parquet(ex)
            e = e[e["method"].isin(["kernel_shap", "conditional_shap", "lime"])]
            ms = ms_per_eval(e["seconds"].tolist(), e["n_evals"].tolist())
            break
    limit = float(settings["run"].get("max_seconds_per_instance", 600))
    budgets = settings["budget"]["budgets"]
    for name in settings["run"]["contexts"] + settings["run"].get("screening_only_contexts", []):
        ctx = load_context(name)
        caps = context_capabilities(ctx)
        df = screen(ctx["context_id"], caps, budgets, ms, limit)
        df["screening_only"] = bool(ctx.get("screening_only", False))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["ms_per_eval_measured"] = ms
    out.to_parquet(safe_path(run_root / "screening.parquet"), index=False)
    return out


def analyse_run(run_root: Path, pilot: bool, workers: int | None = None, log=print, run_file: str | None = None) -> dict:
    from xaicomp.export.component4 import build_export

    dirs = sorted(p.parent for p in run_root.glob("*/run_info.json"))
    infos = {d: json.loads((d / "run_info.json").read_text()) for d in dirs}
    # the run's own budget file (EEG run.yaml, vision run_vision.yaml); old EEG runs did not record it
    run_file = run_file or next((i["run_file"] for i in infos.values() if i.get("run_file")), "run.yaml")
    settings = run_settings(pilot, run_file)
    workers = workers or int(settings["budget"].get("workers", 4))
    control_of = {i["control_of"]: d for d, i in infos.items() if i.get("negative_control")}
    for d in dirs:                                           # tables for every context (incl. control)
        build_tables(d)
    screening_table(run_root, settings)
    results = {}
    for d, info in infos.items():
        if info.get("negative_control"):
            continue
        log(f"[{info['context_id']}] analysis")
        results[info["context_id"]] = analyse_context(d, settings, control_of.get(info["context_id"]), workers, log)
    doc = build_export(run_root)
    log(f"export: {len(doc['contexts'])} context(s), {len(doc['windows'])} window(s) -> component4_export.json")
    return results
