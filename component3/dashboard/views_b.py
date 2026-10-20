"""Dashboard pages 6-11: Reliability, Disagreement, Statistics, Ranking, Component 4, Glossary."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (CRITERIA_PLAIN, METHOD_COLOUR, MUTED, RANKED, RESULTS, SEQUENTIAL, analysis, banner,
                    control_for, fmt, howto, label, legend_row, meaning, method_chip, no_results, pct, plotly_layout,
                    status_badge, summary, table)
from views_a import current_ctx, words_for_corr

CRIT_SHORT = {"fidelity_on": "Fidelity (realistic)", "fidelity_off": "Fidelity (zero)", "stability": "Stability",
              "robustness": "Robustness", "sparsity": "Sparsity", "cost_seconds": "Cost"}


def crit_name(c: str) -> str:
    return " + ".join(CRIT_SHORT.get(x, x) for x in c.split(" + "))


# =========================================================================== 6 Reliability
def page_reliability(run: str):
    st.title("Can we trust the metrics? (Reliability)")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    rk = analysis(ctx, "ranking.json")
    icc = analysis(ctx, "icc.parquet")
    if rk is None or icc is None:
        no_results("reliability results")
        return
    dec = pd.DataFrame(rk["decisions"])
    kept = dec[dec["decision"] != "drop"]["criterion"].map(crit_name).tolist()
    dropped = dec[dec["decision"] == "drop"]["criterion"].map(crit_name).tolist()
    unchecked = dec[(dec["decision"] != "drop") & dec["reasons"].str.contains("not assessable")]["criterion"].map(crit_name).tolist()
    summary(f"A quality score is only useful if it gives the same value when measured again, gets worse when "
            f"explanations are made worse on purpose, and fails on a model with random weights. "
            f"<b>{len(kept)}</b> criteria are used in the ranking ({', '.join(kept) or 'none'}); "
            f"<b>{len(dropped)}</b> were dropped ({', '.join(dropped) or 'none'})."
            + (f" The repeatability of {', '.join(unchecked)} could not be checked in this run (needs more "
               "repeats), so they are kept but flagged." if unchecked else ""))

    st.markdown("#### Decision per criterion")
    rows = []
    for _, r in dec.iterrows():
        badge = status_badge({"keep": "pass", "down-weight": "caution", "drop": "fail"}[r["decision"]])
        badge = badge.replace("OK", "kept").replace("caution", "down-weighted").replace("excluded", "dropped")
        rows.append(f"<tr><td><b>{crit_name(r['criterion'])}</b></td><td>{badge}</td>"
                    f"<td>{fmt(r['weight_factor'])}</td><td style='color:{MUTED}'>{r['reasons']}</td></tr>")
    st.markdown("<table style='width:100%'><tr><th>Criterion</th><th>Decision</th><th>Weight factor</th><th>Why</th></tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)

    st.markdown("#### Same value when measured again? (test-retest ICC)")
    d = icc.dropna(subset=["icc"])
    if len(d):
        fig = go.Figure(go.Bar(x=d["icc"], y=d["criterion"].map(crit_name), orientation="h", marker_color="#1F4E79",
                               text=[fmt(v) for v in d["icc"]], textposition="outside",
                               hovertemplate="%{y}: ICC %{x:.2f}<extra></extra>"))
        for x, t in ((0.5, "poor below"), (0.75, "good above")):
            fig.add_vline(x=x, line=dict(color="#9AA5B1", dash="dot"), annotation_text=t, annotation_position="top")
        fig.update_layout(xaxis=dict(range=[min(0, d["icc"].min() - 0.05), 1.05], title="ICC (1 = perfectly repeatable)"))
        st.plotly_chart(plotly_layout(fig, 120 + 45 * len(d)), use_container_width=True)
        howto("Each bar is how well the metric repeats when the explanation is recomputed with a new random seed. "
              "Below 0.5 the metric is too noisy to compare methods and is dropped.")
    na = icc[icc["icc"].isna()]
    if len(na):
        st.caption("Not assessable in this run: " + ", ".join(f"{crit_name(c)} ({r})" for c, r in zip(na["criterion"], na["repeats"])))
    meaning("ICC", "The intraclass correlation compares the differences between windows/methods with the differences "
                   "between repeated measurements of the same window. 1 = repeat measurements are identical.", "+")

    st.markdown("#### Does the metric get worse when explanations are made worse on purpose?")
    dv = analysis(ctx, "degradation_values.parquet")
    if dv is not None and len(dv):
        g = dv.groupby(["criterion", "level"])["value"].mean().reset_index()
        crits = list(g["criterion"].unique())
        cols = st.columns(min(4, len(crits)))
        for i, c in enumerate(crits):
            gc = g[g["criterion"] == c]
            fig = go.Figure(go.Scatter(x=gc["level"] * 100, y=gc["value"], mode="lines+markers",
                                       line=dict(color="#1F4E79", width=2), marker=dict(size=8),
                                       hovertemplate="%{x:.0f}% noise: %{y:.3f}<extra></extra>"))
            fig.update_layout(title=dict(text=crit_name(c), font=dict(size=14)), xaxis_title="% replaced by noise")
            cols[i % len(cols)].plotly_chart(plotly_layout(fig, 260), use_container_width=True)
        howto("Moving right, explanations are increasingly replaced by random noise. A trustworthy metric falls "
              "steadily; a flat or rising line means the metric does not track explanation quality.")

    st.markdown("#### Do two metrics measure the same thing? (redundancy)")
    R = analysis(ctx, "redundancy.parquet")
    if R is not None and len(R):
        names = [crit_name(c) for c in R.columns]
        fig = go.Figure(go.Heatmap(z=np.abs(R.to_numpy()), x=names, y=names, colorscale=SEQUENTIAL, zmin=0, zmax=1,
                                   text=np.round(R.to_numpy(), 2), texttemplate="%{text}", xgap=2, ygap=2,
                                   hovertemplate="%{y} vs %{x}: rho %{text}<extra></extra>",
                                   colorbar=dict(title="|rho|")))
        st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)
        howto("Darker = the two metrics rank windows and methods almost identically (number = Spearman correlation). "
              "Pairs above 0.9 are merged so one property is not counted twice.")

    st.markdown("#### Negative control: a model with random weights")
    san = analysis(ctx, "control_sanity.parquet")
    col = analysis(ctx, "control_collapse.parquet")
    if san is None:
        st.info("The randomised-weights control has not been run for this context.")
    else:
        c1, c2 = st.columns(2)
        rows = "".join(f"<tr><td>{method_chip(r['method'])}</td><td>{fmt(r['mean_similarity'])}</td>"
                       f"<td>{status_badge('pass' if r['passes_sanity_check'] else 'fail').replace('excluded', 'fails')}</td></tr>"
                       for _, r in san.iterrows())
        c1.markdown("**Do explanations change when the model is destroyed?**<table style='width:100%'><tr><th>Method"
                    "</th><th>Similarity to trained</th><th>Sanity check</th></tr>" + rows + "</table>",
                    unsafe_allow_html=True)
        c1.caption("Low similarity = good: the method really explains the model, not the input alone.")
        if col is not None and len(col):
            fc = col[col["criterion"].str.startswith("fidelity")]
            fig = go.Figure()
            fig.add_bar(x=fc["criterion"].map(crit_name), y=fc["trained_mean"], name="Trained model", marker_color="#1F4E79")
            fig.add_bar(x=fc["criterion"].map(crit_name), y=fc["control_mean"], name="Random weights", marker_color="#9AA5B1")
            fig.update_layout(barmode="group", yaxis_title="Mean fidelity")
            c2.plotly_chart(plotly_layout(fig, 300), use_container_width=True)
            c2.caption("Fidelity should collapse towards 0 on the random model (it has nothing to be faithful to).")


# =========================================================================== 7 Disagreement
def page_disagreement(run: str):
    st.title("Why do methods disagree? (Disagreement)")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    ca = analysis(ctx, "corrected_agreement.parquet")
    dj = analysis(ctx, "disagreement.json")
    if ca is None or dj is None:
        no_results("disagreement results")
        return
    vfe = dj.get("value_function_effect", {})
    r_main = ca[ca["method_a"].isin(RANKED) & ca["method_b"].isin(RANKED)]["raw_agreement"].mean()
    summary(f"Different methods often name different features. On average the four main methods agree "
            f"<b>{words_for_corr(r_main)}</b> (rank correlation {fmt(r_main)}). Part of this is random noise inside "
            "each method; the corrected values below remove that part. "
            + (f"KernelSHAP and Conditional SHAP differ only in how hidden EEG is filled in; their disagreement is "
               f"{fmt(vfe.get('mean_disagreement'))} versus {fmt(vfe.get('mean_disagreement_other_pairs'))} for other pairs."
               if vfe.get("available") else ""))

    methods = [m for m in RANKED + ["ig_zero"] if m in set(ca["method_a"]) | set(ca["method_b"])]
    M = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    C = M.copy() * np.nan
    for _, r in ca.iterrows():
        M.loc[r["method_a"], r["method_b"]] = M.loc[r["method_b"], r["method_a"]] = r["raw_agreement"]
        C.loc[r["method_a"], r["method_b"]] = C.loc[r["method_b"], r["method_a"]] = r["corrected_agreement"]
    names = [label(m) for m in methods]
    c1, c2 = st.columns(2)
    for col, mat, title in ((c1, M, "Raw agreement"), (c2, C, "Corrected for each method's own noise")):
        z = mat.to_numpy(dtype=float)
        txt = np.where(np.isnan(z), "–", np.vectorize(lambda v: fmt(v))(np.nan_to_num(z)))
        fig = go.Figure(go.Heatmap(z=np.clip(z, 0, 1), x=names, y=names, colorscale=SEQUENTIAL, zmin=0, zmax=1, text=txt,
                                   texttemplate="%{text}", xgap=2, ygap=2, showscale=False,
                                   hovertemplate="%{y} vs %{x}: %{text}<extra></extra>"))
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
        col.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
    howto("Darker = the two methods rank features more similarly (1 = identical). '–' = not corrected (the zero-reference "
          "IG is deterministic, or a method's own repeatability is not above 0). Corrected values above 1 are flagged below.")
    flagged = ca[ca["corrected_gt_1"]]
    if len(flagged):
        st.warning("Corrected agreement above 1 (the correction is unreliable because the methods' own repeatability is "
                   "small): " + ", ".join(f"{label(a)} vs {label(b)}" for a, b in zip(flagged["method_a"], flagged["method_b"])))
    meaning("noise-corrected agreement", "Two noisy methods can never agree perfectly, even if they explain the same "
            "thing. Dividing their agreement by the square root of each method's agreement with itself (across "
            "seeds) estimates how much they would agree without that noise.", "+")

    st.markdown("#### What makes methods disagree more?")
    mm = dj.get("mixed_model", {})
    coefs = [c for c in mm.get("coefficients", []) if c["term"].startswith("z_")]
    if coefs:
        plain = {"z_confidence": "Model is more confident", "z_lime_r2": "Model is locally more linear (LIME R²)",
                 "z_local_bp_corr": "Band powers are more correlated", "z_frac_bad": "More noisy epochs"}
        fig = go.Figure()
        for c in coefs:
            fig.add_scatter(x=[c["coef"]], y=[plain.get(c["term"], c["term"])], mode="markers",
                            marker=dict(size=12, color="#1F4E79"),
                            error_x=dict(type="data", symmetric=False, array=[c["ci_high"] - c["coef"]],
                                         arrayminus=[c["coef"] - c["ci_low"]], color="#7B8794", width=6),
                            showlegend=False, hovertemplate=f"effect {fmt(c['coef'], 3)}, p = {fmt(c['p_value'], 2)}<extra></extra>")
        fig.add_vline(x=0, line=dict(color="#9AA5B1"))
        fig.update_layout(xaxis_title="Change in disagreement per 1 standard deviation (← less, more →)")
        st.plotly_chart(plotly_layout(fig, 300), use_container_width=True)
        howto("Each row is one property of the EEG window. A dot right of the line means windows with more of that "
              "property have more disagreement; the line through the dot is the 95% confidence interval. Lines "
              "crossing zero = no clear effect.")
        st.caption(f"Linear mixed model, {mm.get('n')} window-pairs from {mm.get('n_subjects')} subjects, "
                   f"random intercept per subject; converged: {mm.get('converged')}.")
    else:
        st.info("The mixed model could not be fitted for this run (" + str(mm.get("error", "too little data")) + ").")
    if vfe.get("available"):
        with st.expander("KernelSHAP vs Conditional SHAP — the effect of filling hidden EEG realistically"):
            st.write(f"Mean disagreement {fmt(vfe['mean_disagreement'])} (other pairs {fmt(vfe['mean_disagreement_other_pairs'])}). "
                     f"Correlation of this disagreement with band-power correlation: {fmt(vfe['spearman_with_local_bp_corr'])}; "
                     f"with the share of noisy epochs: {fmt(vfe['spearman_with_frac_bad'])}.")


# =========================================================================== 8 Statistics
def page_statistics(run: str):
    st.title("Are the differences real? (Statistics)")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    stats = analysis(ctx, "statistics.json")
    if not stats:
        no_results("statistics")
        return
    sig = [c for c, s in stats.items() if s["friedman"]["p_value"] == s["friedman"]["p_value"] and s["friedman"]["p_value"] < 0.05]
    n = next(iter(stats.values()))["n_subjects"]
    summary(f"All tests treat each <b>subject</b> as one unit ({n} subjects), because windows from the same person "
            f"are not independent. The methods differ clearly (p below 0.05) on <b>{len(sig)}</b> of {len(stats)} "
            f"criteria{': ' + ', '.join(crit_name(c) for c in sig) if sig else ''}.")
    crit = st.selectbox("Criterion", list(stats), format_func=lambda c: CRITERIA_PLAIN.get(c, (c,))[0])
    s = stats[crit]
    fr = s["friedman"]
    st.write(f"**Do the methods differ at all?** Friedman test: p = {fmt(fr['p_value'], 2)}; effect size "
             f"(Kendall's W, 0–1) = {fmt(fr['kendalls_w'])}.")
    by = pd.DataFrame(s["bayes"])
    if len(by):
        st.markdown("#### Chance that one method is better, or that they are practically the same")
        fig = go.Figure()
        ylab = [f"{label(a)} vs {label(b)}" for a, b in zip(by["method_a"], by["method_b"])]
        for i, r in by.iterrows():
            parts = [(r["p_a_better"], METHOD_COLOUR.get(r["method_a"], "#999"), f"{label(r['method_a'])} better"),
                     (r["p_rope"], "#CBD2D9", "practically the same"),
                     (r["p_b_better"], METHOD_COLOUR.get(r["method_b"], "#999"), f"{label(r['method_b'])} better")]
            base = 0.0
            for v, colr, name in parts:
                fig.add_bar(x=[v], y=[ylab[i]], base=[base], orientation="h", marker=dict(color=colr, line=dict(color="white", width=2)),
                            text=[pct(v) if v >= 0.08 else ""], textposition="inside", insidetextanchor="middle",
                            showlegend=False, hovertemplate=f"{name}: {pct(v)}<extra></extra>")
                base += v
        fig.update_layout(barmode="overlay", xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"))
        st.plotly_chart(plotly_layout(fig, 100 + 48 * len(by)), use_container_width=True)
        howto("Each row compares two methods. The coloured part on the left is the chance the first method is better, "
              "the grey middle the chance they are practically the same (difference inside the ROPE), the right part "
              f"the chance the second is better. ROPE for this criterion: ±{fmt(by['rope'].iloc[0])}.")
        legend_row(RANKED)
    wx = pd.DataFrame(s["wilcoxon"])
    with st.expander("Pairwise tests (Wilcoxon signed-rank, Holm-corrected) and confidence intervals"):
        if len(wx):
            st.dataframe(wx.assign(pair=wx["method_a"].map(label) + " vs " + wx["method_b"].map(label))[
                ["pair", "mean_diff_a_minus_b", "p_value", "p_holm", "significant", "rank_biserial_a_better"]].rename(
                columns={"mean_diff_a_minus_b": "difference (A − B)", "p_value": "p", "p_holm": "p (Holm)",
                         "rank_biserial_a_better": "effect size (A better, −1..1)"}).round(3), hide_index=True)
        bd = pd.DataFrame(s["bootstrap_diffs"])
        if len(bd):
            st.dataframe(bd.assign(pair=bd["method_a"].map(label) + " vs " + bd["method_b"].map(label))[
                ["pair", "diff", "ci_low", "ci_high"]].rename(columns={"diff": "difference", "ci_low": "95% CI low",
                                                                      "ci_high": "95% CI high"}).round(3), hide_index=True)
    meaning("the tests", "Friedman asks whether any method differs; Wilcoxon compares each pair; Holm's correction "
            "keeps the chance of a false alarm at 5% across all pairs. The Bayesian test adds what p-values cannot: "
            "the probability that two methods are practically the same.")


# =========================================================================== 9 Ranking
def page_ranking(run: str):
    st.title("Final recommendation (Ranking)")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    rk = analysis(ctx, "ranking.json")
    if not rk:
        no_results("ranking")
        return
    p1 = rk["p_rank1"]
    order = sorted(p1, key=lambda m: -p1[m])
    eq = rk.get("equivalence_top2", {})
    summary(f"Nobody has to choose how much each quality criterion matters: we tried {rk['n_draws']:,} random "
            f"weightings (× {len(rk['seeds'])} repeats) and re-sampled the subjects each time. "
            f"<b>{label(order[0])}</b> ranked first in <b>{pct(p1[order[0]])}</b> of them, "
            f"{label(order[1])} in {pct(p1[order[1]])}. "
            + ("The top two are <b>practically the same</b> (inside the ROPE)." if eq.get("practically_equivalent")
               else "The top two are not shown to be practically the same."))
    fig = go.Figure(go.Bar(x=[p1[m] for m in order], y=[label(m) for m in order], orientation="h",
                           marker=dict(color=[METHOD_COLOUR.get(m) for m in order], line=dict(color="white", width=2)),
                           text=[pct(p1[m]) for m in order], textposition="outside",
                           hovertemplate="%{y}: ranked first in %{x:.0%} of weightings<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1.1], tickformat=".0%", title="Chance of ranking first (SMAA rank-1 probability)"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(plotly_layout(fig, 90 + 55 * len(order)), use_container_width=True)
    howto("Longer bar = the method comes out on top under more possible priorities. Bars sum to 100%.")

    st.markdown("#### Chance of each rank")
    shades = ["#1F4E79", "#4F7CAC", "#8FB0D2", "#D3E0EE"]
    fig = go.Figure()
    for r in range(len(order)):
        fig.add_bar(x=[rk["rank_acceptability"][m][r] for m in order], y=[label(m) for m in order], orientation="h",
                    name=f"rank {r + 1}", marker=dict(color=shades[r % 4], line=dict(color="white", width=2)),
                    text=[pct(rk["rank_acceptability"][m][r]) if rk["rank_acceptability"][m][r] >= 0.08 else "" for m in order],
                    textposition="inside", hovertemplate="%{y}: rank " + str(r + 1) + " in %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"))
    st.plotly_chart(plotly_layout(fig, 90 + 55 * len(order)), use_container_width=True)
    howto("Darkest = first place, lightest = last place. Each row adds up to 100%.")

    with st.expander("Which criteria were used, and how stable is the result?"):
        st.write("Criteria used: " + ", ".join(f"{crit_name(f['name'])} (weight factor {fmt(f['factor'])})"
                                               for f in rk["final_criteria"]))
        st.write(f"Largest change in any 'chance of ranking first' between Monte-Carlo repeats: {pct(rk['mc_max_abs_diff'])} "
                 "(small = the result is not a fluke of the random draws).")
        st.dataframe(pd.DataFrame(rk["p_rank1_by_seed"]).T.rename(columns=label).round(3), use_container_width=True)
        if eq:
            st.write(f"Top two compared per subject (TOPSIS score, ROPE ±{eq['rope']}): "
                     f"P({label(eq['method_a'])} better) = {pct(eq['p_a_better'])}, P(same) = {pct(eq['p_rope'])}, "
                     f"P({label(eq['method_b'])} better) = {pct(eq['p_b_better'])}.")
        cw = pd.DataFrame(rk["central_weights"], index=[crit_name(f["name"]) for f in rk["final_criteria"]]).rename(columns=label)
        st.write("Typical priorities under which each method wins (average weights when it ranks first):")
        st.dataframe(cw.round(2), use_container_width=True)
    meaning("SMAA and TOPSIS", "TOPSIS scores each method by how close it is to an imaginary 'best on every criterion' "
            "method. SMAA repeats this for thousands of random weightings and re-sampled subjects, so the result "
            "does not depend on one person's choice of weights.")


# =========================================================================== 10 C4 output
def page_c4(run: str):
    st.title("Output for Component 4")
    banner(run)
    p = RESULTS / run / "component4_export.json"
    if not p.exists():
        no_results("export")
        return
    doc = json.loads(p.read_text())
    if not doc["contexts"]:
        no_results("export")
        return
    c = doc["contexts"][0]
    summary(f"Component 4 (the teaching agent) receives, for each kind of prediction, which explanation method to "
            f"trust — here <b>{c['recommended_label']}</b> ({pct(c['p_rank1'][c['recommended_method']])} chance of "
            "ranking first) — and, for each EEG window, the brain regions and bands that drove the prediction with a "
            "trust level.")
    for c in doc["contexts"]:
        st.markdown(f"**{c['title']}** — {c['plain_summary']}")
    if doc["windows"]:
        ids = [w["instance_id"] for w in doc["windows"]]
        iid = st.selectbox("Example window", ids)
        w = doc["windows"][ids.index(iid)]
        c1, c2 = st.columns([1, 2])
        tl = w["trust"]["level"]
        c1.markdown(f'<div class="card"><h4>Prediction</h4><div class="big">{w["prediction"]["predicted_class"]} workload</div>'
                    f'<div class="sub">confidence {pct(w["prediction"]["confidence"])}</div><br><h4>Trust</h4>'
                    f'<div>{status_badge({"high": "pass", "medium": "caution", "low": "fail"}[tl]).replace("OK", "high").replace("caution", "medium").replace("excluded", "low")}'
                    f' (score {fmt(w["trust"]["score"])})</div><div class="sub">{"; ".join(w["trust"]["notes"])}</div></div>',
                    unsafe_allow_html=True)
        rows = []
        for lvl, name in (("region", "Brain region"), ("band", "Frequency band"), ("channel_x_band", "Electrode × band")):
            for it in w["top_k"][lvl]:
                arrow = "↑ towards high" if it["direction"] == "towards_target" else "↓ towards low"
                rows.append({"level": name, "feature": it["name"], "effect": fmt(it["attribution"], 3), "direction": arrow})
        c2.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Download the export (JSON)", p.read_bytes(), file_name="component4_export.json", mime="application/json")
    with st.expander("How Component 4 can use it (Python)"):
        st.code("from xaicomp.export.component4 import recommendation, window_explanation\n"
                f"rec = recommendation(Path('component3/results/{run}'), '{doc['contexts'][0]['context_id']}')\n"
                "win = window_explanation(run_root, context_id, instance_id, k=3)", language="python")
    with st.expander("Raw JSON (first context)"):
        st.json(doc["contexts"][0], expanded=False)


# =========================================================================== 11 Glossary
GLOSSARY = {
    "Explanation (attribution)": "A score for each part of the input saying how much it pushed the prediction up or down.",
    "Feature (channel × band)": "One electrode's signal in one frequency band, e.g. the alpha rhythm at O1. There are 84.",
    "Frequency band": "A range of brain-wave speeds: slow (below 1 Hz), delta, theta, alpha, beta, gamma.",
    "Region": "A group of electrodes: frontal, temporal, parietal, occipital.",
    "KernelSHAP": "Estimates each feature's fair share of the prediction by hiding random groups of features and "
                  "filling them with EEG from other people.",
    "Conditional SHAP": "Like KernelSHAP, but the filled-in EEG is made realistic given the parts that are still visible.",
    "LIME": "Fits a simple straight-line model around one window and reads its weights.",
    "Integrated Gradients (IG)": "Follows the model's gradients along a path from a reference signal to the real one.",
    "Reference / baseline": "What 'absent' means: other people's EEG (donor) or zero.",
    "Fidelity": "Whether removing the parts an explanation calls important really changes the prediction.",
    "Deletion / insertion curve": "The prediction while removing (or adding back) features in order of importance.",
    "On-/off-manifold": "Removing a feature by replacing it with realistic EEG (on) or with zeros (off, unrealistic).",
    "Stability": "Whether the method gives the same explanation when run again with another random seed.",
    "Robustness": "Whether the explanation stays the same when a little noise is added to the EEG.",
    "Sparsity (Gini)": "Whether a few features carry most of the importance (easier to read).",
    "Cost": "Time to explain one window.",
    "Spearman rank correlation": "Agreement between two rankings: 1 identical, 0 unrelated, −1 reversed.",
    "ICC (test-retest)": "How repeatable a measurement is: 1 = perfectly repeatable, below 0.5 = poor.",
    "Controlled degradation": "Deliberately mixing explanations with noise to check that a metric notices.",
    "Negative control (random weights)": "A copy of the model with random weights; good explanations and metrics must react to it.",
    "Noise-corrected agreement": "Agreement between two methods after removing each method's own randomness.",
    "Mixed model": "A regression that allows each subject their own baseline, because windows of one person are related.",
    "Friedman test": "Checks whether several methods differ, using each subject as one block.",
    "Wilcoxon signed-rank test": "Compares two methods subject by subject.",
    "Holm correction": "Adjusts p-values when many pairs are tested so false alarms stay rare.",
    "Effect size (rank-biserial)": "How consistently one method beats the other, from −1 to 1.",
    "Bootstrap confidence interval": "A range of plausible averages, found by re-sampling subjects many times.",
    "ROPE": "Region of practical equivalence: differences smaller than this are treated as 'practically the same'.",
    "Bayesian signed-rank test": "Gives the probabilities that A is better, that they are the same (ROPE), or that B is better.",
    "TOPSIS": "Scores each method by its closeness to an imaginary method that is best on every criterion.",
    "SMAA": "Repeats the ranking for thousands of random weightings to see how often each method comes first.",
    "Rank-1 probability": "The share of SMAA weightings in which a method ranks first.",
    "In-sample": "The model was trained on the same people it is explained on; results look better than they would be.",
    "Pilot run": "A small, fast run to check the pipeline; not the final evidence.",
    "Fold model": "A model trained without the explained person's data (cross-validation), so explanations are out-of-sample.",
}


def page_glossary(run: str):
    st.title("Glossary")
    summary("Every technical word used in this dashboard, explained in one or two sentences.")
    q = st.text_input("Search", "")
    for k, v in GLOSSARY.items():
        if q.lower() in (k + v).lower():
            st.markdown(f"**{k}** — {v}")
