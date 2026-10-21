"""Dashboard pages 1-5: Overview, Model & data, Screening, Explanations, Metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BAND_PLAIN, CRITERIA_PLAIN, DIVERGING, HEAD_XY, METHOD_COLOUR, METHOD_PLAIN, MUTED, RANKED,
                    analysis, banner, explanation_values, fmt, howto, label, legend_row, meaning, method_chip,
                    no_results, pct, plotly_layout, primary_contexts, read_json, read_parquet, status_badge, summary,
                    table)


def current_ctx(run: str) -> dict | None:
    ctxs = primary_contexts(run)
    if not ctxs:
        return None
    ids = [c["context_id"] for c in ctxs]
    cid = st.session_state.get("ctx_id")
    return ctxs[ids.index(cid)] if cid in ids else ctxs[0]


def words_for_corr(r: float) -> str:
    if r != r:
        return "cannot be compared"
    a = abs(r)
    return "hardly" if a < 0.2 else "weakly" if a < 0.4 else "moderately" if a < 0.6 else "strongly"


# =========================================================================== 1 Overview
def page_overview(run: str):
    st.title("Which explanation method should we trust?")
    banner(run)
    ctxs = primary_contexts(run)
    rk = [(c, analysis(c, "ranking.json")) for c in ctxs]
    rk = [(c, r) for c, r in rk if r]
    if not rk:
        summary("This dashboard compares four ways of explaining the EEG workload model. No ranking has been "
                "computed for this run yet.")
        no_results("ranking")
        return
    texts = []
    for c, r in rk:
        p1 = r["p_rank1"]
        best = max(p1, key=p1.get)
        texts.append(f"For <b>{c['title']}</b>, <b>{label(best)}</b> was the most reliable method in "
                     f"<b>{pct(p1[best])}</b> of the possible weightings of the quality criteria.")
    summary(" ".join(texts) + " The cards below give the result; the pages on the left show how it was reached.")

    for c, r in rk:
        p1 = r["p_rank1"]
        order = sorted(p1, key=lambda m: -p1[m])
        eq = r.get("equivalence_top2", {})
        st.subheader(c["title"])
        cols = st.columns(4)
        cards = [
            ("Recommended explainer", method_chip(order[0]), "ranked first most often"),
            ("Chance of ranking first (SMAA)", pct(p1[order[0]]), f"runner-up {label(order[1])}: {pct(p1[order[1]])}"),
            ("Practically the same as runner-up? (ROPE)",
             "Yes" if eq.get("practically_equivalent") else "No",
             f"P(same) = {pct(eq.get('p_rope'))}" if eq else "not computed"),
            ("Evidence", f"{r['n_subjects']} subjects", f"{c['n_instances']} EEG windows · {len(c['seeds'])} repeats"),
        ]
        for col, (h, big, sub) in zip(cols, cards):
            col.markdown(f'<div class="card"><h4>{h}</h4><div class="big">{big}</div><div class="sub">{sub}</div></div>',
                         unsafe_allow_html=True)
    st.markdown("#### How the study works")
    steps = ["EEG window (18 s)", "Split into 14 channels × 6 frequency bands", "Four explanation methods",
             "Quality checks (fidelity, stability, …)", "Are the checks trustworthy?", "Statistics per subject",
             "Ranking under uncertainty (SMAA)", "Recommendation for Component 4"]
    st.markdown('<div class="pipe">' + '<span class="arrow">→</span>'.join(f'<div class="step">{s}</div>' for s in steps)
                + "</div>", unsafe_allow_html=True)
    with st.expander("What does this mean? — the four methods"):
        for m in RANKED:
            st.markdown(f"{method_chip(m)} — {METHOD_PLAIN[m]}", unsafe_allow_html=True)


# =========================================================================== 2 Model & data
def page_model(run: str):
    st.title("The model and the data")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    man = read_parquet(str(Path(ctx["_dir"]) / "manifest.parquet"))
    sel = read_parquet(str(Path(ctx["_dir"]) / "selected.parquet"))
    acc = float((man["pred"] == man["y"]).mean()) if man is not None else float("nan")
    src = "out-of-sample fold models" if ctx["model_source"] == "fold" else "the deployed model (trained on all subjects)"
    summary(f"The explained model is Component 1's <b>CNN-LSTM-Transformer</b>, which reads 18 seconds of 14-channel "
            f"EEG and predicts <b>low</b> (rest) or <b>high</b> (task) workload. Predictions here come from {src}; "
            f"it labels {pct(acc)} of all {len(man) if man is not None else '–'} windows correctly"
            + (" — on data it was trained on, so this is optimistic." if ctx["in_sample"] else "."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects explained", ctx["n_subjects"])
    c2.metric("EEG windows explained", ctx["n_instances"])
    c3.metric("Features per explanation", f"{len(ctx['channels'])} × {len(ctx['bands'])} = {len(ctx['feature_names'])}")
    if man is not None:
        fig = go.Figure()
        for cond, name, col in (("lo", "Rest recordings (true: low)", "#1F5FA8"), ("hi", "Task recordings (true: high)", "#C65D00")):
            fig.add_histogram(x=man.loc[man["condition"] == cond, "p"], name=name, marker_color=col, opacity=0.75,
                              xbins=dict(start=0, end=1, size=0.05),
                              hovertemplate="P(high) %{x}<br>%{y} windows<extra>" + name + "</extra>")
        fig.update_layout(barmode="overlay", xaxis_title="Model's probability of high workload",
                          yaxis_title="Number of windows")
        st.plotly_chart(plotly_layout(fig, 340), use_container_width=True)
        howto("Each bar counts EEG windows. Rest windows should sit on the left (low), task windows on the right "
              "(high). Overlap means the model is unsure or wrong for those windows.")
    with st.expander("How the EEG is turned into features"):
        st.write("Each 18-second window is split into **6 frequency bands** for each of the **14 channels**: "
                 + ", ".join(BAND_PLAIN[b] for b in ctx["bands"]) + ". Removing a feature means replacing that one "
                 "band of that one channel. The 'slow' band is not one of Component 1's bands: it was added because "
                 "about two thirds of the cleaned signal's power lies below 1 Hz and the model uses it.")
    with st.expander("Which windows were explained?"):
        if sel is not None:
            st.write(f"{len(sel)} windows from {sel['subject_id'].nunique()} subjects, chosen at random with a fixed "
                     "seed, balanced between rest and task. The choice never looked at the model's answer.")
            st.dataframe(sel[["instance_id", "subject_id", "condition", "t_start", "t_end", "p", "frac_bad"]]
                         .rename(columns={"instance_id": "window", "subject_id": "subject", "p": "P(high)",
                                          "frac_bad": "share of noisy epochs"}).round(3), hide_index=True)


# =========================================================================== 3 Screening
def page_screening(run: str):
    from xaicomp.screening.applicability import CHECK_COLUMNS, CHECK_PLAIN

    st.title("Which methods can be used? (Screening)")
    banner(run)
    from common import RESULTS
    sc = read_parquet(str(RESULTS / run / "screening.parquet"))
    if sc is None:
        no_results("screening table")
        return
    main = sc[~sc["screening_only"]].drop_duplicates("method")
    ok = main[main["verdict"] == "applicable"]["label"].tolist()
    bad = main[main["verdict"] == "excluded"]["label"].tolist()
    summary(f"Before comparing, every candidate method is checked against 8 requirements. For the EEG deep model, "
            f"<b>{len(ok)}</b> methods can be used ({', '.join(ok)}); <b>{', '.join(bad) or 'none'}</b> "
            "cannot, for the reasons listed below.")
    for cid, g in sc.groupby("context_id", sort=False):
        st.subheader(cid + ("  (screening only)" if g["screening_only"].iloc[0] else ""))
        rows = []
        for _, r in g.iterrows():
            cells = "".join(f"<td style='text-align:center'>{status_badge(r[c])}</td>" for c in CHECK_COLUMNS)
            verdict = status_badge("pass" if r["verdict"] == "applicable" else "fail").replace("OK", "usable")
            rows.append(f"<tr><td><b>{r['label']}</b></td>{cells}<td>{verdict}</td>"
                        f"<td style='color:{MUTED}'>{r['reasons']}</td></tr>")
        head = "".join(f"<th title='{CHECK_PLAIN[c]}' style='font-size:0.8em'>{i + 1}</th>" for i, c in enumerate(CHECK_COLUMNS))
        st.markdown(f"<table style='width:100%'><tr><th>Method</th>{head}<th>Verdict</th><th>Why</th></tr>"
                    + "".join(rows) + "</table>", unsafe_allow_html=True)
    howto("Each column is one check (numbered as in the list below). ✓ OK = satisfied, ⚠ caution = usable with a "
          "caveat, ✕ = a requirement fails, so the method is excluded.")
    with st.expander("What does this mean? — the 8 checks"):
        for i, c in enumerate(CHECK_COLUMNS):
            st.write(f"**{i + 1}. {CHECK_PLAIN[c]}**")
        ms = sc["ms_per_eval_measured"].dropna()
        if len(ms):
            st.write(f"Cost estimates use the measured {fmt(ms.iloc[0])} ms per model evaluation.")


# =========================================================================== 4 Explanations
def head_figure(values_by_channel: dict[str, float], title: str, vmax: float):
    t = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#9AA5B1", width=2), hoverinfo="skip")
    fig.add_scatter(x=[-0.1, 0, 0.1], y=[0.99, 1.12, 0.99], mode="lines", line=dict(color="#9AA5B1", width=2), hoverinfo="skip")
    for s in (-1, 1):
        fig.add_scatter(x=[s * 1.0, s * 1.07, s * 1.0], y=[0.15, 0, -0.15], mode="lines",
                        line=dict(color="#9AA5B1", width=2), hoverinfo="skip")
    chs = list(HEAD_XY)
    v = np.array([values_by_channel.get(c, 0.0) for c in chs])
    fig.add_scatter(x=[HEAD_XY[c][0] for c in chs], y=[HEAD_XY[c][1] for c in chs], mode="markers+text",
                    text=chs, textposition="bottom center", textfont=dict(size=11, color="#3E4C59"),
                    marker=dict(size=14 + 22 * np.abs(v) / (vmax or 1), color=v, colorscale=DIVERGING, cmin=-vmax,
                                cmax=vmax, line=dict(color="#FFFFFF", width=2)),
                    customdata=[("pushes towards HIGH workload" if x > 0 else "pushes towards LOW workload" if x < 0
                                 else "no effect") for x in v],
                    hovertemplate="<b>%{text}</b><br>relative importance %{marker.color:.2f}<br>%{customdata}<extra></extra>")
    fig.update_layout(showlegend=False, title=dict(text=title, font=dict(size=14)))
    fig.update_xaxes(visible=False, range=[-1.25, 1.25])
    fig.update_yaxes(visible=False, range=[-1.15, 1.25], scaleanchor="x")
    return plotly_layout(fig, 300)


def page_explanations(run: str):
    from xaicomp.representation.groups import InterpretableSpace

    st.title("Explanations for one EEG window")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    sel = read_parquet(str(Path(ctx["_dir"]) / "selected.parquet"))
    inst = table(ctx, "instances")
    if sel is None or inst is None:
        no_results("explanations")
        return
    done = set(inst["instance_id"])
    sel = sel[sel["instance_id"].isin(done)]
    space = InterpretableSpace(ctx["channels"], ctx["bands"], ctx["regions"])
    c1, c2 = st.columns(2)
    subj = c1.selectbox("Subject", sorted(sel["subject_id"].unique()), format_func=lambda s: f"Subject {s:02d}")
    g = sel[sel["subject_id"] == subj]
    iid = c2.selectbox("Window", g["instance_id"].tolist(),
                       format_func=lambda i: (f"{'Task' if g.set_index('instance_id').loc[i, 'condition'] == 'hi' else 'Rest'}"
                                              f" recording, {g.set_index('instance_id').loc[i, 't_start']:.0f}–"
                                              f"{g.set_index('instance_id').loc[i, 't_end']:.0f} s"))
    row = inst.set_index("instance_id").loc[iid]
    seed = ctx["seeds"][0]
    vals = {m: explanation_values(ctx["_dir"], iid, m, seed) for m in RANKED}
    vals = {m: v for m, v in vals.items() if v is not None}
    p = float(row["p"])
    pred = "high" if p >= 0.5 else "low"
    truth = "high" if int(row["y"]) == 1 else "low"

    # words: where do methods agree?
    tops = {}
    for m, v in vals.items():
        rv, rn = space.aggregate(v, "region")
        bv, bn = space.aggregate(v, "band")
        tops[m] = (rn[int(np.argmax(np.abs(rv)))], bn[int(np.argmax(np.abs(bv)))])
    regions = pd.Series([t[0] for t in tops.values()]).value_counts()
    bands = pd.Series([t[1] for t in tops.values()]).value_counts()
    ag = table(ctx, "agreement")
    mean_r = float(ag[(ag["instance_id"] == iid) & ag["method_a"].isin(RANKED) & ag["method_b"].isin(RANKED)
                      & (ag["seed"] == seed)]["rank_corr"].mean()) if ag is not None else float("nan")
    summary(f"The model says <b>{pred} workload</b> (probability of high = {pct(p)}; confidence {pct(max(p, 1 - p))}); "
            f"the true label is <b>{truth}</b>. {regions.iloc[0]} of {len(vals)} methods point to the "
            f"<b>{regions.index[0]}</b> region as most important and {bands.iloc[0]} of {len(vals)} to the "
            f"<b>{BAND_PLAIN.get(bands.index[0], bands.index[0])}</b> band. Overall the methods' rankings agree "
            f"<b>{words_for_corr(mean_r)}</b> (average rank correlation {fmt(mean_r)}).")

    legend_row(list(vals))
    st.markdown("#### Where on the head? (each method scaled to its own largest value)")
    cols = st.columns(len(vals))
    for col, (m, v) in zip(cols, vals.items()):
        ch, names = space.aggregate(v, "channel")
        vmax = float(np.max(np.abs(ch))) or 1.0
        col.plotly_chart(head_figure(dict(zip(names, ch / vmax)), label(m), 1.0), use_container_width=True)
    howto("Each dot is one EEG electrode seen from above (nose at the top). Orange = this part of the signal pushes "
          "the prediction towards HIGH workload, blue = towards LOW, grey = little effect. Bigger dots matter more.")

    st.markdown("#### Channel × frequency band")
    cols = st.columns(len(vals))
    for col, (m, v) in zip(cols, vals.items()):
        M = space.as_matrix(v)
        vmax = float(np.max(np.abs(M))) or 1.0
        fig = go.Figure(go.Heatmap(z=M / vmax, x=ctx["bands"], y=ctx["channels"], colorscale=DIVERGING, zmin=-1, zmax=1,
                                   xgap=2, ygap=2, showscale=False, customdata=M,
                                   hovertemplate="%{y}, %{x}<br>attribution %{customdata:.3g}<extra>" + label(m) + "</extra>"))
        fig.update_layout(title=dict(text=label(m), font=dict(size=14)))
        fig.update_yaxes(autorange="reversed")
        col.plotly_chart(plotly_layout(fig, 430), use_container_width=True)
    howto("Rows are electrodes, columns are frequency bands. Same colours as above; hover to see the raw value.")

    st.markdown("#### By brain region")
    fig = go.Figure()
    for m, v in vals.items():
        rv, rn = space.aggregate(v, "region")
        rv = rv / (np.max(np.abs(rv)) or 1)
        fig.add_bar(x=rn, y=rv, name=label(m), marker_color=METHOD_COLOUR[m], text=[f"{x:+.2f}" for x in rv],
                    textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra>" + label(m) + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      yaxis_title="Relative effect (+ towards high, − towards low)")
    st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
    howto("For each region the bars show each method's total effect, scaled per method. Bars on the same side of "
          "zero mean the methods agree on the direction.")
    with st.expander("Details for this window"):
        rows = [{"method": label(m), "most important region": t[0], "most important band": t[1]} for m, t in tops.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if ag is not None:
            a = ag[(ag["instance_id"] == iid) & (ag["seed"] == seed)]
            st.dataframe(a.assign(pair=a["method_a"].map(label) + " vs " + a["method_b"].map(label))[
                ["pair", "rank_corr", "feature_agree", "sign_agree"]].rename(columns={
                    "rank_corr": "rank agreement", "feature_agree": "share of top-10 in common",
                    "sign_agree": "same direction (top-10)"}).round(2), hide_index=True)


# =========================================================================== 5 Metrics
@st.cache_data(show_spinner=False)
def mean_curves(ctx_dir: str, mode: str, seed: int) -> dict:
    out: dict[str, list] = {}
    rand = []
    for p in Path(ctx_dir, "metrics").glob("*.json"):
        r = json.loads(p.read_text())
        c = r["curves"].get(mode, {})
        for k, v in c.items():
            if k.endswith(f"|s{seed}"):
                out.setdefault(k.split("|")[0], []).append(v["deletion"])
        if "random_deletion" in c:
            rand.append(c["random_deletion"])
    res = {m: np.mean(v, axis=0).tolist() for m, v in out.items()}
    if rand:
        res["__random"] = np.mean(rand, axis=0).tolist()
    return res


def ci_dot_chart(bm: pd.DataFrame, title: str):
    bm = bm.copy()
    bm["lab"] = bm["method"].map(label)
    fig = go.Figure()
    for _, r in bm.iterrows():
        fig.add_scatter(x=[r["mean"]], y=[r["lab"]], mode="markers+text", text=[fmt(r["mean"])], textposition="top center",
                        marker=dict(size=13, color=METHOD_COLOUR.get(r["method"], "#999"), line=dict(color="white", width=2)),
                        error_x=dict(type="data", symmetric=False, array=[r["ci_high"] - r["mean"]],
                                     arrayminus=[r["mean"] - r["ci_low"]], color="#7B8794", thickness=2, width=6),
                        name=r["lab"], showlegend=False,
                        hovertemplate=f"<b>{r['lab']}</b><br>mean {fmt(r['mean'], 3)}<br>95% CI {fmt(r['ci_low'], 3)} to "
                                      f"{fmt(r['ci_high'], 3)}<extra></extra>")
    fig.update_layout(xaxis_title=title)
    return plotly_layout(fig, 90 + 60 * len(bm))


def page_metrics(run: str):
    st.title("How good are the explanations? (Metrics)")
    ctx = current_ctx(run)
    if not ctx:
        no_results()
        return
    banner(run, ctx)
    stats = analysis(ctx, "statistics.json")
    if not stats:
        no_results("metrics")
        return
    fo = pd.DataFrame(stats.get("fidelity_on", {}).get("bootstrap_means", []))
    best = fo.loc[fo["mean"].idxmax()] if len(fo) else None
    summary("Each method is scored on six quality questions, averaged per subject first. "
            + (f"On the most important one — does the explanation match what the model really does? — "
               f"<b>{label(best['method'])}</b> scores highest ({fmt(best['mean'])} above random order)." if best is not None else ""))
    legend_row(RANKED)
    for crit, (title, direction, text) in CRITERIA_PLAIN.items():
        if crit not in stats:
            continue
        st.markdown(f"#### {title}")
        bm = pd.DataFrame(stats[crit]["bootstrap_means"])
        st.plotly_chart(ci_dot_chart(bm, ("higher is better →" if direction == "+" else "← lower is better")),
                        use_container_width=True)
        howto("Dot = average over subjects; line = 95% range of plausible averages (bootstrap over subjects). "
              "Overlapping lines mean the difference may be chance.")
        meaning(title, text, direction)
        if crit == "stability":
            st.caption("Integrated Gradients with a zero reference is not shown: it is deterministic, so its "
                       "stability is always 1 and is not rewarded.")

    st.markdown("#### Deletion curves: how fast does the prediction fall when important parts are removed?")
    mode = st.radio("Removal type", ["on_manifold", "off_manifold"], horizontal=True,
                    format_func=lambda m: "realistic (other people's EEG)" if m == "on_manifold" else "zero (unrealistic)")
    cur = mean_curves(ctx["_dir"], mode, ctx["seeds"][0])
    if cur:
        fig = go.Figure()
        symbols = {"kernel_shap": "circle", "conditional_shap": "diamond", "lime": "square", "ig": "triangle-up", "ig_zero": "x"}
        for m, y in cur.items():
            x = np.linspace(0, 100, len(y))
            if m == "__random":
                fig.add_scatter(x=x, y=y, name="Random order", line=dict(color="#9AA5B1", width=2, dash="dash"),
                                hovertemplate="%{x:.0f}% removed: %{y:.2f}<extra>random order</extra>")
            else:
                fig.add_scatter(x=x, y=y, name=label(m), mode="lines+markers",
                                line=dict(color=METHOD_COLOUR.get(m, "#999"), width=2),
                                marker=dict(symbol=symbols.get(m, "circle"), size=8),
                                hovertemplate="%{x:.0f}% removed: %{y:.2f}<extra>" + label(m) + "</extra>")
        fig.update_layout(xaxis_title="Share of features removed (most important first, %)",
                          yaxis_title="Probability of the predicted class")
        st.plotly_chart(plotly_layout(fig, 400), use_container_width=True)
        howto("Lines that drop faster than the grey dashed 'random order' line mean the method found parts the model "
              "really relies on. Averaged over all explained windows.")
