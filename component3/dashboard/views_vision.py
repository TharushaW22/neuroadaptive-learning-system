"""Dashboard pages for the vision context, the multimodal explanation, the cross-context comparison and
the project-level Overview / Component 4 pages.

Vision pages reuse the EEG page functions wherever the content is the same (screening, metrics,
reliability, disagreement, statistics, ranking; app.py sets the modality so their wording fits).
Only pages that show the input itself differ: the explanation view draws a SCHEMATIC face (DAiSEE's
licence does not allow showing real faces) with the 5 feature groups coloured by attribution, a
time-segment chart and a bar chart. Until vision results exist every vision page shows the same
friendly empty state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from common import (DIVERGING, METHOD_COLOUR, METHOD_PLAIN_VISION, RANKED, RESULTS, VISION_WAITING, analysis, banner,
                    explanation_values, fmt, howto, label, legend_row, method_chip, no_results, pct, plotly_layout,
                    primary_contexts, read_parquet, summary, table)
from views_a import current_ctx, words_for_corr

GROUP_PLAIN = {"eyes_blink": "Eyes & blinks", "gaze": "Gaze", "head_pose": "Head pose & position",
               "mouth": "Mouth", "brows": "Brows"}
FEATURE_PLAIN = {"ear_left": "left eye openness", "ear_right": "right eye openness", "ear_mean": "eye openness (mean)",
                 "blink_event": "blink", "blink_rate_per_min": "blink rate", "gaze_h": "gaze left/right",
                 "gaze_v": "gaze up/down", "head_yaw": "head turn (yaw)", "head_pitch": "head nod (pitch)",
                 "head_roll": "head tilt (roll)", "mouth_aspect_ratio": "mouth openness", "brow_raise": "brow raise",
                 "face_scale": "distance to camera", "face_center_x": "face position left/right",
                 "face_center_y": "face position up/down"}
MULTIMODAL_TEXT = {
    "why": "A multimodal model would combine a learner's EEG and their webcam video to predict one cognitive state. "
           "Explaining it would show, for example, how much the brain signal and the face each contributed. "
           "No such model is built or explained in this project, because there is no genuinely paired data: STEW "
           "(EEG) and DAiSEE (video) were recorded from different people, at different times, doing different "
           "tasks. Joining them would pair one person's EEG with a stranger's face, and any explanation of that "
           "model — such as 'EEG contributed 60 %' — would describe an artificial pairing, not a real learner.",
    "needed": ["Simultaneous EEG and webcam recordings of the SAME learners, with a shared clock so 4-second "
               "windows line up.",
               "One label per window that applies to both signals (for example workload or engagement rated for "
               "that moment).",
               "Subject IDs and a subject-disjoint train/test split, as for the single-modality contexts.",
               "A fusion model trained on those pairs (early, late or intermediate fusion); Component 3's adapter "
               "interface can then explain it like the other contexts, with EEG and video features as groups."],
    "instead": "Component 3 explains the two single-modality models separately and compares how suitable each "
               "explanation method is in each prediction context (see 'Cross-context comparison').",
}


def vision_empty(show_check: bool = True) -> None:
    st.info(VISION_WAITING)
    if show_check and st.button("Run the readiness check now (read-only)"):
        from xaicomp.config import load_context
        from xaicomp.run.vision_readiness import check_vision_ready, format_report
        with st.spinner("Checking Component 2's files ..."):
            st.code(format_report(check_vision_ready(load_context("vision_engagement_gru"), compute_accuracy=False)))


def _space(ctx: dict):
    from xaicomp.representation.vision_space import space_from_info
    return space_from_info(ctx)


def _colour(v: float) -> str:
    return sample_colorscale(DIVERGING, [float(np.clip((v + 1) / 2, 0, 1))])[0]


# =========================================================================== schematic face
def face_figure(group_values: dict[str, float], title: str):
    """Schematic face (no real faces: DAiSEE licence). Values in [-1, 1]: orange = towards HIGH engagement,
    blue = towards LOW, grey = little effect. Head pose & position colours the head outline."""
    fig = go.Figure()
    shapes = []
    hp = group_values.get("head_pose", 0.0)
    shapes.append(dict(type="circle", x0=-1, y0=-1.3, x1=1, y1=1.3, fillcolor="#FAFAFA",
                       line=dict(color=_colour(hp), width=10)))
    ey = group_values.get("eyes_blink", 0.0)
    gz = group_values.get("gaze", 0.0)
    for sx in (-1, 1):
        shapes.append(dict(type="circle", x0=sx * 0.4 - 0.24, y0=0.15, x1=sx * 0.4 + 0.24, y1=0.39,
                           fillcolor=_colour(ey), line=dict(color="#7B8794", width=1.5)))
        shapes.append(dict(type="circle", x0=sx * 0.4 - 0.08, y0=0.19, x1=sx * 0.4 + 0.08, y1=0.35,
                           fillcolor=_colour(gz), line=dict(color="#3E4C59", width=1.5)))
        shapes.append(dict(type="line", x0=sx * 0.18, y0=0.55, x1=sx * 0.64, y1=0.62,
                           line=dict(color=_colour(group_values.get("brows", 0.0)), width=12)))
    shapes.append(dict(type="circle", x0=-0.34, y0=-0.72, x1=0.34, y1=-0.5,
                       fillcolor=_colour(group_values.get("mouth", 0.0)), line=dict(color="#7B8794", width=1.5)))
    shapes.append(dict(type="line", x0=0, y0=0.05, x1=-0.08, y1=-0.25, line=dict(color="#9AA5B1", width=2)))
    anchors = {"head_pose": (0.98, -0.95), "eyes_blink": (-0.4, 0.27), "gaze": (0.4, 0.27), "brows": (0.41, 0.6),
               "mouth": (0, -0.61)}
    names = [g for g in anchors if g in group_values]
    fig.add_scatter(x=[anchors[g][0] for g in names], y=[anchors[g][1] for g in names], mode="markers",
                    marker=dict(size=26, color="rgba(0,0,0,0)"), showlegend=False,
                    customdata=[[GROUP_PLAIN.get(g, g), group_values[g],
                                 "towards HIGH engagement" if group_values[g] > 0 else
                                 "towards LOW engagement" if group_values[g] < 0 else "no effect"] for g in names],
                    hovertemplate="<b>%{customdata[0]}</b><br>relative effect %{customdata[1]:+.2f}<br>"
                                  "%{customdata[2]}<extra></extra>")
    fig.update_layout(shapes=shapes, title=dict(text=title, font=dict(size=14)), showlegend=False)
    fig.update_xaxes(visible=False, range=[-1.3, 1.3])
    fig.update_yaxes(visible=False, range=[-1.45, 1.45], scaleanchor="x")
    return plotly_layout(fig, 320)


# =========================================================================== vision summary
def page_vision_overview(run: str | None):
    st.title("Vision engagement: which explanation method should we trust?")
    if run is None:
        vision_empty()
        return
    banner(run)
    ctxs = primary_contexts(run)
    rk = [(c, analysis(c, "ranking.json")) for c in ctxs]
    rk = [(c, r) for c, r in rk if r]
    if not rk:
        summary("This section compares four ways of explaining Component 2's vision engagement model. No ranking "
                "has been computed for this run yet.")
        no_results("ranking")
        return
    c, r = rk[0]
    p1 = r["p_rank1"]
    order = sorted(p1, key=lambda m: -p1[m])
    eq = r.get("equivalence_top2", {})
    summary(f"For <b>{c['title']}</b>, <b>{label(order[0])}</b> was the most reliable method in "
            f"<b>{pct(p1[order[0]])}</b> of the possible weightings of the quality criteria. This is a statement "
            "about explanation methods for this model, not about vision versus EEG.")
    cols = st.columns(4)
    cards = [("Recommended explainer", method_chip(order[0]), "ranked first most often"),
             ("Chance of ranking first (SMAA)", pct(p1[order[0]]), f"runner-up {label(order[1])}: {pct(p1[order[1]])}"),
             ("Practically the same as runner-up? (ROPE)", "Yes" if eq.get("practically_equivalent") else "No",
              f"P(same) = {pct(eq.get('p_rope'))}" if eq else "not computed"),
             ("Evidence", f"{r['n_subjects']} students", f"{c['n_instances']} video windows · {len(c['seeds'])} repeats")]
    for col, (h, big, sub) in zip(cols, cards):
        col.markdown(f'<div class="card"><h4>{h}</h4><div class="big">{big}</div><div class="sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("#### How the vision study works")
    steps = ["Video window (4 s, 20 frames)", "15 facial measurements per frame", "Cut into 4 time segments",
             "Four explanation methods", "Quality checks", "Are the checks trustworthy?", "Statistics per student",
             "Ranking under uncertainty (SMAA)", "Recommendation for Component 4"]
    st.markdown('<div class="pipe">' + '<span class="arrow">→</span>'.join(f'<div class="step">{s}</div>' for s in steps)
                + "</div>", unsafe_allow_html=True)
    with st.expander("What does this mean? — the four methods"):
        for m in RANKED:
            st.markdown(f"{method_chip(m)} — {METHOD_PLAIN_VISION[m]}", unsafe_allow_html=True)


# =========================================================================== vision model & data
def page_vision_model(run: str | None):
    st.title("The vision model and the data")
    ctx = current_ctx(run) if run else None
    if not ctx:
        vision_empty()
        return
    banner(run, ctx)
    man = read_parquet(str(Path(ctx["_dir"]) / "manifest.parquet"))
    sel = read_parquet(str(Path(ctx["_dir"]) / "selected.parquet"))
    acc = float((man["pred"] == man["y"]).mean()) if man is not None else float("nan")
    sc = ctx.get("split_check", {})
    summary(f"The explained model is Component 2's <b>GRU</b>, which reads 4 seconds of webcam features (15 facial "
            f"measurements per frame, 5 frames per second) and predicts <b>low</b> (DAiSEE level 0–1) or <b>high</b> "
            f"(level 2–3) engagement. Only windows of Component 2's <b>test</b> split are explained; the model labels "
            f"{pct(acc)} of all {len(man) if man is not None else '–'} test windows correctly. "
            + ("The split is verified subject-disjoint, so these students were never seen in training."
               if sc.get("subject_disjoint") else "<b>The split is not verified subject-disjoint, so results are IN-SAMPLE.</b>"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Students explained", ctx["n_subjects"])
    c2.metric("Video windows explained", ctx["n_instances"])
    c3.metric("Features per explanation", f"{len(ctx['features'])} × {len(ctx['segments'])} = {len(ctx['feature_names'])}")
    if man is not None:
        fig = go.Figure()
        for cond, name, col in (("low", "True: low engagement", "#1F5FA8"), ("high", "True: high engagement", "#C65D00")):
            fig.add_histogram(x=man.loc[man["condition"] == cond, "p"], name=name, marker_color=col, opacity=0.75,
                              xbins=dict(start=0, end=1, size=0.05),
                              hovertemplate="P(high) %{x}<br>%{y} windows<extra>" + name + "</extra>")
        fig.update_layout(barmode="overlay", xaxis_title="Model's probability of high engagement",
                          yaxis_title="Number of windows")
        st.plotly_chart(plotly_layout(fig, 340), use_container_width=True)
        howto("Each bar counts video windows. Windows rated low should sit on the left, high on the right. Overlap "
              "means the model is unsure or wrong for those windows.")
    with st.expander("How the video is turned into features"):
        st.write("Component 2 measures 15 things in every frame from face landmarks (no pixels reach the model): "
                 + ", ".join(FEATURE_PLAIN.values()) + ". Each 4-second window is cut into 4 one-second segments, so one "
                 "explained feature is one measurement during one second. Removing a feature replaces those 5 frames. "
                 "Groups: " + "; ".join(f"{GROUP_PLAIN.get(g, g)} = {', '.join(FEATURE_PLAIN.get(f, f) for f in fs)}"
                                        for g, fs in ctx["groups"].items()) + ".")
    cc = ctx.get("conditional_check", {})
    with st.expander("Model, split and Conditional SHAP checks"):
        m = ctx.get("c2_model", {})
        st.write(f"Checkpoint: `{m.get('name')}` (created {m.get('created_utc')}, label scheme {m.get('label_scheme')}, "
                 f"cell {m.get('cell')}). Participants per split: {sc.get('participants_per_split')}.")
        st.write(f"Conditional SHAP used the **{cc.get('mode', '–')}** conditional: {cc.get('reason', '')}"
                 + (f" (held-out coverage {fmt(cc.get('coverage'))} for a nominal {fmt(cc.get('nominal'))}; "
                    f"discrete share {fmt(cc.get('discrete_share'))})." if "coverage" in cc else "."))
    with st.expander("Which windows were explained?"):
        if sel is not None:
            st.write(f"{len(sel)} windows from {sel['subject_id'].nunique()} students, chosen at random with a fixed "
                     "seed, balanced between low and high where a student has both. The choice never looked at the "
                     "model's answer.")
            st.dataframe(sel[["instance_id", "participant_id", "condition", "t_start", "t_end", "p", "frac_bad"]]
                         .rename(columns={"instance_id": "window", "participant_id": "student", "condition": "true label",
                                          "p": "P(high)", "frac_bad": "share of low-quality frames"}).round(3),
                         hide_index=True)


# =========================================================================== vision explanations
def page_vision_explanations(run: str | None):
    st.title("Explanations for one video window")
    ctx = current_ctx(run) if run else None
    if not ctx:
        vision_empty()
        return
    banner(run, ctx)
    sel = read_parquet(str(Path(ctx["_dir"]) / "selected.parquet"))
    inst = table(ctx, "instances")
    if sel is None or inst is None:
        no_results("explanations")
        return
    sel = sel[sel["instance_id"].isin(set(inst["instance_id"]))]
    space = _space(ctx)
    c1, c2 = st.columns(2)
    subj = c1.selectbox("Student", sorted(sel["participant_id"].unique()), format_func=lambda s: f"Student {s}")
    g = sel[sel["participant_id"] == subj].set_index("instance_id")
    iid = c2.selectbox("Window", g.index.tolist(),
                       format_func=lambda i: f"clip {g.loc[i, 'video_id']}, {g.loc[i, 't_start']:.0f}–{g.loc[i, 't_end']:.0f} s")
    row = inst.set_index("instance_id").loc[iid]
    seed = ctx["seeds"][0]
    vals = {m: explanation_values(ctx["_dir"], iid, m, seed) for m in RANKED}
    vals = {m: v for m, v in vals.items() if v is not None}
    if not vals:
        no_results("explanations")
        return
    p = float(row["p"])
    pred, truth = ("high" if p >= 0.5 else "low"), ("high" if int(row["y"]) == 1 else "low")
    tops = {}
    for m, v in vals.items():
        gv, gn = space.aggregate(v, "group")
        sv, sn = space.aggregate(v, "segment")
        tops[m] = (gn[int(np.argmax(np.abs(gv)))], sn[int(np.argmax(np.abs(sv)))])
    groups = pd.Series([t[0] for t in tops.values()]).value_counts()
    ag = table(ctx, "agreement")
    mean_r = float(ag[(ag["instance_id"] == iid) & ag["method_a"].isin(RANKED) & ag["method_b"].isin(RANKED)
                      & (ag["seed"] == seed)]["rank_corr"].mean()) if ag is not None else float("nan")
    summary(f"The model says <b>{pred} engagement</b> (probability of high = {pct(p)}; confidence {pct(max(p, 1 - p))}); "
            f"the observers' rating is <b>{truth}</b>. {groups.iloc[0]} of {len(vals)} methods point to "
            f"<b>{GROUP_PLAIN.get(groups.index[0], groups.index[0]).lower()}</b> as most important. Overall the methods' "
            f"rankings agree <b>{words_for_corr(mean_r)}</b> (average rank correlation {fmt(mean_r)}).")
    legend_row(list(vals))
    st.markdown("#### Which part of the face? (each method scaled to its own largest value)")
    cols = st.columns(len(vals))
    for col, (m, v) in zip(cols, vals.items()):
        gv, gn = space.aggregate(v, "group")
        vmax = float(np.max(np.abs(gv))) or 1.0
        col.plotly_chart(face_figure(dict(zip(gn, gv / vmax)), label(m)), use_container_width=True)
    howto("A drawn face, not the student (the dataset's licence forbids showing faces). Orange = these features push the "
          "prediction towards HIGH engagement, blue = towards LOW, grey = little effect. Eyes = eye openness and blinks, "
          "irises = gaze, bars = brows, mouth = mouth openness, outline = head pose & position. Hover for values.")

    st.markdown("#### When in the 4 seconds?")
    fig = go.Figure()
    for m, v in vals.items():
        sv, sn = space.aggregate(v, "segment")
        sv = sv / (np.max(np.abs(sv)) or 1)
        fig.add_bar(x=sn, y=sv, name=label(m), marker_color=METHOD_COLOUR[m], text=[f"{x:+.2f}" for x in sv],
                    textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra>" + label(m) + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, xaxis_title="Time segment of the window",
                      yaxis_title="Relative effect (+ towards high, − towards low)")
    st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)
    howto("For each one-second segment the bars show each method's total effect over all 15 measurements, scaled per method.")

    st.markdown("#### By face region")
    fig = go.Figure()
    for m, v in vals.items():
        gv, gn = space.aggregate(v, "group")
        gv = gv / (np.max(np.abs(gv)) or 1)
        fig.add_bar(x=[GROUP_PLAIN.get(n, n) for n in gn], y=gv, name=label(m), marker_color=METHOD_COLOUR[m],
                    text=[f"{x:+.2f}" for x in gv], textposition="outside",
                    hovertemplate="%{x}: %{y:.2f}<extra>" + label(m) + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      yaxis_title="Relative effect (+ towards high, − towards low)")
    st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
    howto("Bars on the same side of zero mean the methods agree on the direction for that face region.")
    with st.expander("Details: measurement × time segment"):
        cols = st.columns(len(vals))
        for col, (m, v) in zip(cols, vals.items()):
            M = space.as_matrix(v)
            vmax = float(np.max(np.abs(M))) or 1.0
            fig = go.Figure(go.Heatmap(z=M / vmax, x=space.segments, y=[FEATURE_PLAIN.get(f, f) for f in space.features],
                                       colorscale=DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2, showscale=False,
                                       customdata=M, hovertemplate="%{y}, %{x}<br>attribution %{customdata:.3g}<extra>"
                                       + label(m) + "</extra>"))
            fig.update_layout(title=dict(text=label(m), font=dict(size=14)))
            fig.update_yaxes(autorange="reversed")
            col.plotly_chart(plotly_layout(fig, 460), use_container_width=True)
        st.dataframe(pd.DataFrame([{"method": label(m), "most important region": GROUP_PLAIN.get(t[0], t[0]),
                                    "most important second": t[1]} for m, t in tops.items()]), hide_index=True)


# =========================================================================== multimodal
def page_multimodal(*_):
    st.title("Multimodal (EEG + video)")
    summary("No multimodal model is built or explained, because the project has no EEG and video recorded from the "
            "same learners at the same time.")
    st.markdown("#### Why not?")
    st.write(MULTIMODAL_TEXT["why"])
    st.markdown("#### What would be needed")
    for item in MULTIMODAL_TEXT["needed"]:
        st.markdown(f"- {item}")
    st.markdown("#### What Component 3 does instead")
    st.write(MULTIMODAL_TEXT["instead"])


# =========================================================================== cross-context
def page_cross_context(eeg_run: str | None, vision_run: str | None):
    from xaicomp.compare.cross_context import CAVEATS, FRAMING, compare_contexts, find_context_dir

    st.title("Cross-context comparison")
    eeg_dir = find_context_dir(RESULTS / eeg_run, "eeg") if eeg_run else None
    vis_dir = find_context_dir(RESULTS / vision_run, "vision") if vision_run else None
    doc = compare_contexts(eeg_dir, vis_dir)
    summary(FRAMING)
    if not doc["available"]:
        if vis_dir is None:
            st.info(VISION_WAITING)
        st.write("This page appears when both prediction contexts have results. It will show, side by side: each "
                 "method's chance of ranking first, which quality criteria were reliable enough to use in each context, "
                 "the criterion averages, and whether the recommended method is the same.")
        st.caption("Still missing: " + "; ".join(doc["missing"]))
        with st.expander("Why the comparison is limited"):
            for c in CAVEATS:
                st.markdown(f"- {c}")
        return
    a, b = doc["eeg"], doc["vision"]
    st.markdown(f"**EEG** recommends {method_chip(a['recommended'])} · **Vision** recommends "
                f"{method_chip(b['recommended'])} · same recommendation: **{'yes' if doc['same_recommendation'] else 'no'}**",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, s, name in ((c1, a, "EEG workload"), (c2, b, "Vision engagement")):
        order = sorted(s["p_rank1"], key=lambda m: -s["p_rank1"][m])
        fig = go.Figure(go.Bar(x=[s["p_rank1"][m] for m in order], y=[label(m) for m in order], orientation="h",
                               marker=dict(color=[METHOD_COLOUR.get(m) for m in order]),
                               text=[pct(s["p_rank1"][m]) for m in order], textposition="outside",
                               hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_layout(title=dict(text=f"{name} — {s['run_type']}", font=dict(size=14)),
                          xaxis=dict(range=[0, 1.15], tickformat=".0%", title="Chance of ranking first"),
                          yaxis=dict(autorange="reversed"))
        col.plotly_chart(plotly_layout(fig, 300), use_container_width=True)
    howto("Each chart is one prediction context. Compare which method comes first in each, not the bar lengths "
          "across charts: the contexts differ in data, people and labels.")
    tau = doc["ranking_agreement"]["kendall_tau_expected_rank"]
    st.write(f"Agreement of the two method rankings (Kendall's tau on expected rank, {doc['ranking_agreement']['n_methods']} "
             f"methods): {fmt(tau)} — descriptive only.")
    st.markdown("#### Which quality criteria were reliable enough to use?")
    st.dataframe(pd.DataFrame(doc["criteria_reliability"]).rename(columns={"eeg": "EEG", "vision": "Vision"}),
                 hide_index=True, use_container_width=True)
    st.markdown("#### Criterion averages per method")
    st.dataframe(pd.DataFrame(doc["methods"]).assign(method=lambda d: d["method"].map(label)).round(3),
                 hide_index=True, use_container_width=True)
    with st.expander("Caveats"):
        for c in doc["caveats"]:
            st.markdown(f"- {c}")


# =========================================================================== project overview
def page_project_overview(eeg_run: str | None, vision_run: str | None):
    st.title("Comparing explanation methods for cognitive-state predictions")
    summary("Component 3 checks which explanation (XAI) method — KernelSHAP, Conditional SHAP, LIME or Integrated "
            "Gradients — can be trusted for each prediction context, using quality checks that are themselves tested "
            "for reliability, statistics per person and a ranking that does not depend on one choice of weights. "
            "The answer goes to Component 4.")
    c1, c2, c3 = st.columns(3)
    for col, run, mod, title in ((c1, eeg_run, "eeg", "EEG workload (Component 1)"),
                                 (c2, vision_run, "vision", "Vision engagement (Component 2)")):
        rk = None
        if run:
            ctx = next(iter(primary_contexts(run)), None)
            rk = analysis(ctx, "ranking.json") if ctx else None
        if rk:
            p1 = rk["p_rank1"]
            best = max(p1, key=p1.get)
            big, sub = method_chip(best), f"ranked first in {pct(p1[best])} · run '{run}' ({ctx['run_type']})"
        elif mod == "vision":
            big, sub = "Waiting", "for Component 2's trained model"
        else:
            big, sub = "No results", "run the EEG pilot first"
        col.markdown(f'<div class="card"><h4>{title}</h4><div class="big">{big}</div><div class="sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>Multimodal (EEG + video)</h4><div class="big">Not built</div>'
                '<div class="sub">no paired EEG + video from the same learners</div></div>', unsafe_allow_html=True)
    st.markdown("#### How to read this dashboard")
    st.markdown("- **EEG** and **Computer Vision**: the full study for each prediction context, in research order.\n"
                "- **Multimodal**: why there is no fusion model.\n"
                "- **Cross-context comparison**: is the same method the most suitable in both contexts? (It compares "
                "methods, never which modality explains a cognitive state better.)\n"
                "- **Output for Component 4** and **Glossary**.")


# =========================================================================== Component 4 (both contexts)
def page_c4_all(eeg_run: str | None, vision_run: str | None):
    import views_b as B
    from common import set_modality

    if eeg_run:
        set_modality("eeg")
        B.page_c4(eeg_run)
    else:
        st.title("Output for Component 4")
        no_results("export")
    st.markdown("### Vision engagement context")
    p = RESULTS / vision_run / "component4_export.json" if vision_run else None
    if p is None or not p.exists():
        st.info(VISION_WAITING)
        st.caption("Until then the export's `modalities.vision.status` is \"not_available\"; the EEG part of the "
                   "export is unchanged.")
        return
    doc = json.loads(p.read_text())
    for c in doc.get("contexts", []):
        st.markdown(f"**{c['title']}** — {c['plain_summary']}")
    vw = doc.get("vision_windows", [])
    if vw:
        ids = [w["instance_id"] for w in vw]
        iid = st.selectbox("Example video window", ids)
        w = vw[ids.index(iid)]
        rows = []
        for lvl, name in (("group", "Face region"), ("segment", "Time segment"), ("feature_x_segment", "Measurement × second")):
            for it in w["top_k"].get(lvl, []):
                arrow = "↑ towards high" if it["direction"] == "towards_target" else "↓ towards low"
                rows.append({"level": name, "feature": GROUP_PLAIN.get(it["name"], it["name"]),
                             "effect": fmt(it["attribution"], 3), "direction": arrow})
        st.write(f"Prediction: **{w['prediction']['predicted_class']} engagement** (confidence "
                 f"{pct(w['prediction']['confidence'])}), trust **{w['trust']['level']}**.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Download the vision export (JSON)", p.read_bytes(), file_name="component4_export_vision.json",
                       mime="application/json")
