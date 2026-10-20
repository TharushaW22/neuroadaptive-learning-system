"""Shared helpers for the dashboard: reading results (no heavy recomputation), colours, layout pieces.

Everything shown is read from results/<run>/...; nothing is re-computed except light summaries
(means, sorting). Missing results produce a friendly message, never a stack trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

C3 = Path(__file__).resolve().parents[1]
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(C3 / ".pycache")
if str(C3 / "src") not in sys.path:
    sys.path.insert(0, str(C3 / "src"))

RESULTS = C3 / "results"
PILOT_CMD = ".venv/bin/python scripts/run_experiment.py --pilot"

# ---------------------------------------------------------------- colours (one meaning each)
EXPL_CFG = yaml.safe_load(open(C3 / "configs" / "explainers.yaml", encoding="utf-8"))["methods"]
METHOD_COLOUR = {m: c["colour"] for m, c in EXPL_CFG.items()}
METHOD_LABEL = {m: c["label"] for m, c in EXPL_CFG.items()}
METHOD_PLAIN = {m: c["plain"] for m, c in EXPL_CFG.items()}
RANKED = ["kernel_shap", "conditional_shap", "lime", "ig"]

# attributions: blue = pushes towards LOW workload, orange = towards HIGH, grey = no effect
DIVERGING = [[0.0, "#1F5FA8"], [0.25, "#7FA6D3"], [0.5, "#E3E3E3"], [0.75, "#F0AA66"], [1.0, "#C65D00"]]
SEQUENTIAL = [[0.0, "#F2F5FA"], [1.0, "#1F4E79"]]                 # one hue, light -> dark (agreement etc.)
STATUS = {"pass": ("#1A7F37", "✓"), "caution": ("#9A6700", "⚠"), "fail": ("#6E7781", "✕"),
          "not assessed": ("#6E7781", "–"), "na": ("#6E7781", "–")}
INK, MUTED, GRID = "#1F2933", "#52606D", "#E4E7EB"

# 2-D top view of the 14 Emotiv EPOC channels (nose at the top)
HEAD_XY = {"AF3": (-0.27, 0.83), "AF4": (0.27, 0.83), "F7": (-0.68, 0.55), "F8": (0.68, 0.55),
           "F3": (-0.33, 0.52), "F4": (0.33, 0.52), "FC5": (-0.58, 0.25), "FC6": (0.58, 0.25),
           "T7": (-0.9, -0.02), "T8": (0.9, -0.02), "P7": (-0.66, -0.55), "P8": (0.66, -0.55),
           "O1": (-0.27, -0.86), "O2": (0.27, -0.86)}

BAND_PLAIN = {"slow": "slow (below 1 Hz; drifts, eye movements)", "delta": "delta (1–4 Hz)", "theta": "theta (4–8 Hz)",
              "alpha": "alpha (8–13 Hz)", "beta": "beta (13–30 Hz)", "gamma": "gamma (30–45 Hz)"}

CRITERIA_PLAIN = {
    "fidelity_on": ("Does the explanation match what the model really does? (Fidelity, realistic removal)", "+",
                    "We remove the EEG parts the explanation calls most important (replacing them with realistic EEG "
                    "from other people) and watch the prediction. If it changes faster than when removing parts in "
                    "random order, the explanation is faithful. Higher is better; 0 means no better than random."),
    "fidelity_off": ("Does the explanation match what the model really does? (Fidelity, zero removal)", "+",
                     "Same test, but removed parts are simply set to zero (an unrealistic signal). Higher is better."),
    "stability": ("Same answer when run again? (Stability)", "+",
                  "Each method is run again with a different random seed on the same window. 1 means the ranking "
                  "of important features is identical, 0 means unrelated. Higher is better."),
    "robustness": ("Same answer when the EEG is slightly noisy? (Robustness)", "+",
                   "A little random noise (10 % of each channel's typical size) is added. If the prediction stays the "
                   "same, the explanation should too. 1 = unchanged. Higher is better."),
    "sparsity": ("Short, focused explanation? (Sparsity)", "+",
                 "Measures whether a few features carry most of the importance (Gini index, 0 = everything equally "
                 "important, 1 = one feature). Higher is easier to read, but says nothing about correctness."),
    "cost_seconds": ("How long does it take? (Cost)", "-",
                     "Seconds to explain one window on one CPU core. Lower is better."),
}


# ---------------------------------------------------------------- data access
def list_runs() -> list[str]:
    if not RESULTS.exists():
        return []
    runs = [p.parent.parent.relative_to(RESULTS).as_posix() for p in RESULTS.glob("*/*/run_info.json")
            if not p.parent.parent.name.startswith("_")]
    return sorted(set(runs), key=lambda r: (r != "full", r != "pilot", r))


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict | list | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def contexts(run: str) -> list[dict]:
    out = []
    for p in sorted((RESULTS / run).glob("*/run_info.json")):
        info = read_json(str(p))
        info["_dir"] = str(p.parent)
        out.append(info)
    return out


def primary_contexts(run: str) -> list[dict]:
    return [c for c in contexts(run) if not c.get("negative_control")]


def control_for(run: str, context_id: str) -> dict | None:
    return next((c for c in contexts(run) if c.get("control_of") == context_id), None)


def table(ctx: dict, name: str) -> pd.DataFrame | None:
    return read_parquet(str(Path(ctx["_dir"]) / "tables" / f"{name}.parquet"))


def analysis(ctx: dict, name: str):
    p = Path(ctx["_dir"]) / "analysis" / name
    return read_json(str(p)) if name.endswith(".json") else read_parquet(str(p))


@st.cache_data(show_spinner=False)
def explanation_values(ctx_dir: str, instance_id: str, method: str, seed: int) -> np.ndarray | None:
    p = Path(ctx_dir) / "explanations" / instance_id / f"{method}__s{seed}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["values"]


# ---------------------------------------------------------------- formatting
def fmt(x, digits: int = 2) -> str:
    """Round to `digits` significant figures; '–' for missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    from math import floor, log10
    d = max(digits - 1 - int(floor(log10(abs(x)))), 0)
    return f"{x:.{min(d, 4)}f}"


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * float(x):.0f}%"


def label(m: str) -> str:
    return METHOD_LABEL.get(m, m)


# ---------------------------------------------------------------- layout pieces
def apply_style() -> None:
    big = st.session_state.get("big_text", False)
    base = 19 if big else 16
    st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-size: {base}px; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
.summary {{ background:#F4F6F9; border-left:4px solid #1F4E79; padding:0.8rem 1rem; border-radius:6px;
            font-size:{base + 2}px; line-height:1.45; margin-bottom:1rem; color:{INK}; }}
.banner {{ padding:0.45rem 0.9rem; border-radius:6px; font-weight:600; margin-bottom:0.6rem; font-size:{base}px; }}
.banner.amber {{ background:#FFF4D6; border:1px solid #E9C46A; color:#5C4300; }}
.banner.neutral {{ background:#EEF2F6; border:1px solid #CBD2D9; color:{INK}; }}
.card {{ border:1px solid #E4E7EB; border-radius:10px; padding:1rem 1.1rem; background:#FFFFFF; height:100%; }}
.card h4 {{ margin:0 0 0.3rem 0; color:{MUTED}; font-weight:600; font-size:{base - 1}px; }}
.card .big {{ font-size:{base + 12}px; font-weight:700; color:{INK}; }}
.card .sub {{ color:{MUTED}; font-size:{base - 1}px; }}
.swatch {{ display:inline-block; width:0.8em; height:0.8em; border-radius:2px; margin-right:0.35em; vertical-align:-0.05em; }}
.howto {{ color:{MUTED}; font-size:{base - 1}px; margin-top:-0.4rem; }}
.pipe {{ display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center; }}
.pipe div.step {{ background:#F4F6F9; border:1px solid #CBD2D9; border-radius:8px; padding:0.45rem 0.7rem; font-size:{base - 1}px; }}
.pipe span.arrow {{ color:{MUTED}; }}
</style>""", unsafe_allow_html=True)


def plotly_layout(fig, height: int = 380, **kw):
    big = st.session_state.get("big_text", False)
    fig.update_layout(template="plotly_white", height=height, font=dict(size=16 if big else 13, color=INK),
                      margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_size=14),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#CBD2D9")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#CBD2D9")
    return fig


def banner(run: str, ctx: dict | None = None) -> None:
    ctxs = [ctx] if ctx else primary_contexts(run)
    tags = sorted({t for c in ctxs for t in c.get("run_tags", [])}) or ["FINAL"]
    amber = any(t in ("PILOT", "IN-SAMPLE") for t in tags)
    parts = []
    if "PILOT" in tags:
        parts.append("PILOT run — small budget, for checking the pipeline")
    if "IN-SAMPLE" in tags:
        parts.append("IN-SAMPLE — deployed model trained on all subjects; not for final results")
    if not parts:
        parts.append("FINAL run — out-of-sample fold models")
    st.markdown(f'<div class="banner {"amber" if amber else "neutral"}">Run type: {" + ".join(tags)} · '
                f'{" · ".join(parts)}</div>', unsafe_allow_html=True)


def summary(text: str) -> None:
    st.markdown(f'<div class="summary">{text}</div>', unsafe_allow_html=True)


def howto(text: str) -> None:
    st.markdown(f'<div class="howto"><b>How to read this chart:</b> {text}</div>', unsafe_allow_html=True)


def meaning(title: str, text: str, direction: str | None = None) -> None:
    with st.expander(f"What does this mean? — {title}"):
        st.write(text)
        if direction:
            st.write("**Higher is better.**" if direction == "+" else "**Lower is better.**")


def status_badge(status: str) -> str:
    colour, icon = STATUS.get(status, STATUS["na"])
    text = {"pass": "OK", "caution": "caution", "fail": "excluded", "not assessed": "not assessed"}.get(status, status)
    return f'<span style="color:{colour};font-weight:600">{icon} {text}</span>'


def method_chip(m: str) -> str:
    return f'<span class="swatch" style="background:{METHOD_COLOUR.get(m, "#999")}"></span>{label(m)}'


def legend_row(methods: list[str]) -> None:
    st.markdown(" &nbsp;&nbsp; ".join(method_chip(m) for m in methods), unsafe_allow_html=True)


def no_results(what: str = "results") -> None:
    st.info(f"No {what} yet — run the pilot with:\n\n`{PILOT_CMD}`\n\n(from the component3/ folder), "
            "then reload this page.")
