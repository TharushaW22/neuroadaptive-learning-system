"""Reusable Streamlit UI pieces (native Streamlit + a small amount of plain CSS, no Bootstrap).

The app is light-theme only (see .streamlit/config.toml); the CSS below uses the same palette.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.utils.status import RESEARCH_MODE

CSS = """
<style>
:root {
  color-scheme: light;
  --ink:#1f2933; --muted:#52606d; --line:#dde3ea; --panel:#ffffff; --soft:#f3f5f8;
  --accent:#0f6b7a; --accent-soft:#e6f2f4;
  --ok:#1b6e3a; --ok-soft:#e8f4ec; --warn:#8a5300; --warn-soft:#fdf4e3; --bad:#a3192e; --bad-soft:#fbeaec;
}
[data-testid="stApp"], [data-testid="stHeader"] { background:#ffffff; color:var(--ink); }
[data-testid="stMainBlockContainer"] { padding-top:4.5rem !important; padding-bottom:3rem !important;
                                    max-width:1200px; }
[data-testid="stHeader"] { background:rgba(255,255,255,.92); }

/* headings */
.vcbu-header { margin:0 0 1.1rem; padding-bottom:.8rem; border-bottom:1px solid var(--line); }
.vcbu-header h1 { font-size:1.6rem; font-weight:650; margin:0; padding:0; color:var(--ink); line-height:1.25; }
.vcbu-header p { margin:.3rem 0 0; color:var(--muted); font-size:.95rem; }
.vcbu-section { margin:1.6rem 0 .6rem; padding-top:1rem; border-top:1px solid var(--line);
                display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }
.vcbu-section h3 { font-size:1.05rem; font-weight:600; margin:0; padding:0; color:var(--ink); }
.vcbu-section span { color:var(--muted); font-size:.88rem; }
.vcbu-section.first { border-top:0; padding-top:0; margin-top:.4rem; }

/* status pills */
.vcbu-pill { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.8rem;
             font-weight:600; letter-spacing:.02em; border:1px solid transparent; white-space:nowrap; }
.vcbu-pill.ok { background:var(--ok-soft); color:var(--ok); border-color:#b9dcc5; }
.vcbu-pill.warn { background:var(--warn-soft); color:var(--warn); border-color:#f0d49c; }
.vcbu-pill.bad { background:var(--bad-soft); color:var(--bad); border-color:#efbcc4; }
.vcbu-pill.info { background:var(--accent-soft); color:var(--accent); border-color:#b8dbe1; }

/* mode banner */
.vcbu-banner { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; padding:.55rem .9rem;
               border-radius:8px; border:1px solid var(--line); background:var(--soft);
               margin-bottom:1rem; font-size:.9rem; color:var(--muted); }

/* cards */
.vcbu-card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
             padding:.7rem .9rem; margin-bottom:.6rem; }
.vcbu-card .lbl { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }
.vcbu-card .row { display:flex; justify-content:space-between; align-items:center; gap:.5rem; margin-top:.25rem; }
.vcbu-card .val { font-size:1.02rem; font-weight:600; color:var(--ink); overflow-wrap:anywhere; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:8px;
                           padding:.6rem .9rem; }
[data-testid="stMetricLabel"] p { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
[data-testid="stMetricValue"] { font-size:1.45rem; font-weight:600; font-variant-numeric:tabular-nums; }

/* live prediction */
.vcbu-state { font-size:1.35rem; font-weight:650; padding:.6rem .9rem; border-radius:8px;
              border:1px solid var(--line); border-left:4px solid var(--accent); background:var(--soft);
              color:var(--ink); overflow-wrap:anywhere; }
.vcbu-state-lbl { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; margin-bottom:.25rem; }
.vcbu-scores { display:flex; flex-wrap:wrap; gap:.5rem; margin:.6rem 0 .25rem; }
.vcbu-score { flex:1 1 6rem; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:.45rem .65rem; }
.vcbu-score .lbl { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }
.vcbu-score .val { font-size:1.3rem; font-weight:650; font-variant-numeric:tabular-nums; color:var(--ink); }
.vcbu-score .pct { color:var(--muted); font-size:.82rem; }
.vcbu-note { color:var(--muted); font-size:.88rem; margin:.3rem 0 .6rem; line-height:1.45; }
.vcbu-chart-title { font-size:.9rem; font-weight:600; color:var(--ink); margin:.8rem 0 .1rem; }

/* sidebar */
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
.vcbu-brand { font-size:1.15rem; font-weight:700; color:var(--ink); margin:0; }
.vcbu-brand-sub { color:var(--muted); font-size:.82rem; margin:.1rem 0 .9rem; line-height:1.35; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.28rem .4rem; border-radius:6px; margin:0; width:100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#e8edf2; }
.vcbu-side-foot { color:var(--muted); font-size:.8rem; margin-top:1rem; padding-top:.7rem; border-top:1px solid var(--line); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, level: str = "info") -> str:
    """HTML for a small status pill (level: ok | warn | bad | info)."""
    return f'<span class="vcbu-pill {level}">{html.escape(str(text))}</span>'


def page_header(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="vcbu-header"><h1>{html.escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None, first: bool = False) -> None:
    cap = f"<span>{html.escape(caption)}</span>" if caption else ""
    st.markdown(f'<div class="vcbu-section{" first" if first else ""}"><h3>{html.escape(title)}</h3>{cap}</div>',
                unsafe_allow_html=True)


def mode_banner(status: dict) -> None:
    research = status["mode"] == RESEARCH_MODE
    text = ("Real DAiSEE files detected." if research else
            "No real dataset detected. Synthetic data and mock inputs only; results are not research findings.")
    st.markdown(f'<div class="vcbu-banner" role="status">{pill(status["mode"], "ok" if research else "warn")}'
                f"<span>{html.escape(text)}</span></div>", unsafe_allow_html=True)


def status_card(label: str, value, level: str = "ok", tag: str | None = None) -> None:
    """Card with a label, a value and an optional status pill."""
    badge = pill(tag or {"ok": "ready", "warn": "pending", "bad": "missing"}.get(level, level), level)
    st.markdown(f'<div class="vcbu-card"><div class="lbl">{html.escape(label)}</div>'
                f'<div class="row"><div class="val">{html.escape(str(value))}</div>{badge}</div></div>',
                unsafe_allow_html=True)


def state_box(text: str, detail: str | None = None, target=None) -> None:
    """Prediction state (and optional reason) as ONE element, so placeholders never keep stale parts."""
    body = (f'<div class="vcbu-state-lbl">Engagement prediction</div>'
            f'<div class="vcbu-state" aria-live="polite">{html.escape(text)}</div>')
    if detail:
        body += f'<p class="vcbu-note">{html.escape(detail)}</p>'
    (target or st).markdown(body, unsafe_allow_html=True)


def score_box(row: dict, target=None) -> None:
    """Heuristic attention / fatigue / readiness scores as 0-1 and %, labelled experimental."""
    from src.models.interpretation import QUALITY_OK, format_score
    items = []
    for key, label in (("attention_score", "Attention"), ("fatigue_score", "Fatigue"),
                       ("readiness_score", "Readiness")):
        v = row.get(key)
        pct = format_score(v, percent=True)
        items.append(f'<div class="vcbu-score"><div class="lbl">{label}</div>'
                     f'<div class="val">{html.escape(format_score(v))}</div>'
                     f'<div class="pct">{html.escape(pct if pct != "UNKNOWN" else "-")}</div></div>')
    q = row.get("quality_state", "UNKNOWN")
    reason = "" if q == QUALITY_OK and row.get("interpretation_reason") == "ok" else \
        f" ({row.get('interpretation_reason', '')})"
    body = (f'<div class="vcbu-scores" aria-live="polite">{"".join(items)}</div>'
            f'<p class="vcbu-note">{pill("Heuristic", "warn")} '
            f'{html.escape(str(row.get("interpretation_source", "")))} - quality {html.escape(str(q))}'
            f'{html.escape(reason)}. Not validated measurements.</p>')
    (target or st).markdown(body, unsafe_allow_html=True)


def _md_safe(text: str) -> str:
    """HTML-escape and neutralise markdown link/emphasis syntax (e.g. "[0,1]")."""
    return html.escape(text).replace("[", "&#91;").replace("]", "&#93;").replace("*", "&#42;")


def note(text: str) -> None:
    st.markdown(f'<p class="vcbu-note">{_md_safe(text)}</p>', unsafe_allow_html=True)


def show_frame(placeholder, frame_bgr, max_width: int = 640, quality: int = 75) -> None:
    """Render a BGR frame as an inline base64 JPEG.

    Avoids st.image's media-file storage, which in fast loops deletes frames before the
    browser fetches them (HTTP 500 / 'Image source error' in the browser console).
    """
    import base64

    import cv2

    h, w = frame_bgr.shape[:2]
    if w > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    placeholder.markdown(f'<img src="data:image/jpeg;base64,{b64}" alt="Live camera frame" '
                         f'style="width:100%;max-width:{max_width}px;border-radius:8px;border:1px solid #dde3ea;">',
                         unsafe_allow_html=True)


def image_file(path, caption: str | None = None, target=None) -> None:
    """Show a PNG/JPEG file inline (base64). Avoids Streamlit's media-file server, which fails
    to serve st.image files on some versions (blank 'broken image' icons)."""
    import base64
    from pathlib import Path

    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    cap = f'<figcaption class="vcbu-note">{html.escape(caption)}</figcaption>' if caption else ""
    (target or st).markdown(f'<figure style="margin:0 0 .8rem"><img src="data:{mime};base64,{b64}" '
                            f'alt="{html.escape(caption or p.stem)}" style="width:100%;border:1px solid #dde3ea;'
                            f'border-radius:8px;background:#fff">{cap}</figure>', unsafe_allow_html=True)


def timeline(df: pd.DataFrame, cols: list[str], title: str, height: int = 220) -> None:
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        st.caption(f"{title}: no data.")
        return
    st.markdown(f'<div class="vcbu-chart-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.line_chart(df.set_index("timestamp_s")[cols], height=height)
