"""Styling helpers for the Streamlit dashboard.

Centralises the inline CSS and the small set of HTML components that the rest
of the app reuses (hero header, KPI cards, callouts). Streamlit doesn't expose
themable card / hero primitives, so we render plain HTML inside
``st.markdown(unsafe_allow_html=True)``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st


PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E40AF"
ACCENT = "#10B981"
WARN = "#F59E0B"
DANGER = "#EF4444"
SLATE_50 = "#F8FAFC"
SLATE_100 = "#F1F5F9"
SLATE_200 = "#E2E8F0"
SLATE_500 = "#64748B"
SLATE_900 = "#0F172A"

PLOTLY_COLORWAY = [PRIMARY, ACCENT, WARN, DANGER, "#8B5CF6", "#0EA5E9"]


GLOBAL_CSS = f"""
<style>
:root {{
  --primary: {PRIMARY};
  --primary-dark: {PRIMARY_DARK};
  --accent: {ACCENT};
  --warn: {WARN};
  --danger: {DANGER};
  --slate-50: {SLATE_50};
  --slate-100: {SLATE_100};
  --slate-200: {SLATE_200};
  --slate-500: {SLATE_500};
  --slate-900: {SLATE_900};
}}

/* widen the main content area and add breathing room */
.block-container {{
  max-width: 1280px;
  padding-top: 1.2rem;
  padding-bottom: 4rem;
  padding-left: 2rem;
  padding-right: 2rem;
}}

/* hide streamlit chrome */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* hero banner */
.hero {{
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
  color: white;
  border-radius: 18px;
  padding: 2rem 2.25rem;
  box-shadow: 0 18px 40px -22px rgba(37, 99, 235, 0.55);
  margin-bottom: 1.5rem;
}}
.hero h1 {{
  font-size: 2.1rem;
  font-weight: 700;
  margin: 0 0 0.45rem 0;
  color: white;
  letter-spacing: -0.01em;
}}
.hero p.subtitle {{
  margin: 0;
  font-size: 1.02rem;
  color: rgba(255,255,255,0.85);
  max-width: 720px;
  line-height: 1.5;
}}
.hero .kpi-strip {{
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
  flex-wrap: wrap;
}}
.hero .kpi-pill {{
  flex: 1 1 160px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  backdrop-filter: blur(2px);
}}
.hero .kpi-pill .label {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: rgba(255,255,255,0.78);
  margin-bottom: 0.25rem;
}}
.hero .kpi-pill .value {{
  font-size: 1.55rem;
  font-weight: 700;
  color: white;
  line-height: 1.1;
}}
.hero .kpi-pill .delta {{
  font-size: 0.78rem;
  color: rgba(255,255,255,0.78);
  margin-top: 0.15rem;
}}

/* generic content cards */
.card {{
  background: #fff;
  border-radius: 14px;
  padding: 1.25rem 1.4rem;
  border: 1px solid var(--slate-200);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  margin-bottom: 1rem;
}}
.card h3 {{
  margin: 0 0 0.5rem 0;
  font-size: 1.05rem;
  color: var(--slate-900);
  font-weight: 600;
}}
.card p {{
  margin: 0;
  color: var(--slate-500);
  font-size: 0.92rem;
  line-height: 1.5;
}}

/* KPI grid card */
.kpi-card {{
  background: #fff;
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
  border: 1px solid var(--slate-200);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}
.kpi-card .label {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--slate-500);
  font-weight: 600;
  margin-bottom: 0.35rem;
}}
.kpi-card .value {{
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--slate-900);
  line-height: 1.05;
}}
.kpi-card .delta {{
  font-size: 0.82rem;
  color: var(--slate-500);
  margin-top: 0.25rem;
}}
.kpi-card.accent {{ border-top: 3px solid var(--primary); }}
.kpi-card.success {{ border-top: 3px solid var(--accent); }}
.kpi-card.warn {{ border-top: 3px solid var(--warn); }}
.kpi-card.danger {{ border-top: 3px solid var(--danger); }}

/* callouts */
.callout {{
  border-radius: 12px;
  padding: 1rem 1.15rem;
  margin: 0.85rem 0;
  border-left: 4px solid var(--primary);
  background: var(--slate-50);
  color: var(--slate-900);
  font-size: 0.93rem;
  line-height: 1.55;
}}
.callout.warn {{
  border-left-color: var(--warn);
  background: #FFFBEB;
}}
.callout.danger {{
  border-left-color: var(--danger);
  background: #FEF2F2;
}}
.callout.success {{
  border-left-color: var(--accent);
  background: #ECFDF5;
}}
.callout strong {{ color: var(--slate-900); }}

/* prediction flag card */
.flag-card {{
  border-radius: 16px;
  padding: 1.6rem 1.4rem;
  text-align: center;
  color: white;
  box-shadow: 0 12px 28px -16px rgba(0,0,0,0.25);
}}
.flag-card.high {{ background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%); }}
.flag-card.low {{ background: linear-gradient(135deg, #10B981 0%, #047857 100%); }}
.flag-card .label {{
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  opacity: 0.85;
  margin-bottom: 0.4rem;
}}
.flag-card .value {{
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.flag-card .sub {{
  font-size: 0.85rem;
  opacity: 0.85;
  margin-top: 0.4rem;
}}

/* sidebar polish */
section[data-testid="stSidebar"] {{
  background: var(--slate-50);
  border-right: 1px solid var(--slate-200);
}}
.sidebar-snapshot {{
  background: white;
  border: 1px solid var(--slate-200);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin-top: 1rem;
}}
.sidebar-snapshot .row {{
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  margin: 0.18rem 0;
  color: var(--slate-500);
}}
.sidebar-snapshot .row span.value {{
  color: var(--slate-900);
  font-weight: 600;
}}
.sidebar-snapshot h4 {{
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--slate-500);
  margin: 0 0 0.5rem 0;
  font-weight: 600;
}}

/* dataframe shadow trim */
[data-testid="stDataFrame"] {{
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--slate-200);
}}

/* tab styling tweak */
.stTabs [data-baseweb="tab-list"] {{
  gap: 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
  padding: 0.5rem 1rem;
  border-radius: 8px 8px 0 0;
}}

/* section heading */
.section-title {{
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--slate-900);
  margin: 1.2rem 0 0.6rem 0;
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
}}
.section-title .accent-bar {{
  width: 4px;
  height: 18px;
  background: var(--primary);
  border-radius: 2px;
  display: inline-block;
}}
.section-title small {{
  font-size: 0.85rem;
  color: var(--slate-500);
  font-weight: 400;
}}
</style>
"""


def inject_global_css() -> None:
    """Call once at the top of the app."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def _format_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_hero(
    title: str,
    subtitle: str,
    kpis: Sequence[tuple[str, object, str | None]] = (),
) -> None:
    """Render a gradient hero banner with optional KPI strip.

    ``kpis`` is a sequence of ``(label, value, delta_text_or_None)`` tuples.
    """
    pills = ""
    for label, value, delta in kpis:
        delta_html = f'<div class="delta">{delta}</div>' if delta else ""
        pills += (
            '<div class="kpi-pill">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{_format_value(value)}</div>'
            f"{delta_html}"
            "</div>"
        )
    strip = f'<div class="kpi-strip">{pills}</div>' if pills else ""
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p class="subtitle">{subtitle}</p>
          {strip}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: object,
    delta: str | None = None,
    color: str | None = None,
) -> str:
    """Return KPI card HTML (caller wraps in st.markdown)."""
    cls = "kpi-card"
    if color in {"accent", "success", "warn", "danger"}:
        cls += f" {color}"
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (
        f'<div class="{cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{_format_value(value)}</div>'
        f"{delta_html}"
        "</div>"
    )


def kpi_row(items: Iterable[tuple[str, object, str | None, str | None]]) -> None:
    """Render a 1-row grid of KPI cards.

    Each item is ``(label, value, delta_or_None, color_or_None)``.
    """
    items = list(items)
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value, delta, color) in zip(cols, items):
        with col:
            st.markdown(kpi_card(label, value, delta, color), unsafe_allow_html=True)


def callout(text: str, kind: str = "info", title: str | None = None) -> None:
    """Render a colored callout. ``kind`` ∈ {info, warn, danger, success}."""
    cls = "callout" if kind == "info" else f"callout {kind}"
    title_html = f"<strong>{title}</strong><br/>" if title else ""
    st.markdown(
        f'<div class="{cls}">{title_html}{text}</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    sub = f"<small>{subtitle}</small>" if subtitle else ""
    st.markdown(
        f'<div class="section-title"><span class="accent-bar"></span>'
        f"<span>{title}</span>{sub}</div>",
        unsafe_allow_html=True,
    )


def flag_card(probability: float, threshold: float) -> None:
    """Big colored card showing the binary screening flag."""
    high = probability >= threshold
    cls = "flag-card high" if high else "flag-card low"
    label = "POSITIVE SCREEN" if high else "NEGATIVE SCREEN"
    sub = (
        f"Probability ≥ threshold ({threshold:.2f}) — recommend follow-up"
        if high
        else f"Probability &lt; threshold ({threshold:.2f}) — routine"
    )
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{probability:.1%}</div>'
        f'<div class="sub">{sub}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar_snapshot(rows: Sequence[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="row"><span>{k}</span><span class="value">{v}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="sidebar-snapshot"><h4>Model snapshot</h4>{body}</div>',
        unsafe_allow_html=True,
    )
