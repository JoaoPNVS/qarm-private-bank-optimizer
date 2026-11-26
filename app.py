"""Private Client Portfolio Optimizer — Streamlit UI only (no heavy logic here)."""
from __future__ import annotations

import html
import json
import logging
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (backtest as btm, benchmarks as bm, bonds, data, diversification as div, estimators as est,
                 frontier as fr, insiders as ins, optimizer as opt, profiling as prof, risk, team)

st.set_page_config(page_title="Private Client Portfolio Optimizer", layout="wide",
                   initial_sidebar_state="collapsed")

# Dark theme only — chart palette tuned for readability on the dark navy background.
# Colour themes: page colours (CSS variables) and chart palette in one place. The page colours of
# Streamlit's own widgets come from .streamlit/config.toml, which must use the same base (light/dark).
THEMES = {
    "night": {"base": "dark", "bg": "#0d1a2b", "surface": "#122238", "surface_2": "#1a2d47", "border": "#2b4466",
              "text": "#e6ecf5", "text_dim": "#9aa9bf", "accent": "#5a92c4", "accent_hover": "#7bace0",
              "accent_soft": "rgba(90,146,196,0.12)", "hover": "rgba(255,255,255,0.08)",
              "shadow": "0 8px 32px rgba(0,0,0,0.5)", "dock_bg": "rgba(26,45,71,0.95)", "dock_border": "rgba(123,172,224,0.35)",
              "dock_shadow": "0 12px 40px rgba(0,0,0,0.6), 0 0 32px rgba(90,146,196,0.18)",
              "plotly": "plotly_dark", "chart_text": "#e6ecf5",
              "navy": "#7bace0", "grey": "#8f9bac", "red": "#e07f6b", "ew": "#b39ddb", "bond": "#3f6694",
              "gold": "#c9a227", "cash": "#d9d9d9", "fill_main": "rgba(123,172,224,0.28)", "fill_bench": "rgba(143,155,172,0.20)"},
    "slate": {"base": "dark", "bg": "#1c2838", "surface": "#243349", "surface_2": "#2d3f58", "border": "#3d5373",
              "text": "#eef2f7", "text_dim": "#b0bccd", "accent": "#6aa0d4", "accent_hover": "#8bb8e3",
              "accent_soft": "rgba(106,160,212,0.14)", "hover": "rgba(255,255,255,0.08)",
              "shadow": "0 8px 28px rgba(0,0,0,0.35)", "dock_bg": "rgba(45,63,88,0.96)", "dock_border": "rgba(139,184,227,0.35)",
              "dock_shadow": "0 10px 32px rgba(0,0,0,0.45)",
              "plotly": "plotly_dark", "chart_text": "#eef2f7",
              "navy": "#8bb8e3", "grey": "#a3afbf", "red": "#e8907d", "ew": "#c3b1e1", "bond": "#4f79a8",
              "gold": "#d4b041", "cash": "#e3e3e3", "fill_main": "rgba(139,184,227,0.28)", "fill_bench": "rgba(163,175,191,0.20)"},
    "mist": {"base": "light", "bg": "#eef2f7", "surface": "#ffffff", "surface_2": "#e4eaf2", "border": "#c9d3e0",
             "text": "#1c2833", "text_dim": "#55657a", "accent": "#2f5f8f", "accent_hover": "#3d73a8",
             "accent_soft": "rgba(47,95,143,0.10)", "hover": "rgba(28,40,51,0.06)",
             "shadow": "0 6px 22px rgba(28,40,51,0.12)", "dock_bg": "rgba(255,255,255,0.96)", "dock_border": "rgba(47,95,143,0.30)",
             "dock_shadow": "0 10px 30px rgba(28,40,51,0.18)",
             "plotly": "plotly_white", "chart_text": "#1c2833",
             "navy": "#1f4e79", "grey": "#7f8c9d", "red": "#c0392b", "ew": "#7e57c2", "bond": "#8fb3d9",
             "gold": "#b8860b", "cash": "#b8bfca", "fill_main": "rgba(31,78,121,0.20)", "fill_bench": "rgba(127,140,157,0.20)"},
    "ivory": {"base": "light", "bg": "#f6f4ef", "surface": "#ffffff", "surface_2": "#eeeae2", "border": "#d9d3c7",
              "text": "#1f2a37", "text_dim": "#62666d", "accent": "#1f3b57", "accent_hover": "#2e5580",
              "accent_soft": "rgba(31,59,87,0.08)", "hover": "rgba(31,42,55,0.06)",
              "shadow": "0 6px 22px rgba(31,42,55,0.10)", "dock_bg": "rgba(255,255,255,0.96)", "dock_border": "rgba(31,59,87,0.25)",
              "dock_shadow": "0 10px 30px rgba(31,42,55,0.16)",
              "plotly": "plotly_white", "chart_text": "#1f2a37",
              "navy": "#1f3b57", "grey": "#8a9099", "red": "#b03a2e", "ew": "#7e57c2", "bond": "#9db7d1",
              "gold": "#b08d2f", "cash": "#c4c0b8", "fill_main": "rgba(31,59,87,0.18)", "fill_bench": "rgba(138,144,153,0.20)"},
}
THEME = THEMES[os.environ.get("APP_THEME", "mist")]
NAVY, GREY, ACCENT = THEME["navy"], THEME["grey"], THEME["red"]
EW_COLOR, BOND_COLOR, GOLD_COLOR, CASH_COLOR = THEME["ew"], THEME["bond"], THEME["gold"], THEME["cash"]
CLASS_COLORS = [NAVY, BOND_COLOR, GOLD_COLOR, CASH_COLOR]  # Equity, Bond, Commodity, Cash
CLASS_ORDER = ["Equity", "Bond", "Commodity", "Cash"]
CLASS_LABELS = {"Equity": "Equities", "Bond": "Bonds", "Commodity": "Gold", "Cash": "Cash & equivalents"}
WINDOW = est.DEFAULT_WINDOW
EXPECTED_ERRORS = (data.DataError, est.EstimationError, btm.BacktestError, ValueError)
log = logging.getLogger("app")


@contextmanager
def guarded(section: str):
    """Last line of defence: an unexpected error shows a clear message (details go to the server
    log) instead of a Python traceback. Streamlit's st.stop/rerun derive from BaseException and
    pass through."""
    try:
        yield
    except Exception:
        log.exception("Unexpected error in %s", section)
        st.error(f"Something went wrong while building the {section} view. Please change the inputs "
                 "and try again; the other tabs are not affected.")


# --- helpers ---------------------------------------------------------------------------------

def usd(x: float) -> str:
    return f"USD {x:,.0f}"


def usd_short(x: float) -> str:
    """Compact amount for metric tiles (the exact amount goes in tables and help texts)."""
    for size, unit in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if abs(x) >= size:
            return f"USD {x / size:,.{2 if unit == 'bn' else 1 if unit == 'm' else 0}f}{unit}"
    return usd(x)


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


EW_LABEL = "Equal weight (same universe)"
INSIDER, PAST = "Insider trading", "Past performance"   # Black-Litterman views: the house view, and the naive
                                                        # comparison shown only in the Backtest tab


def bench_label(profile: str) -> str:
    e = prof.SPECS[profile].benchmark_equity
    return "Benchmark (100% SPY)" if e >= 1 else f"Benchmark ({e:.0%} SPY / {1 - e:.0%} AGG)"


def fig_layout(fig, height=380):
    # legend below the plot: above it, it collided with the title and wrapped over narrow charts
    fig.update_layout(template=THEME["plotly"],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.42, x=0, xanchor="left"),
                      font=dict(size=13, color=THEME["chart_text"]), title=dict(x=0, xanchor="left"))
    return fig


@st.cache_data(show_spinner=False)
def load_data():
    return data.load_universe(), data.load_returns(), data.load_rf(), data.load_benchmark_returns()


@st.cache_data(show_spinner=False)
def get_estimates() -> est.Estimates:
    universe, returns, rf, bench = load_data()
    window = est.estimation_window(returns, returns.index[-1], WINDOW)
    return est.estimate(window, rf, est.shrinkage_groups(universe), benchmarks=bench,
                        benchmark_class=bm.BENCHMARK_CLASS)


@st.cache_data(show_spinner=False)
def load_insiders():
    """(breadth signal, insider trades in our stocks, metadata, None), or (None, None, None, reason) when
    the insider files are missing: the app then uses the equilibrium returns (no view)."""
    try:
        signal = ins.market_signal(data.load_insider_market())
        return signal, data.load_insider_trades(), data.load_insider_metadata(), None
    except data.DataError as exc:
        return None, None, None, str(exc)


def with_views(e: est.Estimates, source: str, confidence: float) -> est.Estimates:
    """Black-Litterman expected returns: the insider house view, or the naive past-performance view for the
    backtest comparison (at 0% both give the equilibrium), known at the window's last month."""
    if source == INSIDER:
        return ins.with_insider_view(e, e.end, load_insiders()[0], confidence)[0]
    return est.with_view_confidence(e, confidence)


def build_constraints(profile: str, spec: dict) -> list[opt.Constraint]:
    cons = [c for c in prof.default_constraints(profile) if c.key in spec]
    for c in cons:
        c.params, c.active = spec[c.key]["params"], spec[c.key]["active"]
    return cons


@st.cache_data(show_spinner="Optimising the portfolio...")
def optimise(profile: str, cons_json: str, source: str, confidence: float):
    e = with_views(get_estimates(), source, confidence)
    d = prof.problem_data(profile, e, load_data()[0])
    res = opt.solve(d, build_constraints(profile, json.loads(cons_json)))
    eq = prof.SPECS[profile].benchmark_equity
    te = bm.tracking_error(res.weights, eq, e) if res.ok else np.nan
    return res, bm.ex_ante_stats(eq, e), bm.ex_ante_stats(1.0, e), te


@st.cache_data(show_spinner=False)
def backtest_estimates() -> dict:
    """Estimates at every quarter-end of the backtest (semi-annual/annual dates are a subset)."""
    universe, returns, rf, bench = load_data()
    dates = btm.rebalance_dates(returns.index, btm.DEFAULT_START, "Quarterly")
    return btm.estimates_by_date(returns, rf, est.shrinkage_groups(universe), dates, benchmarks=bench)


def backtest_views(source: str, confidence: float) -> dict:
    """Estimates at every rebalancing with the given view (each date uses only data known then)."""
    if source == INSIDER:
        return btm.with_insider_view(backtest_estimates(), load_insiders()[0], confidence)
    return btm.with_view_confidence(backtest_estimates(), confidence)


@st.cache_data(show_spinner="Running the backtest (re-optimising at every rebalancing)...")
def run_backtest(profile: str, cons_json: str, freq: str, source: str, confidence: float):
    universe, returns, rf, bench = load_data()
    res = btm.run_backtest(returns, rf, universe, build_constraints(profile, json.loads(cons_json)),
                           freq=freq, estimates=backtest_views(source, confidence))
    ew = btm.run_equal_weight(returns, universe, prof.SPECS[profile].min_bonds, freq=freq)
    idx = res.returns.index
    series = pd.DataFrame({"Strategy": res.returns, EW_LABEL: ew.returns.reindex(idx),
                           bench_label(profile): bm.profile_benchmark_returns(prof.SPECS[profile].benchmark_equity,
                                                                             bench).reindex(idx),
                           "SPY": bench["SPY"].reindex(idx)})
    return res, ew, series


@st.cache_data(show_spinner="Computing the efficient frontier...")
def frontiers(profile: str, cons_json: str, source: str, confidence: float):
    """Efficient frontier with the client's constraints and without them (long-only only), and the
    maximum-Sharpe (tangency) portfolio under the client's constraints."""
    e = with_views(get_estimates(), source, confidence)
    d = prof.problem_data(profile, e, load_data()[0])
    top = max(0.25, 1.6 * prof.SPECS[profile].max_volatility)
    cons = build_constraints(profile, json.loads(cons_json))
    front = fr.efficient_frontier(d, cons, max_vol=top)
    return front, fr.efficient_frontier(d, [], max_vol=top), fr.max_sharpe_portfolio(d, cons, front)


@st.cache_data(show_spinner="Simulating 5,000 wealth paths...")
def simulate(profile: str, cons_json: str, freq: str, source: str, confidence: float, years: int, wealth: float,
             loss_limit: float):
    """Block bootstrap of the strategy's OUT-OF-SAMPLE backtest returns (at every date the portfolio
    was chosen with past data only, net of trading costs) and of the profile benchmark on the same
    months. Replaying today's weights on the months used to choose them would be in-sample (look-ahead)."""
    _, series = run_backtest(profile, cons_json, freq, source, confidence)[1:]
    hist = series[["Strategy", bench_label(profile)]].dropna()
    hist.columns = ["Portfolio", "Benchmark"]
    sim = risk.wealth_simulation(hist, wealth, years, year_loss_limit=loss_limit)
    return sim, hist.index[0], hist.index[-1], len(hist)


def pct_slider(label, default, lo, hi, step, key, disabled=False):
    """Percent slider. While it shows the (rounded) default, the exact default is returned, so
    e.g. the 1/3 emerging-markets limit is not silently turned into 33%."""
    fmt = "%.1f%%" if step < 1 else "%d%%"
    cast = float if step < 1 else int
    shown = cast(round(default * 100, 1))
    value = st.slider(label, cast(lo), cast(hi), shown, cast(step), format=fmt, key=key, disabled=disabled)
    return default if value == shown else value / 100


def require_ok(res: opt.OptimizationResult) -> bool:
    if not res.ok:
        st.error(res.message)
        st.info("Open the **Adjust constraints** panel above to loosen the conflicting rules.")
    return res.ok


def grouped_bars(table: pd.DataFrame, title: str, colors=(NAVY, GREY)):
    f = go.Figure([go.Bar(name=c, x=table.index, y=table[c] * 100, marker_color=clr)
                   for c, clr in zip(table.columns, colors)])
    f.update_layout(barmode="group", yaxis_title="% of portfolio", title=title)
    return f


def avatar_html(member: team.Member, size: int = 112) -> str:
    """Round photo, or a round avatar with the initials when the photo is missing."""
    uri = team.photo_data_uri(member)
    style = f"width:{size}px;height:{size}px;border-radius:50%;margin:0.25rem auto 0.75rem auto;display:block;"
    if uri:
        return f'<img src="{uri}" alt="{html.escape(member.name)}" style="{style}object-fit:cover;">'
    return (f'<div style="{style}background:{NAVY};color:#ffffff;display:flex;align-items:center;'
            f'justify-content:center;font-size:{size // 3}px;font-weight:600;letter-spacing:1px;">'
            f'{html.escape(team.initials(member.name))}</div>')


# --- data & header ---------------------------------------------------------------------------

try:
    universe, returns, rf, bench = load_data()
except data.DataError as exc:
    st.error(f"Market data unavailable: {exc}")
    st.stop()

names = universe["name"].to_dict()
label = lambda t: f"{names.get(t, t)} ({t})"  # noqa: E731

answers = {q.key: st.session_state.get(f"q.{q.key}", 1) for q in prof.QUESTIONS}
assessment = prof.assess(answers)

if "open_q" not in st.session_state:
    st.session_state["open_q"] = prof.QUESTIONS[0].key

# Drop stale selectbox state: an earlier run may have left a display label (e.g.
# "From questionnaire (Conservative)") in st.session_state["profile"], which is no longer a
# valid option and would crash prof.SPECS lookups downstream.
if st.session_state.get("profile") not in ("auto", *prof.PROFILES):
    st.session_state.pop("profile", None)

# Accordion renders only one question's radio at a time; Streamlit garbage-collects widget state
# when the widget isn't rendered this run, which would reset previous answers. Re-asserting the
# value here prevents the GC and keeps the Resulting profile stable across accordion switches.
# Defaults are set here (not with index= on the radio): a widget with both a default and a value set
# through the Session State API makes Streamlit log a warning.
for _q in prof.QUESTIONS:
    _k = f"q.{_q.key}"
    st.session_state[_k] = st.session_state.get(_k, 1)


def _toggle_question(key: str) -> None:
    st.session_state["open_q"] = None if st.session_state.get("open_q") == key else key


_tv = {
    k_: THEME[k_] for k_ in ("bg", "surface", "surface_2", "border", "text", "text_dim", "accent", "accent_hover",
                             "accent_soft", "hover", "shadow", "dock_bg", "dock_border", "dock_shadow")
}

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: {_tv['bg']}; --surface: {_tv['surface']}; --surface-2: {_tv['surface_2']};
  --border: {_tv['border']}; --text: {_tv['text']}; --text-dim: {_tv['text_dim']};
  --accent: {_tv['accent']}; --accent-hover: {_tv['accent_hover']};
  --shadow: {_tv['shadow']}; --dock-bg: {_tv['dock_bg']}; --dock-border: {_tv['dock_border']};
  --dock-shadow: {_tv['dock_shadow']}; --accent-soft: {_tv['accent_soft']}; --hover: {_tv['hover']};
  --font-sans: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
  --font-serif: 'IBM Plex Serif', Georgia, serif;
}}
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, select, textarea {{
  font-family: var(--font-sans) !important;
}}
h1, h2, h3 {{ font-family: var(--font-serif) !important; letter-spacing: -0.01em; font-weight: 500 !important; }}
.stApp h1 {{ color: var(--text) !important; }}
.stApp, html, body {{ background: var(--bg); color: var(--text); }}
/* Hide Streamlit's native header (Deploy button + hamburger) — overlaps our top controls */
[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stCaption, .stText {{ color: var(--text); }}
[data-testid="stCaptionContainer"] p, small, .stCaption {{ color: var(--text-dim) !important; }}
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {{ color: var(--text); }}
/* Containers / borders */
div[data-testid="stExpander"], div[data-testid="stAlert"], div[data-testid="stNotification"],
div[data-testid="stDataFrame"], .stTabs [data-baseweb="tab-panel"] > div > div > div[data-testid*="stVertical"] > div > div > div[data-testid="stContainer"] {{
  border-color: var(--border) !important;
}}
div[data-testid="stContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {{
  background: var(--surface);
}}
/* Inputs */
[data-baseweb="input"] input, [data-baseweb="select"] div, textarea {{
  background: var(--surface-2) !important; color: var(--text) !important;
}}
/* Primary buttons */
button[kind="primary"] {{ background: var(--accent) !important; color: #fff !important; border: none !important; }}
button[kind="primary"]:hover {{ background: var(--accent-hover) !important; }}
/* Tighten top padding (Streamlit leaves ~6rem by default) and reserve space for the bottom nav */
.block-container {{ padding-top: 1.5rem !important; padding-bottom: 120px !important; }}
h1 {{ margin-top: 0 !important; padding-top: 0 !important; }}
/* Main nav (Client Profile / Portfolio / ...): floating iOS-style pill, fixed bottom */
.st-key-main-nav [role="tablist"] {{
  position: fixed !important; bottom: 24px !important; left: 50% !important;
  transform: translateX(-50%) !important; width: auto !important; max-width: 92vw !important;
  background: var(--dock-bg) !important; backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--dock-border) !important; border-radius: 999px !important;
  padding: 10px 14px !important;
  box-shadow: var(--dock-shadow) !important;
  z-index: 1000 !important; gap: 8px !important; justify-content: center !important;
  display: flex !important;
}}
.st-key-main-nav [role="tab"] {{
  background: transparent !important; color: var(--text) !important;
  border-radius: 999px !important; padding: 14px 36px !important;
  font-weight: 500 !important; font-size: 1.2rem !important; line-height: 1.2 !important;
  border: none !important; transition: background 0.15s, color 0.15s, transform 0.15s;
  min-width: 120px !important; text-align: center !important;
}}
.st-key-main-nav [role="tab"]:hover {{ background: var(--hover) !important; transform: translateY(-1px); }}
.st-key-main-nav [role="tab"][aria-selected="true"] {{
  background: var(--accent) !important; color: #ffffff !important; font-weight: 600 !important;
  box-shadow: 0 4px 14px var(--accent-soft) !important;
}}
/* Sub-tabs inside Portfolio: accent-bordered pill group */
.st-key-portfolio-subtabs [role="tablist"] {{
  position: static !important; transform: none !important;
  background: var(--surface) !important;
  backdrop-filter: none !important; -webkit-backdrop-filter: none !important;
  border: 2px solid var(--accent) !important; border-radius: 10px !important;
  box-shadow: 0 0 0 1px var(--accent-soft) !important;
  padding: 5px !important; width: fit-content !important; max-width: none !important;
  gap: 2px !important; justify-content: flex-start !important;
  margin: 0.5rem 0 1.5rem !important;
  z-index: auto !important;
}}
.st-key-portfolio-subtabs [role="tab"] {{
  padding: 8px 22px !important; border-radius: 6px !important;
  font-size: 0.95rem !important; font-weight: 500 !important;
  color: var(--text-dim) !important; background: transparent !important;
  min-width: unset !important;
}}
.st-key-portfolio-subtabs [role="tab"]:hover {{
  background: var(--accent-soft) !important; color: var(--text) !important;
}}
.st-key-portfolio-subtabs [role="tab"][aria-selected="true"] {{
  background: var(--accent) !important; color: #ffffff !important;
  border-bottom: none !important; border-radius: 6px !important;
  box-shadow: none !important; font-weight: 600 !important;
}}
/* Selected tab label stays white on the accent background (the global p colour rule would win otherwise) */
.st-key-main-nav [role="tab"][aria-selected="true"] *, .st-key-portfolio-subtabs [role="tab"][aria-selected="true"] * {{
  color: #ffffff !important;
}}
/* Kill the react-aria selection indicator (thin blue bar under the active tab) */
.react-aria-SelectionIndicator {{ display: none !important; }}
/* Backtest "Rebalancing" selectbox — accent colour to match the sub-tabs */
.st-key-bt-rebalance label p {{ color: var(--accent) !important; font-weight: 600 !important; }}
.st-key-bt-rebalance [data-baseweb="select"] > div {{
  background: var(--surface) !important;
  border: 2px solid var(--accent) !important; border-radius: 10px !important;
  box-shadow: 0 0 0 1px var(--accent-soft) !important;
}}
.st-key-bt-rebalance [data-baseweb="select"] > div:hover {{
  border-color: var(--accent-hover) !important;
}}
.st-key-bt-rebalance [data-baseweb="select"] input,
.st-key-bt-rebalance [data-baseweb="select"] div[role="combobox"] {{
  color: var(--text) !important; font-weight: 500 !important;
}}
/* Top-right About Us button */
.st-key-top-controls {{
  position: fixed !important; top: 14px !important; right: 20px !important;
  z-index: 1001 !important; width: auto !important;
}}
.st-key-top-controls button {{
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 999px !important;
  padding: 4px 18px !important; font-size: 0.85rem !important;
  white-space: nowrap !important; overflow: visible !important;
}}
.st-key-top-controls button:hover {{ background: var(--surface-2) !important; }}
/* Hide the floating nav while the About Us dialog is open (it sat on top of the dialog) */
body:has([role="dialog"]) .st-key-main-nav [role="tablist"] {{ display: none !important; }}
/* Medium screens (small laptops, split windows): six nav buttons in one row */
@media (max-width: 1200px) {{
  .st-key-main-nav [role="tab"] {{ padding: 10px 14px !important; min-width: 0 !important; font-size: 1.05rem !important; }}
}}
/* Narrow screens (phones, small windows): all six nav buttons must fit */
@media (max-width: 820px) {{
  .st-key-main-nav [role="tablist"] {{ bottom: 12px !important; padding: 6px !important; gap: 2px !important;
    width: calc(100vw - 24px) !important; max-width: none !important; justify-content: space-between !important; }}
  .st-key-main-nav [role="tab"] {{ min-width: 0 !important; flex: 1 1 0 !important; padding: 8px 2px !important;
    font-size: 0.74rem !important; line-height: 1.15 !important; white-space: normal !important; }}
  .st-key-portfolio-subtabs [role="tablist"] {{ width: auto !important; flex-wrap: wrap !important; }}
  .st-key-portfolio-subtabs [role="tab"] {{ flex: 0 0 auto !important; padding: 6px 12px !important; font-size: 0.85rem !important; }}
  .st-key-top-controls {{ top: 8px !important; right: 8px !important; }}
  .block-container {{ padding-bottom: 96px !important; }}
}}
</style>
""", unsafe_allow_html=True)


@st.dialog("About us", width="large")
def _about_dialog():
    st.markdown(
        "We play the role of an **international private bank** advising clients whose reference currency is "
        "the US dollar. The app turns a short suitability questionnaire into a risk profile, then builds the "
        "portfolio with the highest expected return whose estimated volatility is within the client's budget and "
        "investment constraints. Every recommendation is backtested and its future range of outcomes simulated, "
        "so the client can see both the potential and the risks. It was built for the course *Quantitative "
        "Asset and Risk Management II* at HEC Lausanne.")
    members, repository, team_error = team.load_team()
    if team_error:
        st.warning(team_error)
    st.markdown("#### The team")
    for start in range(0, len(members), 3):
        cols = st.columns(3, gap="medium")
        for col, m in zip(cols, members[start:start + 3]):
            with col.container(border=True):
                st.markdown(avatar_html(m), unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;font-weight:600;font-size:1.1rem;">{html.escape(m.name)}</div>'
                            f'<div style="text-align:center;color:{NAVY};font-size:0.9rem;margin-bottom:0.5rem;">'
                            f'{html.escape(m.role)}</div>'
                            f'<div style="text-align:center;color:var(--text-dim);font-size:0.9rem;">{html.escape(m.description)}</div>',
                            unsafe_allow_html=True)
                if m.linkedin:
                    st.markdown(f'<div style="text-align:center;margin-top:0.5rem;"><a href="{html.escape(m.linkedin)}" '
                                f'target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
    if repository:
        st.markdown(f"Source code: [{repository}]({repository})")


with st.container(key="top-controls"):
    if st.button("About Us", key="btn-about", use_container_width=True):
        _about_dialog()



def page_header() -> None:
    """Title of every tab except Home, whose blue ribbon carries the name."""
    st.title("Private Client Portfolio Optimizer")
    st.caption("The highest expected return within the client's volatility budget. All figures in USD.")


with st.container(key="main-nav"):
    tabs = st.tabs(["Home", "Client Profile", "Portfolio", "Insider Signal", "Backtest", "Simulation"])

# --- Home ------------------------------------------------------------------------------------

HOME_STEPS = [
    ("1", "Client Profile", "Answer seven questions. They set your risk profile, your volatility budget and "
                            "the minimum share in bonds."),
    ("2", "Portfolio", "The portfolio with the highest expected return within your budget, the efficient "
                       "frontier and where the risk comes from. Every constraint can be changed."),
    ("3", "Insider Signal", "Our house view: whether company insiders are buying their own shares, and how "
                            "that tilts the portfolio between equities and bonds."),
    ("4", "Backtest", "How the strategy would have done since 2012, against SPY, the profile benchmark and "
                      "an equal-weight portfolio."),
    ("5", "Simulation", "The range of outcomes for your wealth over 5 to 20 years, and the risk along the way."),
]

HOME_CHIPS = ["Suitability profile in 7 questions", "Black-Litterman with insider trading",
              "Backtest since 2012", "5,000 wealth scenarios"]

# Home page styles: a deep-blue ribbon (fixed colours, readable on every theme) and cards that use
# the theme's CSS variables. HTML blocks have no blank lines or indentation (Markdown would break them).
HOME_CSS = """<style>
.home-hero { position: relative; overflow: hidden; min-height: 58vh; display: flex; flex-direction: column;
  justify-content: center; padding: 3.2rem 3.6rem; margin: 0.4rem 0 2.2rem; border-radius: 24px;
  background: linear-gradient(135deg, #0f2f52 0%, #1f4e79 48%, #3a78b5 100%);
  box-shadow: 0 18px 50px rgba(15, 47, 82, 0.32); }
.home-hero::before, .home-hero::after { content: ""; position: absolute; border-radius: 50%; pointer-events: none; }
.home-hero::before { width: 560px; height: 560px; right: -150px; top: -200px;
  background: radial-gradient(circle, rgba(255,255,255,0.17), rgba(255,255,255,0) 70%); }
.home-hero::after { width: 420px; height: 420px; left: -140px; bottom: -230px;
  background: radial-gradient(circle, rgba(255,255,255,0.10), rgba(255,255,255,0) 70%); }
.home-kicker { position: relative; text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.85rem;
  font-weight: 600; color: rgba(255,255,255,0.78) !important; }
.home-welcome { position: relative; font-family: var(--font-serif); font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: 400; color: rgba(255,255,255,0.88) !important; margin-top: 1.2rem; }
.home-title { position: relative; font-family: var(--font-serif); font-size: clamp(2.8rem, 6.2vw, 5.4rem);
  line-height: 1.05; font-weight: 500; letter-spacing: -0.02em; color: #ffffff !important; margin: 0.2rem 0 1.2rem;
  max-width: 1100px; }
.home-sub { position: relative; max-width: 780px; font-size: clamp(1.05rem, 1.6vw, 1.35rem); line-height: 1.55;
  color: rgba(255,255,255,0.92) !important; }
.home-chips { position: relative; display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.9rem; }
.home-chips span { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.30); color: #ffffff;
  border-radius: 999px; padding: 0.42rem 1rem; font-size: 0.92rem; }
.home-scroll { position: relative; margin-top: 2.6rem; font-size: 0.95rem; color: rgba(255,255,255,0.82) !important; }
.home-scroll b { display: inline-block; margin-left: 0.3rem; animation: home-bounce 1.8s ease-in-out infinite; }
@keyframes home-bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(6px); } }
.home-section { font-family: var(--font-serif); font-size: 1.6rem; color: var(--text); margin: 0.4rem 0 0.9rem; }
.home-steps { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; margin-bottom: 1.4rem; }
.home-step { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 1.25rem 1.15rem; box-shadow: var(--shadow); }
.home-step .num { width: 2.3rem; height: 2.3rem; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; background: var(--accent); color: #ffffff; font-weight: 600; margin-bottom: 0.85rem; }
.home-step .name { font-weight: 600; font-size: 1.05rem; color: var(--text); margin-bottom: 0.35rem; }
.home-step .text { font-size: 0.92rem; line-height: 1.45; color: var(--text-dim); }
.st-key-home-stats [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--font-serif); }
@media (max-width: 1000px) { .home-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .home-hero { padding: 2.4rem 1.8rem; min-height: 50vh; } }
@media (max-width: 600px) { .home-steps { grid-template-columns: 1fr; } }
</style>"""

with tabs[0], guarded("Home"):
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    chips = "".join(f"<span>{html.escape(c)}</span>" for c in HOME_CHIPS)
    st.markdown(
        '<div class="home-hero">'
        '<div class="home-kicker">International private bank · US dollar clients</div>'
        '<div class="home-welcome">Welcome to the</div>'
        '<div class="home-title">Private Client Portfolio Optimizer</div>'
        '<div class="home-sub">From seven questions, we build the portfolio with the highest expected return that '
        "keeps its estimated volatility within your budget, tilted by our house view: what company insiders do with their own "
        "money.</div>"
        f'<div class="home-chips">{chips}</div>'
        '<div class="home-scroll">Scroll down for today\'s portfolio and the five steps<b>↓</b></div>'
        "</div>", unsafe_allow_html=True)
    home_live = st.container()   # "Today at a glance": filled once the portfolio is computed (see below)
    steps = "".join(
        f'<div class="home-step"><div class="num">{num}</div><div class="name">{html.escape(tab)}</div>'
        f'<div class="text">{html.escape(text)}</div></div>' for num, tab, text in HOME_STEPS)
    st.markdown(f'<div class="home-section">How to use the app</div><div class="home-steps">{steps}</div>',
                unsafe_allow_html=True)
    st.caption("Move between the steps with the bar at the bottom of the screen. The team is presented under "
               "\"About Us\" (top right). Figures are estimates in USD, not a guarantee of future returns.")

# --- Client Profile --------------------------------------------------------------------------

with tabs[1], guarded("Client Profile"):
    page_header()
    with st.expander("Client settings", expanded=True):
        signal, insider_trades, insider_meta, insider_error = load_insiders()
        s1, s2, s3 = st.columns(3)
        with s1:
            wealth = st.number_input("Client wealth (USD)", min_value=100_000, max_value=1_000_000_000,
                                     value=5_000_000, step=250_000, format="%d")
        with s2:
            choice = st.selectbox("Risk profile", ["auto"] + prof.PROFILES, key="profile",
                                  format_func=lambda p: f"From questionnaire ({assessment.profile})" if p == "auto" else p)
        with s3:
            # One house view: insider trading (Black-Litterman). Without the insider files the app falls back to
            # the equilibrium returns (no view), which with_views computes as the past-performance view at 0%.
            if signal is not None:
                view_source = INSIDER
                confidence = st.slider(
                    "Confidence in the insider view", 0, int(ins.MAX_CONFIDENCE * 100),
                    int(ins.DEFAULT_CONFIDENCE * 100), 5, format="%d%%", key="views_insider",
                    help="Expected returns start from the market equilibrium (each asset's return reflects only its "
                         "risk relative to a 60% SPY / 40% AGG market portfolio) and are tilted by our house view: "
                         "when officers and directors across US companies buy their own shares more than usual, "
                         "equities are expected to beat bonds by more (see the Insider Signal tab). At 50% the "
                         "expected premium of equities over bonds moves halfway from the equilibrium to the insider "
                         "view. 0% = pure equilibrium. Higher = larger tilts, with more trading.") / 100
            else:
                view_source, confidence = PAST, 0.0
                st.slider("Confidence in the insider view", 0, 90, 0, disabled=True, key="views_insider_off",
                          format="%d%%", help="The insider data is not available: the portfolio uses the market "
                                              "equilibrium returns, without a view.")
        insider_note = (f" Insider trades: SEC Form 4 filings up to {pd.Timestamp(insider_meta['last_filing']):%B %Y}."
                        if insider_meta else f" Insider view unavailable: {insider_error}")
        st.caption(f"Data: monthly total returns in USD up to {returns.index[-1]:%B %Y} (yfinance). "
                   f"Estimates use the last {WINDOW} months. Risk-free: 3-month T-bill, currently "
                   f"{rf.iloc[-1] * 12:.2%} p.a.{insider_note}")

    profile = assessment.profile if choice == "auto" else choice
    spec_p = prof.SPECS[profile]
    if choice != "auto" and profile != assessment.profile:
        st.warning(f"The manual override uses **{profile}** instead of the questionnaire's {assessment.profile}.")

    st.subheader("Suitability questionnaire")
    left, right = st.columns([3, 2], gap="large")
    with left:
        for i, q in enumerate(prof.QUESTIONS, 1):
            is_open = st.session_state.get("open_q") == q.key
            st.button(f"**Q{i}**  {q.text}", key=f"btn.{q.key}",
                      use_container_width=True, on_click=_toggle_question, args=(q.key,))
            if is_open:
                with st.container(border=True):
                    st.radio(q.text, list(range(len(q.options))),
                             format_func=lambda i, q=q: q.options[i],
                             key=f"q.{q.key}", label_visibility="collapsed")
    with right:
        a = prof.assess({q.key: st.session_state.get(f"q.{q.key}", 1) for q in prof.QUESTIONS})
        sp = prof.SPECS[a.profile]
        with st.container(border=True):
            st.markdown("#### Resulting profile")
            st.markdown(f"## {a.profile}")
            st.caption(sp.description)
            c1, c2, c3 = st.columns(3)
            c1.metric("Capacity (1-4)", f"{a.dimensions['capacity']:.2f}", a.levels["capacity"], delta_color="off")
            c2.metric("Willingness (1-4)", f"{a.dimensions['willingness']:.2f}", a.levels["willingness"],
                      delta_color="off")
            k_cap = prof.KNOWLEDGE_CAP.get(round(a.dimensions["knowledge"]))
            c3.metric("Knowledge (1-4)", f"{a.dimensions['knowledge']:.2f}",
                      f"at most {k_cap}" if k_cap else "no limit", delta_color="off",
                      help="One question, so a whole score. It does not set a level: it caps the profile "
                           "(no knowledge: at most Balanced; basic: at most Growth).")
            st.markdown("**Why this profile**")
            for r in a.reasons:
                st.markdown(f"- {r}")
            st.markdown(f"**What it means:** estimated volatility at most {sp.max_volatility:.0%}, "
                        f"at least {sp.min_bonds:.0%} in bonds, compared with a "
                        f"{sp.benchmark_equity:.0%} SPY / {1 - sp.benchmark_equity:.0%} AGG benchmark.")
            st.caption("The volatility budget applies to the estimate before investing; realised volatility may be higher.")

    with st.expander("All profiles & scoring methodology", expanded=False):
        profiles_table = pd.DataFrame({p: {"Max volatility": f"{s.max_volatility:.0%}",
                                           "Min bonds": f"{s.min_bonds:.0%}",
                                           "Benchmark": f"{s.benchmark_equity:.0%} SPY / {1 - s.benchmark_equity:.0%} AGG"}
                                       for p, s in prof.SPECS.items()}).T
        st.dataframe(profiles_table, width="stretch")
        st.caption("Each answer scores 1-4. Capacity and willingness averages map to a profile "
                   "(<1.75 Conservative, <2.50 Balanced, <3.25 Growth, otherwise Dynamic); the lower one is "
                   "kept, then capped by knowledge. A horizon below 3 years always gives Conservative.")

# --- Portfolio (constraints live in an expander inside this tab) -----------------------------

defaults = {c.key: c for c in prof.default_constraints(profile)}
cons_spec: dict = {}
stock_tickers = [t for t in universe.index if universe.at[t, "vehicle"] == "Stock"]
sectors = sorted(universe.loc[universe["asset_class"] == "Equity", "sector"].unique())

with tabs[2], guarded("Portfolio"):
    page_header()
    # Constraints first, so "what if" changes are one click away during the pitch; the status line
    # (binding constraints, or the conflict) sits right under them, above the results.
    with st.expander(f"Adjust constraints ({profile} defaults)", expanded=False):
        k = lambda name: f"{profile}.{name}"  # noqa: E731  (each profile keeps its own settings)

        def card(key, column, render, mandatory=False):
            c = defaults[key]
            with column.container(border=True):
                if mandatory:
                    st.markdown(f"**{c.name}** (volatility budget of the profile)")
                    active = True
                else:
                    active = st.toggle(c.name, value=True, key=k(f"{key}.on"))
                params = render(not active)
                cons_spec[key] = {"active": active, "params": params}
                st.caption(c.describe(params))

        st.markdown("##### Volatility budget and asset mix")
        col_a, col_b = st.columns(2, gap="large")
        card("max_volatility", col_a, lambda off: {"limit": pct_slider(
            "Max estimated volatility", spec_p.max_volatility, 2, 25, 0.5, k("vol"), off)}, mandatory=True)
        card("min_bonds", col_a, lambda off: {"min_share": pct_slider(
            "Min share in bond funds", spec_p.min_bonds, 0, 100, 5, k("bonds"), off)})
        card("max_cash", col_b, lambda off: {"limit": pct_slider(
            "Max cash & equivalents", prof.MAX_CASH, 0, 100, 5, k("cash"), off)})

        st.markdown("##### Bond sleeve")
        col_a, col_b = st.columns(2, gap="large")
        card("max_bond_etf_weight", col_a, lambda off: {"limit": pct_slider(
            "Max weight per bond ETF", prof.MAX_BOND_ETF_WEIGHT, 5, 100, 5, k("bondetf"), off)})
        card("max_bond_duration", col_a, lambda off: {"limit": st.slider(
            "Max average duration of the bonds (years)", 0.5, 15.0, float(spec_p.max_bond_duration), 0.5,
            key=k("duration"), disabled=off)})
        card("min_investment_grade", col_b, lambda off: {"min_share": pct_slider(
            "Min investment grade (share of bonds)", spec_p.min_investment_grade, 0, 100, 5, k("ig"), off)})
        card("max_high_yield", col_b, lambda off: {"limit": pct_slider(
            "Max high yield incl. EM bonds (share of portfolio)", spec_p.max_high_yield, 0, 100, 1, k("hy"), off)})

        st.markdown("##### Equities")
        col_a, col_b = st.columns(2, gap="large")
        card("max_stock_weight", col_a, lambda off: {"limit": pct_slider(
            "Max weight per stock", prof.MAX_STOCK_WEIGHT, 1, 30, 1, k("stock"), off)})
        card("max_etf_weight", col_a, lambda off: {"limit": pct_slider(
            "Max weight per equity / gold ETF", prof.MAX_ETF_WEIGHT, 5, 100, 5, k("etf"), off)})
        card("max_sector_weight", col_b, lambda off: {"limit": pct_slider(
            "Max weight per equity sector", prof.MAX_SECTOR_WEIGHT, 5, 100, 5, k("sector"), off)})
        card("em_equity_cap", col_b, lambda off: {"ratio": pct_slider(
            "Max emerging markets (share of equities)", prof.EM_EQUITY_CAP, 0, 100, 1, k("em"), off)})

        st.markdown("##### Exclusions")
        card("exclusions", st, lambda off: {
            "sectors": st.multiselect("Excluded sectors", sectors, default=[], key=k("excl.sectors"), disabled=off),
            "tickers": st.multiselect("Excluded companies", stock_tickers, default=[], format_func=label,
                                      key=k("excl.tickers"), disabled=off)})

        st.caption("Emerging markets = the EEM fund plus PDD (China) and MELI (Latin America). International "
                   "equity funds (VGK, EWJ, EEM) count as one sector. Bond funds and gold are not subject to "
                   "the sector limit. Bond durations and ratings are the funds' current fact-sheet values. "
                   "High yield = HYG and the emerging-market bond fund EMB.")
    status_box = st.container()
    # Sub-tabs wrapped in a keyed container so our CSS can target them unambiguously.
    with st.container(key="portfolio-subtabs"):
        sub_overview, sub_alloc, sub_bonds, sub_holdings = st.tabs(
            ["Overview", "Allocation", "Bonds", "Holdings"])

    cons_json = json.dumps(cons_spec, sort_keys=True)
    try:
        res, bench_stats, spy_stats, te = optimise(profile, cons_json, view_source, confidence)
    except EXPECTED_ERRORS as exc:
        st.error(f"The portfolio could not be computed: {exc}")
        st.stop()
    except Exception:
        log.exception("Unexpected error in the optimisation")
        st.error("The portfolio could not be computed because of an unexpected error. Please change the inputs "
                 "and try again.")
        st.stop()

    with status_box:
        if res.ok:
            binding = res.duals.loc[res.duals["active"], "constraint"].unique().tolist()
            st.success("All active constraints are satisfied. Binding: " + (", ".join(binding) or "none") + ".")
        else:
            st.error(res.message)

    if require_ok(res):
        w = res.weights
        uni = universe.loc[w.index]
        is_bond = uni["asset_class"] == "Bond"
        is_eq = uni["asset_class"] == "Equity"

        with sub_overview:
            c = st.columns(4)
            c[0].metric("Expected return above T-bill", pct(res.expected_return, 2),
                        f"{(res.expected_return - bench_stats['expected_return']) * 1e4:+.0f} bps vs benchmark")
            c[1].metric("Estimated volatility", pct(res.volatility, 2),
                        f"{(res.volatility - bench_stats['volatility']) * 100:+.2f} pp vs benchmark", delta_color="inverse")
            c[2].metric("Sharpe ratio", f"{res.sharpe:.2f}", f"{res.sharpe - bench_stats['sharpe']:+.2f} vs benchmark")
            c[3].metric("Tracking error vs benchmark", pct(te, 2))
            comp = pd.DataFrame({"Portfolio": [res.expected_return, res.volatility, res.sharpe],
                                 bench_label(profile): [bench_stats["expected_return"], bench_stats["volatility"], bench_stats["sharpe"]],
                                 "SPY": [spy_stats["expected_return"], spy_stats["volatility"], spy_stats["sharpe"]]},
                                index=["Expected return above T-bill", "Estimated volatility", "Sharpe ratio"])
            st.dataframe(comp.style.format("{:.2%}", subset=pd.IndexSlice[["Expected return above T-bill", "Estimated volatility"], :])
                         .format("{:.2f}", subset=pd.IndexSlice[["Sharpe ratio"], :]), width="stretch")
            view_text = ("the insider view (equities vs bonds, from the breadth of insider buying across US "
                         "companies; see the Insider Signal tab)" if view_source == INSIDER
                         else "no view (the insider data is not available)")
            st.caption(f"Ex-ante figures, estimated on the last {WINDOW} months. Expected returns: Black-Litterman, i.e. "
                       "market-equilibrium returns (implied by a 60% SPY / 40% AGG market portfolio) combined with "
                       f"{view_text}, at {confidence:.0%} confidence. Risk: Ledoit-Wolf "
                       "covariance. The portfolio and its benchmarks are estimated with the same method; see the "
                       "Backtest tab for what this strategy actually delivered.")

            st.markdown("### Efficient frontier")
            try:
                front, free, tangency = frontiers(profile, cons_json, view_source, confidence)
            except EXPECTED_ERRORS as exc:
                st.error(f"The efficient frontier could not be computed: {exc}")
                front = None
            if front is not None:
                cap = cons_spec["max_volatility"]["params"]["limit"]
                free_at_cap = float(np.interp(res.volatility, free["volatility"], free["expected_return"]))
                cost_bps = max(free_at_cap - res.expected_return, 0.0) * 1e4
                gmv = front.iloc[0]  # the curve starts at the minimum-variance portfolio
                # capital market line from the T-bill (origin) through the tangency point, kept inside the plotted range
                top_ret = max(front["expected_return"].max(), free["expected_return"].max())
                cml_x = np.array([0.0, min(front["volatility"].max(), top_ret / max(tangency["sharpe"], 1e-9))])
                f1, f2 = st.columns([3, 2], gap="large")
                f = go.Figure([
                    go.Scatter(x=free["volatility"] * 100, y=free["expected_return"] * 100, name="Without client constraints",
                               line=dict(color=GREY, width=2, dash="dash")),
                    go.Scatter(x=front["volatility"] * 100, y=front["expected_return"] * 100,
                               name="With your constraints", line=dict(color=NAVY, width=3)),
                    go.Scatter(x=cml_x * 100, y=cml_x * tangency["sharpe"] * 100, name="Capital market line",
                               line=dict(color=GOLD_COLOR, width=1.5, dash="dot")),
                    go.Scatter(x=[gmv["volatility"] * 100], y=[gmv["expected_return"] * 100], name="Minimum variance",
                               mode="markers", marker=dict(color="white", size=11, symbol="circle",
                                                           line=dict(color=NAVY, width=2))),
                    go.Scatter(x=[tangency["volatility"] * 100], y=[tangency["expected_return"] * 100],
                               name="Maximum Sharpe (tangency)", mode="markers",
                               marker=dict(color=GOLD_COLOR, size=13, symbol="triangle-up", line=dict(color="white", width=1))),
                    go.Scatter(x=[bench_stats["volatility"] * 100], y=[bench_stats["expected_return"] * 100],
                               name=bench_label(profile), mode="markers",
                               marker=dict(color=GREY, size=13, symbol="diamond", line=dict(color="white", width=1))),
                    go.Scatter(x=[spy_stats["volatility"] * 100], y=[spy_stats["expected_return"] * 100], name="SPY",
                               mode="markers", marker=dict(color=ACCENT, size=12, line=dict(color="white", width=1))),
                    go.Scatter(x=[res.volatility * 100], y=[res.expected_return * 100], name="Recommended portfolio",
                               mode="markers", marker=dict(color=NAVY, size=18, symbol="star", line=dict(color="white", width=1)))])
                f.add_vline(x=cap * 100, line=dict(color=NAVY, width=1, dash="dot"),
                            annotation_text=f"Volatility budget {cap:.0%}", annotation_position="top left")
                f.update_layout(title="Best expected return for each level of risk", xaxis_title="Estimated volatility (% p.a.)",
                                yaxis_title="Expected return above T-bill (% p.a.)")
                f1.plotly_chart(fig_layout(f, 430), width="stretch")
                mix = front.set_index("volatility")[fr.CLASS_ORDER].rename(columns=CLASS_LABELS) * 100
                g = go.Figure([go.Scatter(x=mix.index * 100, y=mix[c], name=c, stackgroup="one", mode="none", fillcolor=clr)
                               for c, clr in zip(mix.columns, CLASS_COLORS)])
                g.add_vline(x=cap * 100, line=dict(color="white", width=2, dash="dot"))
                g.update_layout(title="Allocation along the frontier", xaxis_title="Estimated volatility (% p.a.)",
                                yaxis_title="% of portfolio", yaxis_range=[0, 100])
                f2.plotly_chart(fig_layout(g, 430), width="stretch")
                st.caption(f"Each point of the curves is the portfolio with the highest expected return for that level of "
                           f"risk. The recommended portfolio (star) is the point at the volatility budget. At this risk, "
                           f"dropping all client constraints would add about {cost_bps:.0f} bps of expected return per "
                           "year: that is what diversification limits, the bond floor and the other rules cost on these "
                           "estimates, in exchange for a sturdier portfolio. Both curves are almost straight lines, and "
                           "the benchmark and SPY sit on them: our expected returns start from the market equilibrium, "
                           "in which a SPY / AGG mix is efficient by construction, so on these estimates no portfolio "
                           "of the universe beats its Sharpe ratio. The house view only tilts the mix between equities "
                           "and bonds; the Backtest tab shows what the strategy actually delivered.")
                gap = res.volatility - tangency["volatility"]
                if abs(gap) < 0.002:
                    where = (f"Your portfolio is practically the maximum-Sharpe portfolio (Sharpe ratio {res.sharpe:.2f}).")
                elif tangency["sharpe"] - res.sharpe < 0.005:
                    where = (f"Your portfolio has practically the same Sharpe ratio ({res.sharpe:.2f}): the curve is almost "
                             "straight, so more or less risk changes the expected return but hardly the trade-off.")
                elif gap > 0:
                    where = (f"Your portfolio sits to its right (Sharpe ratio {res.sharpe:.2f}): the extra expected return "
                             "is bought with proportionally more risk, the price of a higher return target without borrowing.")
                else:
                    where = (f"Your portfolio sits to its left (Sharpe ratio {res.sharpe:.2f}): the volatility budget is below "
                             "the risk of the maximum-Sharpe portfolio, so some of the best trade-off is given up for safety.")
                st.caption(f"Special points of your curve. The circle is the "
                           f"minimum-variance portfolio, the least risk your constraints allow ({pct(gmv['volatility'])} "
                           f"volatility). The triangle is the maximum-Sharpe (tangency) portfolio: Sharpe ratio "
                           f"{tangency['sharpe']:.2f} at {pct(tangency['volatility'])} volatility, with "
                           f"{pct(tangency['Equity'], 0)} in equities. The dotted capital market line starts at the T-bill "
                           f"(the origin, since returns are measured above it) and touches the curve there: its slope is "
                           f"the best Sharpe ratio available. {where}")

        with sub_alloc:
            b_eq = spec_p.benchmark_equity
            by_class = pd.DataFrame({"Portfolio": w.groupby(uni["asset_class"]).sum(),
                                     "Benchmark": pd.Series({"Equity": b_eq, "Bond": 1 - b_eq})}).reindex(CLASS_ORDER).fillna(0)
            by_class.index = by_class.index.map(CLASS_LABELS)
            by_sector = w[is_eq].groupby(uni.loc[is_eq, "sector"]).sum().sort_values(ascending=False)
            b1, b3 = st.columns(2)
            b1.plotly_chart(fig_layout(grouped_bars(by_class, "By asset class"), 340), width="stretch")
            f = go.Figure(go.Bar(x=by_sector.index, y=by_sector * 100, marker_color=NAVY))
            f.update_layout(title=f"Equities by sector ({by_sector.sum():.0%} of the portfolio)", yaxis_title="% of portfolio")
            b3.plotly_chart(fig_layout(f, 340), width="stretch")
            em = w[is_eq & uni["is_EM"]].sum()
            st.markdown(f"**Emerging markets:** {pct(em)} of the portfolio, {pct(em / max(w[is_eq].sum(), 1e-12))} of equities. "
                        f"**Cash:** {pct(w[uni['asset_class'] == 'Cash'].sum())}. **Positions above 0.5%:** {(w > 0.005).sum()}.")

            st.markdown("### Where the risk comes from")
            if res.volatility <= 1e-6:
                st.info("This portfolio carries no market risk, so there is no risk to split.")
            else:
                e0 = get_estimates()  # covariances do not depend on the view confidence
                rc = div.risk_contributions(w, e0.cov)
                by_cls = div.by_group(rc, uni["asset_class"]).reindex(CLASS_ORDER).fillna(0.0)
                by_cls.index = by_cls.index.map(CLASS_LABELS)
                b_w = pd.Series({bm.EQUITY_LEG: b_eq, bm.BOND_LEG: 1 - b_eq})
                b_rc = div.risk_contributions(b_w, e0.joint_cov)
                dm, dm_b = div.diversification_measures(w, e0.cov), div.diversification_measures(b_w, e0.joint_cov)
                money, risk_ = "Share of the money", "Share of the risk"
                r1, r2 = st.columns(2, gap="large")
                f = go.Figure([go.Bar(name=money, x=by_cls.index, y=by_cls["weight"] * 100, marker_color=GREY),
                               go.Bar(name=risk_, x=by_cls.index, y=by_cls["relative"] * 100, marker_color=NAVY)])
                f.update_layout(barmode="group", title="Money vs risk by asset class", yaxis_title="%")
                r1.plotly_chart(fig_layout(f, 420), width="stretch")
                top = rc.nlargest(div.TOP_N, "relative").iloc[::-1]
                names_ = [label(t) for t in top.index]
                f = go.Figure([go.Bar(name=money, y=names_, x=top["weight"] * 100, orientation="h", marker_color=GREY),
                               go.Bar(name=risk_, y=names_, x=top["relative"] * 100, orientation="h", marker_color=NAVY)])
                f.update_layout(barmode="group", title=f"The {div.TOP_N} largest risk contributors", xaxis_title="%")
                r2.plotly_chart(fig_layout(f, 420), width="stretch")
                c = st.columns(5)
                c[0].metric("Effective number of positions", f"{dm['effective_n']:.1f}",
                            help=f"1 / Herfindahl index. The Herfindahl index is the sum of the squared weights "
                                 f"({dm['herfindahl']:.3f} here; 1 = a single position). Its inverse is the number of "
                                 "equal positions that would be as concentrated as this portfolio.")
                c[1].metric("Effective number (entropy)", f"{dm['entropy_n']:.1f}",
                            help="exp(Shannon entropy of the weights), with entropy = -sum of w x ln(w): another count of "
                                 "equivalent equal positions, less driven by the largest ones.")
                c[2].metric("Gini coefficient", f"{dm['gini']:.2f}",
                            help=f"From the Lorenz curve of the weights over the {len(w)} eligible assets: 0 = the same "
                                 "weight in every asset, close to 1 = everything in one.")
                c[3].metric("Diversification ratio", f"{dm['diversification_ratio']:.2f}",
                            f"{dm['diversification_ratio'] - dm_b['diversification_ratio']:+.2f} vs benchmark",
                            help="Weighted average volatility of the holdings divided by the portfolio's volatility. "
                                 "1 = no diversification benefit; the higher, the more of the holdings' risks cancel out. "
                                 "The most diversified portfolio is the one that maximises it.")
                c[4].metric(f"Risk from the top {div.TOP_N}", pct(dm["top10_risk_share"], 0),
                            help=f"Share of the portfolio's volatility that comes from its {div.TOP_N} largest risk contributors.")
                spy_risk = b_rc.loc[bm.EQUITY_LEG, "relative"]
                st.caption(f"Risk contributions split the portfolio's volatility ({pct(res.volatility)}) among the holdings: "
                           "each position's weight times its marginal risk, i.e. how much the volatility rises when the "
                           "position grows a little. They add up exactly to the total (Euler's theorem), so the shares sum "
                           "to 100%; a negative share means the asset offsets the others. Equities are "
                           f"{pct(by_cls.loc['Equities', 'weight'], 0)} of the money but "
                           f"{pct(by_cls.loc['Equities', 'relative'], 0)} of the risk. In the benchmark "
                           f"({bench_label(profile).removeprefix('Benchmark (')}, SPY carries {pct(spy_risk, 0)} of the risk "
                           f"with {pct(b_eq, 0)} of the money.")

        with sub_bonds:
            bs = bonds.sleeve_summary(w, universe)
            if bs["share"] <= 0:
                st.info("No bonds in this portfolio.")
            else:
                c = st.columns(5)
                c[0].metric("Bonds in the portfolio", pct(bs["share"]))
                c[1].metric("Average duration", f"{bs['duration']:.1f} years")
                c[2].metric("Average rating", bs["avg_rating"])
                c[3].metric("Investment grade", pct(bs["ig_share"]), help="Share of the bond portfolio")
                c[4].metric("High yield incl. EM", pct(bs["hy_share_of_portfolio"]), help="Share of the whole portfolio")
                g1, g2, g3 = st.columns(3)
                for col, key, title in ((g1, "by_issuer", "By issuer type"), (g2, "by_maturity", "By maturity"),
                                        (g3, "by_quality", "By credit quality")):
                    s_ = bs[key]
                    f = go.Figure(go.Bar(x=s_.index, y=s_ * 100, marker_color=NAVY))
                    f.update_layout(title=title, yaxis_title="% of bonds")
                    col.plotly_chart(fig_layout(f, 300), width="stretch")
                st.dataframe(bs["funds"].style.format({"Duration": "{:.1f}", "Weight": "{:.1%}", "Share of bonds": "{:.1%}"}),
                             width="stretch")
                st.caption("Duration and ratings are the funds' current fact-sheet values (iShares, State Street, "
                           "Vanguard, Sept./Oct. 2026), applied to the whole history. Maturity buckets use the average "
                           "maturity: <1y ultra-short, 1-5y short, 5-10y intermediate, 10y+ long.")

        with sub_holdings:
            held = w[w > 0.005].sort_values()
            f = go.Figure(go.Bar(y=[label(t) for t in held.index], x=held * 100, orientation="h",
                                 marker_color=[GREY if universe.at[t, "vehicle"] != "Stock" else NAVY for t in held.index]))
            f.update_layout(xaxis_title="Weight (%)", title="Holdings (funds and cash in grey)")
            st.plotly_chart(fig_layout(f, max(400, 24 * len(held))), width="stretch")

            with st.expander("Full allocation in USD"):
                table = pd.DataFrame({"Asset": [label(t) for t in w.index], "Class": uni["asset_class"].values,
                                      "Sector / type": np.where(is_bond, uni["bond_type"], uni["sector"]),
                                      "Weight": w.values, "Amount (USD)": w.values * wealth,
                                      "Liquidity": uni["liquidity_tier"].values,
                                      "Trading cost (bps)": uni["tc_bps"].values}, index=w.index)
                shown = table[table["Weight"] > 1e-4].sort_values("Weight", ascending=False)
                dust = table.loc[(table["Weight"] > 0) & (table["Weight"] <= 1e-4), ["Weight", "Amount (USD)"]].sum()
                if dust["Weight"] > 0:  # solver residuals: listed as one line so the table sums to 100%
                    shown.loc["_other"] = {"Asset": "Other positions below 0.01%", "Class": "", "Sector / type": "",
                                           "Weight": dust["Weight"], "Amount (USD)": dust["Amount (USD)"],
                                           "Liquidity": "", "Trading cost (bps)": np.nan}
                st.dataframe(shown.style.format({"Weight": "{:.2%}", "Amount (USD)": lambda x: usd(x),
                                                 "Trading cost (bps)": lambda x: "" if pd.isna(x) else f"{x:.0f}"}),
                             hide_index=True, width="stretch")
                st.caption("Trading cost = estimated cost of buying or selling one dollar of the asset (spread and "
                           "market impact), by liquidity tier: stocks by average daily dollar volume over the last "
                           "12 months (2 / 5 / 10 bps), high-yield and EM bond funds 8 bps, other funds 3 bps, cash 0.")
            with st.expander("Binding constraints"):
                act = res.duals[res.duals["active"]]
                st.caption("Gain in expected return (bps per year) if the bound were relaxed by one unit, in the "
                           "bound's own unit: +1 percentage point of the portfolio, of the bond portfolio (investment "
                           "grade) or of the equities (emerging markets), +1 year for the bond duration. A first-order "
                           "estimate from the optimiser; n/a for an exclusion or a limit of 0% or 100%, where it is not defined.")
                st.dataframe(act[["constraint", "label", "unit", "bps_per_unit"]].rename(columns={
                    "constraint": "Constraint", "label": "Bound", "unit": "Relaxation",
                    "bps_per_unit": "Gain (bps p.a.)"}).style.format(
                    {"Gain (bps p.a.)": lambda x: "n/a" if pd.isna(x) else f"{x:.1f}"}), hide_index=True,
                    width="stretch")

# --- Home: live figures (written into the Home tab now that the portfolio is known) -----------

with home_live, guarded("Home"), st.container(key="home-stats"):
    st.markdown('<div class="home-section">Today at a glance</div>', unsafe_allow_html=True)
    cap = cons_spec["max_volatility"]["params"]["limit"]
    c = [col.container(border=True) for col in st.columns(4, gap="medium")]
    c[0].metric("Risk profile", profile, f"volatility budget {cap:.0%}", delta_color="off",
                help="From the questionnaire (Client Profile tab), unless overridden in the client settings.")
    if res.ok:
        c[1].metric("Expected return p.a. above T-bill", pct(res.expected_return, 1),
                    help="Recommended portfolio, ex ante. The Portfolio tab compares it with the benchmark "
                         f"({bench_label(profile)}) and SPY.")
        c[2].metric("Estimated volatility p.a.", pct(res.volatility, 1), f"budget {cap:.0%}", delta_color="off",
                    help="Estimated before investing; the optimiser uses the whole budget when it raises the expected "
                         "return. Realised volatility may be higher (see the Backtest tab).")
    else:
        c[1].metric("Recommended portfolio", "Not feasible", delta_color="off",
                    help="The current constraints cannot all be met: see the Portfolio tab.")
    month, share, z = ins.signal_at(signal, returns.index[-1]) if signal is not None else (None, None, None)
    if z is None:
        c[3].metric("Insider signal", "Not available", delta_color="off")
    else:
        mood = "Above normal" if z > 0.5 else "Below normal" if z < -0.5 else "Normal"
        effect = ("not used" if view_source != INSIDER
                  else "leans to equities" if z > 0.5 else "leans to bonds" if z < -0.5 else "no tilt")
        c[3].metric("Insider signal", mood, effect, delta_color="off",
                    help=f"Insider buying breadth {share:.0%} in the 3 months to {month:%B %Y}, {z:+.1f} standard "
                         "deviations against its last 5 years. Details in the Insider Signal tab.")
    if res.ok:
        mix = res.weights.groupby(universe.loc[res.weights.index, "asset_class"]).sum().reindex(CLASS_ORDER).fillna(0)
        st.caption("Recommended mix: " + ", ".join(f"{CLASS_LABELS[k].lower()} {v:.0%}" for k, v in mix.items() if v >= 0.005)
                   + f" (for a {usd_short(wealth)} portfolio).")

# --- Insider Signal --------------------------------------------------------------------------

with tabs[3], guarded("Insider Signal"):
    page_header()
    st.subheader("House view: what company insiders do with their own money")
    if signal is None:
        st.info(f"The insider data is not available, so the portfolio uses the market equilibrium returns, without "
                f"a view. ({insider_error})")
    else:
        st.markdown(
            "Every officer and director of a US listed company must report a trade in their own company's shares "
            "to the SEC within two business days (**Form 4**). We read all of them since 2006, about "
            f"{insider_meta['insider_filings'] / 1e6:.1f} million filings, and measure how many companies' insiders "
            "are **buying** on the open market rather than selling. Insiders sell for many reasons (diversification, "
            "taxes, automatic sale plans); they buy for one: they expect the price to rise. When buying spreads "
            "across companies, we expect equities to beat bonds by more than usual, and tilt the portfolio "
            "toward equities, without leaving the client's volatility budget.")
        conf_ins = confidence
        e_now = get_estimates()
        view = ins.with_insider_view(e_now, e_now.end, signal, conf_ins)[1]
        if view.z is None:
            st.warning("The latest insider data is too old to form a view: the portfolio uses the equilibrium returns.")
        else:
            hist = signal["buy_share"].loc[:view.data_month].tail(ins.Z_WINDOW)
            mood = "above normal" if view.z > 0.5 else "below normal" if view.z < -0.5 else "normal"
            eq_spread = view.table.at[0, "equilibrium"]
            post_spread = view.table.at[0, "posterior"]
            c = st.columns(4)
            c[0].metric("Insider buying breadth", pct(view.buy_share, 0),
                        f"{(view.buy_share - hist.mean()) * 100:+.1f} pp vs 5-year average", delta_color="off",
                        help=f"Over the 3 months to {view.data_month:%B %Y}: each month, the US listed companies "
                             "whose officers or directors bought their own shares on the open market, divided by "
                             "those that bought or sold, with the monthly counts added up (a company active in "
                             f"several months counts each time). 5-year average: {hist.mean():.0%}.")
            c[1].metric("Signal", mood.capitalize(), f"{view.z:+.1f} standard deviations", delta_color="off",
                        help="Breadth compared with its own last 5 years (z-score, capped at ±2). Insider buying has "
                             "drifted down since 2006 as automatic sale plans spread, so it is compared with its "
                             "recent history rather than a fixed level.")
            c[2].metric("Expected premium of equities over bonds", pct(post_spread, 2),
                        f"{(post_spread - eq_spread) * 1e4:+.0f} bps vs equilibrium",
                        help=f"SPY minus AGG, annualised. Equilibrium {eq_spread:.2%}; insider view "
                             f"{view.table.at[0, 'view_value']:.2%}; blended at {conf_ins:.0%} confidence "
                             "(Black-Litterman).")
            if view_source == INSIDER and res.ok:
                try:
                    res0 = optimise(profile, cons_json, INSIDER, 0.0)[0]
                except EXPECTED_ERRORS:
                    res0 = None
                if res0 is not None and res0.ok:
                    is_eq_ = universe["asset_class"] == "Equity"
                    eq_with = res.weights.reindex(universe.index).fillna(0)[is_eq_].sum()
                    eq_without = res0.weights.reindex(universe.index).fillna(0)[is_eq_].sum()
                    c[3].metric("Equities in the portfolio", pct(eq_with, 1),
                                f"{(eq_with - eq_without) * 100:+.1f} pp vs no view",
                                help=f"Without any view (pure equilibrium) the portfolio would hold {eq_without:.1%} "
                                     "in equities. The volatility budget and the bond floor limit the tilt.")

        recent_share = signal["buy_share"].dropna()
        spy_growth = (1 + bench["SPY"].reindex(recent_share.index).fillna(0)).cumprod()
        avg5 = signal["buy_share"].rolling(ins.Z_WINDOW, min_periods=ins.Z_MIN_MONTHS).mean()
        f = go.Figure([
            go.Scatter(x=recent_share.index, y=recent_share * 100, name="Insider buying breadth (3 months)",
                       line=dict(color=NAVY, width=2.5)),
            go.Scatter(x=avg5.index, y=avg5 * 100, name="Its 5-year average", line=dict(color=NAVY, width=1, dash="dot")),
            go.Scatter(x=spy_growth.index, y=spy_growth, name="SPY, growth of 1 USD (right axis)", yaxis="y2",
                       line=dict(color=ACCENT, width=1.5))])
        f.update_layout(title="Insiders buy when markets are low", yaxis_title="% buying (company-months)",
                        yaxis2=dict(overlaying="y", side="right", type="log", showgrid=False, title="SPY (log)",
                                    tickvals=[1, 2, 4, 8], ticktext=["1", "2", "4", "8"]))
        st.plotly_chart(fig_layout(f, 420), width="stretch")
        st.caption("Breadth: each month, count the US listed companies whose officers or directors bought their own "
                   "shares on the open market, and those whose insiders sold. Breadth = purchases / (purchases + "
                   "sales), each count added up over the last 3 months, so the unit is the company-month: a company "
                   "active in several months, or on both sides, counts each time. Counting companies, not dollars, "
                   "stops a few huge sales from dominating. It peaked at the market lows of 2008-2009, autumn 2011 and "
                   "spring 2020.")

        st.markdown("#### Did insider buying predict the market?")
        bench_ex = (bench["SPY"] - rf.reindex(bench.index)).dropna()
        evid = ins.market_evidence(signal, bench_ex)
        e1, e2 = st.columns([3, 2], gap="large")
        if len(evid):
            f = go.Figure(go.Bar(x=evid.index.astype(str), y=evid["next_return"] * 100, marker_color=NAVY,
                                 text=[f"{v:.1%}" for v in evid["next_return"]], textposition="outside"))
            f.update_layout(title="SPY return above T-bill over the next 12 months, by insider-buying quintile",
                            xaxis_title="Months sorted by insider buying (z-score)", yaxis_title="% (average)")
            e1.plotly_chart(fig_layout(f, 380), width="stretch")
        if len(evid):
            top_ret = evid["next_return"].iloc[-1]
            rest_ret = (evid["next_return"].iloc[:-1] * evid["months"].iloc[:-1]).sum() / evid["months"].iloc[:-1].sum()
            e2.markdown(
                f"Months are split into five equal groups by the insider signal ({int(evid['months'].sum())} months "
                f"since {signal['z'].first_valid_index():%Y} with a known next year). After the fifth of months with "
                f"the broadest insider buying, SPY beat the T-bill by **{top_ret:.0%}** over the next year on "
                f"average, against **{rest_ret:.0%}** after the other months.\n\n"
                "**Honest limits.** The effect is modest and noisy: the four other groups are not in order, and a "
                "12-month regression gives a Newey-West t-statistic of about 2. Insiders also tend to buy early: "
                "in spring 2008 they bought before the worst of the crisis. This is why the view only tilts the "
                "portfolio, with a confidence you choose, and always within the volatility budget.")

        with st.expander("How the view enters the portfolio (Black-Litterman)"):
            st.markdown(
                "One relative view, *equities (SPY) will beat bonds (AGG) by Q*, with "
                f"Q = equilibrium spread + {ins.KAPPA:.0%} × z (z capped at ±{ins.Z_CAP:.0f}). Posterior returns: "
                "μ = π + τΣP′(PτΣP′ + Ω)⁻¹(Q − Pπ), with P = (+1 SPY, −1 AGG), τ = "
                f"{est.BL_TAU}, Ω = PτΣP′(1 − c)/c and c the confidence. For this single view, c is exactly its "
                "weight against the equilibrium. Every asset moves according to its covariance with SPY minus AGG: "
                "equity-like assets move the most, up when insiders buy more than usual and down when they buy less. In the backtest, each rebalancing only uses filings published "
                "by the end of the previous month.")
            if len(view.table):
                st.dataframe(view.table.rename(columns={"view": "View", "equilibrium": "Equilibrium",
                                                        "view_value": "Insider view", "posterior": "After blending",
                                                        "evidence": "Evidence"}).style.format(
                    {"Equilibrium": "{:.2%}", "Insider view": "{:.2%}", "After blending": "{:.2%}"}),
                    hide_index=True, width="stretch")

        st.markdown("#### What insiders did in our companies")
        last = returns.index[-1]
        recent = ins.recent_purchases(insider_trades, last)
        sales = insider_trades[(insider_trades["direction"] == "sell")
                               & (insider_trades["filing_date"] > last - pd.DateOffset(months=ins.RECENT_MONTHS))]
        st.markdown(f"In the {ins.RECENT_MONTHS} months to {last:%B %Y}: **{len(recent)} purchases** "
                    f"(at least {usd_short(ins.MIN_PURCHASE_USD)}) in {recent['ticker'].nunique()} of our companies, "
                    f"against **{len(sales):,} sales** in {sales['ticker'].nunique()} companies.")
        if len(recent):
            show = recent.assign(Company=[label(t) for t in recent["ticker"]])[
                ["filing_date", "Company", "insider", "role", "value_usd"]].rename(columns={
                    "filing_date": "Filed", "insider": "Insider", "role": "Role", "value_usd": "Amount (USD)"})
            st.dataframe(show.style.format({"Filed": lambda d: f"{d:%d %b %Y}", "Amount (USD)": lambda x: usd(x)}),
                         hide_index=True, width="stretch", height=min(400, 38 + 35 * len(show)))
        study = ins.purchase_event_study(insider_trades, returns, universe)
        s1, s2 = st.columns([2, 3], gap="large")
        s1.dataframe(study.rename(columns={"vs_universe": "vs our stocks", "t_universe": "t-stat",
                                           "vs_sector": "vs same sector", "t_sector": "t-stat ", "events": "Events"})
                     .style.format({"vs our stocks": "{:+.2%}", "vs same sector": "{:+.2%}", "t-stat": "{:.1f}",
                                    "t-stat ": "{:.1f}"}), width="stretch")
        s2.markdown("**Why we do not bet on individual stocks.** In our large companies, stocks bought by their "
                    "insiders did no better than the average of our stocks, or of their sector, in the same months "
                    "(table: average return after a month with an insider purchase, since 2006). Research finds the "
                    "effect in small companies, which are not in our universe. So insider purchases are shown here "
                    "for information, and only the market-wide signal enters the portfolio.")
        st.caption(f"Source: SEC Insider Transactions Data Sets (Form 4, open-market purchases and sales by officers "
                   f"and directors), filings {pd.Timestamp(insider_meta['first_filing']):%B %Y} – "
                   f"{pd.Timestamp(insider_meta['last_filing']):%B %Y}. Companies are matched by their SEC registration, "
                   "including predecessors (e.g. Google Inc. before Alphabet). Foreign companies are exempt from Form 4, "
                   f"so there is no insider data for {', '.join(insider_meta.get('no_filings', [])) or 'none'}. "
                   "The market signal uses all US listed companies, including those that later disappeared, so it "
                   "is free of the survivorship bias of our stock list.")

with tabs[4], guarded("Backtest"):
    page_header()
    st.subheader(f"Backtest — {profile} profile with the client constraints")
    c1, _ = st.columns([1, 3])
    with c1.container(key="bt-rebalance"):
        freq = st.selectbox("Rebalancing", list(btm.FREQ_MONTHS), key="bt.freq")
    try:
        bt, ew, series = run_backtest(profile, cons_json, freq, view_source, confidence)
    except EXPECTED_ERRORS as exc:
        st.error(f"The backtest could not run: {exc}")
        bt = None
    if bt is not None:
        start, end = series.index[0], series.index[-1]
        metrics = pd.DataFrame({k: risk.performance_metrics(series[k], rf) for k in series}).T
        metrics["Final wealth (USD)"] = [wealth * (1 + series[k]).prod() for k in series]
        s_, b_ = metrics.loc["Strategy"], metrics.loc[bench_label(profile)]
        h = st.columns(4)
        h[0].metric("Annual return", pct(s_["annual_return"]),
                    f"{(s_['annual_return'] - b_['annual_return']) * 100:+.1f} pp vs benchmark")
        h[1].metric("Realised volatility", pct(s_["volatility"]),
                    f"{(s_['volatility'] - b_['volatility']) * 100:+.1f} pp vs benchmark", delta_color="inverse")
        h[2].metric("Sharpe ratio", f"{s_['sharpe']:.2f}", f"{s_['sharpe'] - b_['sharpe']:+.2f} vs benchmark")
        h[3].metric("Max drawdown", pct(s_["max_drawdown"]),
                    f"{(s_['max_drawdown'] - b_['max_drawdown']) * 100:+.1f} pp vs benchmark")
        with st.container(border=True):
            st.markdown("#### How to read these results")
            st.markdown(
                f"This is a simulation of the past, {start:%B %Y} – {end:%B %Y} ({len(series) / 12:.1f} years): every "
                f"{btm.FREQ_MONTHS[freq]} months the portfolio is rebuilt using only the {WINDOW} months of data "
                "that end one month before the trade (data to March, trade at the end-of-April close), and "
                "trading costs are deducted. "
                "The stock list is today's largest US companies, i.e. the ones that succeeded, so any strategy "
                "picking from it looks better than it really could have been. "
                f"The **{EW_LABEL}** portfolio buys the same companies naively, in equal amounts: it shares that "
                "bias, so the comparison is fair on that point. It is not our strategy without optimisation, "
                "though: it ignores the volatility budget and the bond limits (its bonds are longer and riskier "
                "than several profiles allow), so the gap mixes the method with differences in allocation and "
                "constraints. "
                "SPY and the profile benchmark are real index funds and are not affected; past results do not "
                "guarantee future returns.")
        for ev in bt.events:
            st.warning(ev)

        # Charts first, tables after; the headline numbers are above.
        growth = wealth * (1 + series).cumprod()
        styles = (("Strategy", NAVY, 3, "solid"), (EW_LABEL, EW_COLOR, 2, "dash"), (bench_label(profile), GREY, 2, "solid"),
                  ("SPY", ACCENT, 2, "dot"))
        f = go.Figure([go.Scatter(x=growth.index, y=growth[k], name=k, line=dict(color=clr, width=w_, dash=dash))
                       for k, clr, w_, dash in styles])
        f.update_layout(title=f"Growth of {usd(wealth)}", yaxis_title="USD", yaxis_type="log")
        st.plotly_chart(fig_layout(f, 430), width="stretch")

        reb = bt.rebalances[bt.rebalances["status"] == "ok"]
        left, right = st.columns(2, gap="large")
        f = go.Figure([go.Scatter(x=reb.index, y=reb["ex_ante_vol"] * 100, name="Estimated (ex ante)",
                                  line=dict(color=NAVY, width=3)),
                       go.Scatter(x=reb.index, y=reb["realised_vol"] * 100, name="Realised (next 12 months)",
                                  line=dict(color=ACCENT, width=2))])
        f.update_layout(title="Estimated vs realised volatility at each rebalancing", yaxis_title="% p.a.")
        left.plotly_chart(fig_layout(f, 380), width="stretch")
        left.caption(f"Average estimated volatility at rebalancing: {reb['ex_ante_vol'].mean():.1%}. "
                     f"Average realised volatility over the following 12 months: {reb['realised_vol'].mean():.1%}. "
                     f"Realised volatility over the entire backtest: {metrics.loc['Strategy', 'volatility']:.1%}. "
                     "Each 12-month window starts at a rebalancing, so consecutive windows overlap and their "
                     "average is not the same as the volatility over the whole period. The budget caps the "
                     "estimate; it is not a guarantee, and realised volatility can exceed it.")
        dd = pd.DataFrame({k: risk.drawdowns(series[k]) for k in series}) * 100
        f = go.Figure([go.Scatter(x=dd.index, y=dd[k], name=k, line=dict(color=clr, dash=dash))
                       for k, clr, _, dash in styles])
        f.update_layout(title="Drawdowns", yaxis_title="%")
        right.plotly_chart(fig_layout(f, 380), width="stretch")

        alloc = bt.weights.T.groupby(universe.loc[bt.weights.columns, "asset_class"]).sum().T.reindex(columns=CLASS_ORDER)
        alloc.columns = alloc.columns.map(CLASS_LABELS)
        f = go.Figure([go.Scatter(x=alloc.index, y=alloc[c] * 100, name=c, stackgroup="one", mode="none",
                                  fillcolor=clr) for c, clr in zip(alloc.columns, CLASS_COLORS)])
        f.update_layout(title="Allocation by asset class over time (month-end, after drift)", yaxis_title="%",
                        yaxis_range=[0, 100])
        st.plotly_chart(fig_layout(f, 360), width="stretch")

        # Tables after charts.
        st.markdown("#### Results")
        st.dataframe(metrics.rename(columns={"annual_return": "Annual return", "volatility": "Volatility",
                                             "sharpe": "Sharpe ratio", "max_drawdown": "Max drawdown"}).style.format(
            {"Annual return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe ratio": "{:.2f}", "Max drawdown": "{:.1%}",
             "Final wealth (USD)": lambda x: usd(x)}),
            width="stretch")
        if spec_p.benchmark_equity >= 1:
            st.caption("The Dynamic profile's benchmark is 100% SPY, so its row is identical to SPY.")
        # separate table: turnover and costs only exist for the two portfolios we trade (an empty
        # cell for the index benchmarks would show as "None")
        trading = pd.DataFrame({"Turnover p.a.": [btm.annual_turnover(bt), btm.annual_turnover(ew)],
                                "Trading costs p.a.": [btm.annual_cost(bt), btm.annual_cost(ew)]},
                               index=["Strategy", EW_LABEL])
        t1, _ = st.columns([2, 3])
        t1.dataframe(trading.style.format({"Turnover p.a.": "{:.0%}",
                                           "Trading costs p.a.": lambda x: f"{x * 1e4:.1f} bps"}), width="stretch")
        t1.caption("Turnover = buys + sells per year as a share of the portfolio (initial purchase excluded); "
                   "trading costs include the initial purchase.")

        if st.toggle("Compare with naive expected returns", key="bt.compare",
                     help="The same backtest with the insider view at your confidence, a naive view (past "
                          "performance) and no view (it runs up to two more backtests the first time)."):
            choices = {f"Insider view, our house view ({confidence:.0%})": (INSIDER, confidence),
                       f"Naive: past performance ({est.DEFAULT_VIEW_CONFIDENCE:.0%})": (PAST, est.DEFAULT_VIEW_CONFIDENCE),
                       "No view (equilibrium only)": (PAST, 0.0)}
            rows = {}
            for name, (src, conf) in choices.items():
                if src == INSIDER and signal is None:
                    continue
                bt_k = run_backtest(profile, cons_json, freq, src, conf)[0]
                ok_k = bt_k.rebalances[bt_k.rebalances["status"] == "ok"]
                rows[name] = {**risk.performance_metrics(bt_k.returns, rf),
                              "ex_ante": ok_k["ex_ante_vol"].mean(), "turnover": btm.annual_turnover(bt_k)}
            views_cmp = pd.DataFrame(rows).T.rename(columns={
                "annual_return": "Annual return", "volatility": "Volatility", "sharpe": "Sharpe ratio",
                "max_drawdown": "Max drawdown", "ex_ante": "Estimated volatility (ex ante)", "turnover": "Turnover p.a."})
            st.dataframe(views_cmp.style.format({"Annual return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe ratio": "{:.2f}",
                                                 "Max drawdown": "{:.1%}", "Estimated volatility (ex ante)": "{:.2%}",
                                                 "Turnover p.a.": "{:.0%}"}), width="stretch")
            st.caption("Same profile, constraints, dates and trading costs: only the expected returns change. The naive "
                       f"view takes each asset's average return over the last {WINDOW} months as a view (Black-Litterman, "
                       f"{est.DEFAULT_VIEW_CONFIDENCE:.0%} confidence): it simply bets that recent winners keep winning. "
                       "Our stock list is made of today's winners, so its backtest gains are partly hindsight, and it "
                       "takes more risk than its estimate and trades about twice as much. It is shown for comparison "
                       "only, not offered to clients. The insider view is measured on all US listed companies, "
                       "including those that later disappeared.")

with tabs[5], guarded("Simulation"):
    page_header()
    st.subheader(f"Wealth simulation — {profile} profile, starting with {usd(wealth)}")
    if require_ok(res):
        years = st.slider("Horizon (years)", 5, 20, 10, key="sim.years")
        loss_limit = prof.MAX_LOSS_LEVELS[answers["max_loss"]]
        try:
            sim, h_start, h_end, n_hist = simulate(profile, cons_json, st.session_state.get("bt.freq", "Quarterly"),
                                                   view_source, confidence, years, float(wealth), loss_limit)
        except EXPECTED_ERRORS as exc:
            st.error(f"The simulation could not run: {exc}")
            sim = None
        if sim is not None:
            year_col = f"Loss above {loss_limit:.0%} in a given year"
            dd_col, dd_bad_col = "Worst fall, typical", "Worst fall, bad case (5%)"
            var_col, es_col = "1-year VaR (95%)", "1-year expected shortfall (95%)"
            rows = {name: {"Median": v["median"], "Pessimistic (5%)": v["p5"], "Optimistic (95%)": v["p95"],
                           year_col: v["prob_year_loss"], dd_col: v["dd_median"], dd_bad_col: v["dd_p5"],
                           var_col: v["year_var95"], es_col: v["year_es95"]}
                    for name, v in sim.items()}
            table = pd.DataFrame(rows).T.rename(index={"Portfolio": "Recommended portfolio",
                                                       "Benchmark": bench_label(profile)})
            p_, b_ = sim["Portfolio"], sim["Benchmark"]
            c = st.columns(3) + st.columns(3)  # wealth on the first row, risk on the second
            gain = p_["median"] - b_["median"]
            c[0].metric(f"Median wealth in {years} years", usd_short(p_["median"]),
                        f"{'+' if gain >= 0 else '-'}{usd_short(abs(gain))} vs benchmark",
                        help=f"{usd(p_['median'])}; benchmark {usd(b_['median'])}.")
            c[1].metric("Pessimistic scenario (5%)", usd_short(p_["p5"]), help=f"{usd(p_['p5'])}.")
            c[2].metric("Optimistic scenario (95%)", usd_short(p_["p95"]), help=f"{usd(p_['p95'])}.")
            def vs_bench(diff_pp: float, digits: int, color: str) -> dict:
                """Delta text and color; a difference that rounds to zero is shown neutrally."""
                if round(diff_pp, digits) == 0:
                    return {"delta": "same as benchmark", "delta_color": "off"}
                return {"delta": f"{diff_pp:+.{digits}f} pp vs benchmark", "delta_color": color}

            c[3].metric(f"Chance of a loss above {loss_limit:.0%} in a year", pct(p_["prob_year_loss"]),
                        **vs_bench((p_["prob_year_loss"] - b_["prob_year_loss"]) * 100, 1, "inverse"),
                        help=f"Your answer in the questionnaire: a one-year loss of at most {loss_limit:.0%}. "
                             "Share of simulated years that lose more than that.")
            c[4].metric(f"Worst fall within {years} years (typical)", pct(p_["dd_median"], 0),
                        **vs_bench((p_["dd_median"] - b_["dd_median"]) * 100, 0, "normal"),
                        help=f"Largest fall from a previous peak during the {years} years, in a typical scenario "
                             f"(median). In a bad scenario (5% worst): {pct(p_['dd_p5'], 0)}.")
            c[5].metric("Average of the worst 5% of years", pct(p_["year_es95"]),
                        **vs_bench((p_["year_es95"] - b_["year_es95"]) * 100, 1, "normal"),
                        help="Expected shortfall (CVaR) at 95%: the average one-year return in the 5% worst simulated "
                             f"years, a {'loss' if p_['year_es95'] < 0 else 'gain'} of about "
                             f"{usd(abs(p_['year_es95']) * wealth)} on your wealth. The 95% value at risk, "
                             f"the threshold of those years, is {pct(p_['year_var95'])}.")
            money_cols = ["Median", "Pessimistic (5%)", "Optimistic (95%)"]
            st.markdown(f"**Wealth in {years} years**")
            st.dataframe(table[money_cols].style.format({k: lambda x: usd(x) for k in money_cols}), width="stretch")
            st.markdown("**Risk along the way**")
            st.dataframe(table.drop(columns=money_cols).style.format(
                {year_col: "{:.1%}", dd_col: "{:.0%}", dd_bad_col: "{:.0%}", var_col: "{:.1%}", es_col: "{:.1%}"}),
                width="stretch")
            st.caption(f"Loss in a given year: share of simulated years losing more than {loss_limit:.0%}, the "
                       "largest one-year loss you said you could accept in the questionnaire. Worst fall: the "
                       "largest drop from a previous peak during the horizon, in a typical scenario and in the "
                       "5% worst scenarios. VaR (value at risk, 95%): the one-year return that 95% of the simulated "
                       "years beat. Expected shortfall (95%): the average return of the 5% worst years, i.e. how bad "
                       "those years are on average. Expected shortfall is a coherent risk measure (combining two "
                       "portfolios never shows more risk than the two separately); VaR is not.")
            f = go.Figure()
            for name, clr, fill in (("Benchmark", GREY, THEME["fill_bench"]), ("Portfolio", NAVY, THEME["fill_main"])):
                b = sim[name]["bands"]
                x = b.index / 12
                label_ = "Recommended portfolio" if name == "Portfolio" else bench_label(profile)
                f.add_trace(go.Scatter(x=x, y=b["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                f.add_trace(go.Scatter(x=x, y=b["p5"], fill="tonexty", fillcolor=fill, line=dict(width=0),
                                       name=f"{label_}: 5%-95% range"))
                f.add_trace(go.Scatter(x=x, y=b["p50"], line=dict(color=clr, width=3), name=f"{label_}: median"))
            f.update_layout(xaxis_title="Years", yaxis_title="Wealth (USD)", yaxis_type="log",
                            title="Range of possible outcomes")
            st.plotly_chart(fig_layout(f, 460), width="stretch")
            st.caption(f"5,000 scenarios built by drawing random 12-month blocks of the strategy's backtest returns "
                       f"and of its benchmark, on the same months: {h_start:%B %Y} – {h_end:%B %Y} "
                       f"({n_hist / 12:.1f} years). These returns are out of sample: at every rebalancing the "
                       "portfolio was chosen using only data up to the month before, and trading costs are "
                       "deducted. Management fees and taxes are ignored. The simulation still inherits the "
                       "survivorship bias of the stock list (today's winners, picked with hindsight), so the "
                       "portfolio's outcomes are somewhat optimistic; the benchmark's are not.")

