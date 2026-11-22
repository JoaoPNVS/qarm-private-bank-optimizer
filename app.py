"""Private Client Portfolio Optimizer — Streamlit UI only (no heavy logic here)."""
from __future__ import annotations

import html
import json
import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (backtest as btm, benchmarks as bm, bonds, data, diversification as div, estimators as est,
                 frontier as fr, optimizer as opt, profiling as prof, risk, team)

st.set_page_config(page_title="Private Client Portfolio Optimizer", layout="wide")

NAVY, GREY, ACCENT = "#1f3b57", "#a3adb8", "#c0392b"
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


def bench_label(profile: str) -> str:
    e = prof.SPECS[profile].benchmark_equity
    return "Benchmark (100% SPY)" if e >= 1 else f"Benchmark ({e:.0%} SPY / {1 - e:.0%} AGG)"


def fig_layout(fig, height=380):
    # legend below the plot: above it, it collided with the title and wrapped over narrow charts
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, xanchor="left"),
                      font=dict(size=13), title=dict(x=0, xanchor="left"))
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


def build_constraints(profile: str, spec: dict) -> list[opt.Constraint]:
    cons = [c for c in prof.default_constraints(profile) if c.key in spec]
    for c in cons:
        c.params, c.active = spec[c.key]["params"], spec[c.key]["active"]
    return cons


@st.cache_data(show_spinner="Optimising the portfolio...")
def optimise(profile: str, cons_json: str, confidence: float):
    e = est.with_view_confidence(get_estimates(), confidence)
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


@st.cache_data(show_spinner="Running the backtest (re-optimising at every rebalancing)...")
def run_backtest(profile: str, cons_json: str, freq: str, confidence: float):
    universe, returns, rf, bench = load_data()
    res = btm.run_backtest(returns, rf, universe, build_constraints(profile, json.loads(cons_json)),
                           freq=freq, estimates=btm.with_view_confidence(backtest_estimates(), confidence))
    ew = btm.run_equal_weight(returns, universe, prof.SPECS[profile].min_bonds, freq=freq)
    idx = res.returns.index
    series = pd.DataFrame({"Strategy": res.returns, EW_LABEL: ew.returns.reindex(idx),
                           bench_label(profile): bm.profile_benchmark_returns(prof.SPECS[profile].benchmark_equity,
                                                                             bench).reindex(idx),
                           "SPY": bench["SPY"].reindex(idx)})
    return res, ew, series


@st.cache_data(show_spinner="Computing the efficient frontier...")
def frontiers(profile: str, cons_json: str, confidence: float):
    """Efficient frontier with the client's constraints and without them (long-only only), and the
    maximum-Sharpe (tangency) portfolio under the client's constraints."""
    e = est.with_view_confidence(get_estimates(), confidence)
    d = prof.problem_data(profile, e, load_data()[0])
    top = max(0.25, 1.6 * prof.SPECS[profile].max_volatility)
    cons = build_constraints(profile, json.loads(cons_json))
    front = fr.efficient_frontier(d, cons, max_vol=top)
    return front, fr.efficient_frontier(d, [], max_vol=top), fr.max_sharpe_portfolio(d, cons, front)


@st.cache_data(show_spinner="Simulating 5,000 wealth paths...")
def simulate(profile: str, cons_json: str, freq: str, confidence: float, years: int, wealth: float,
             loss_limit: float):
    """Block bootstrap of the strategy's OUT-OF-SAMPLE backtest returns (at every date the portfolio
    was chosen with past data only, net of trading costs) and of the profile benchmark on the same
    months. Replaying today's weights on the months used to choose them would be in-sample (look-ahead)."""
    _, series = run_backtest(profile, cons_json, freq, confidence)[1:]
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
        st.info("Adjust the constraints in the **Constraints** tab.")
    return res.ok


def grouped_bars(table: pd.DataFrame, title: str, colors=(NAVY, GREY)):
    f = go.Figure([go.Bar(name=c, x=table.index, y=table[c] * 100, marker_color=clr)
                   for c, clr in zip(table.columns, colors)])
    f.update_layout(barmode="group", yaxis_title="% of portfolio", title=title)
    return f


# --- data & sidebar --------------------------------------------------------------------------

try:
    universe, returns, rf, bench = load_data()
except data.DataError as exc:
    st.error(f"Market data unavailable: {exc}")
    st.stop()

names = universe["name"].to_dict()
label = lambda t: f"{names.get(t, t)} ({t})"  # noqa: E731

answers = {q.key: st.session_state.get(f"q.{q.key}", 1) for q in prof.QUESTIONS}
assessment = prof.assess(answers)

with st.sidebar:
    st.title("Private Client Portfolio Optimizer")
    st.caption("Highest expected return within the client's volatility budget. All figures in USD.")
    wealth = st.number_input("Client wealth (USD)", min_value=100_000, max_value=1_000_000_000,
                             value=5_000_000, step=250_000, format="%d")
    choice = st.selectbox("Risk profile", ["auto"] + prof.PROFILES, key="profile",
                          format_func=lambda p: f"From questionnaire ({assessment.profile})" if p == "auto" else p)
    profile = assessment.profile if choice == "auto" else choice
    spec_p = prof.SPECS[profile]
    st.markdown(f"**{profile}** — {spec_p.description}")
    st.markdown(f"Volatility budget: **{spec_p.max_volatility:.0%}** · bonds at least "
                f"**{spec_p.min_bonds:.0%}** · benchmark **{spec_p.benchmark_equity:.0%} SPY / "
                f"{1 - spec_p.benchmark_equity:.0%} AGG**")
    st.divider()
    confidence = st.slider(
        "Confidence in past performance", 0, int(est.MAX_VIEW_CONFIDENCE * 100),
        int(est.DEFAULT_VIEW_CONFIDENCE * 100), 5, format="%d%%", key="views",
        help="Black-Litterman. Expected returns start from the market equilibrium: each asset's return "
             "reflects only its risk relative to a 60% SPY / 40% AGG market portfolio. Each asset's average "
             f"return over the last {WINDOW} months is then added as a view, with this confidence. Because "
             "assets move together, the views reinforce each other, so even 10% tilts the portfolio noticeably. "
             "0% = pure equilibrium: most diversified, closest to the promised risk. Higher = more weight on "
             "recent trends, with more risk and trading.") / 100
    st.caption(f"Data: monthly total returns in USD up to {returns.index[-1]:%B %Y} (yfinance). "
               f"Estimates use the last {WINDOW} months. Risk-free: 3-month T-bill, currently "
               f"{rf.iloc[-1] * 12:.2%} p.a.")

tabs = st.tabs(["Client Profile", "Constraints", "Portfolio", "Backtest", "Simulation", "About Us"])

# --- Client Profile --------------------------------------------------------------------------

with tabs[0], guarded("Client Profile"):
    st.subheader("Suitability questionnaire")
    left, right = st.columns([3, 2], gap="large")
    with left:
        for q in prof.QUESTIONS:
            st.radio(q.text, list(range(len(q.options))), format_func=lambda i, q=q: q.options[i],
                     key=f"q.{q.key}", index=1)
    with right:
        a = prof.assess({q.key: st.session_state[f"q.{q.key}"] for q in prof.QUESTIONS})
        sp = prof.SPECS[a.profile]
        with st.container(border=True):
            st.markdown("#### Resulting profile")
            st.markdown(f"## {a.profile}")
            st.caption(sp.description)
            c1, c2, c3 = st.columns(3)
            c1.metric("Capacity (1-4)", f"{a.dimensions['capacity']:.2f}", a.levels["capacity"], delta_color="off")
            c2.metric("Willingness (1-4)", f"{a.dimensions['willingness']:.2f}", a.levels["willingness"],
                      delta_color="off")
            c3.metric("Knowledge (1-4)", f"{a.dimensions['knowledge']:.0f}")
            st.markdown("**Why this profile**")
            for r in a.reasons:
                st.markdown(f"- {r}")
            st.markdown(f"**What it means:** expected volatility at most {sp.max_volatility:.0%}, "
                        f"at least {sp.min_bonds:.0%} in bonds, compared with a "
                        f"{sp.benchmark_equity:.0%} SPY / {1 - sp.benchmark_equity:.0%} AGG benchmark.")
        if choice != "auto" and profile != a.profile:
            st.warning(f"The sidebar overrides the questionnaire: the **{profile}** profile is used "
                       f"instead of {a.profile}.")
        profiles_table = pd.DataFrame({p: {"Max volatility": f"{s.max_volatility:.0%}",
                                           "Min bonds": f"{s.min_bonds:.0%}",
                                           "Benchmark": f"{s.benchmark_equity:.0%} SPY / {1 - s.benchmark_equity:.0%} AGG"}
                                       for p, s in prof.SPECS.items()}).T
        st.dataframe(profiles_table, width="stretch")
        st.caption("Scoring: each answer scores 1-4. Capacity and willingness averages map to a profile "
                   "(<1.75 Conservative, <2.50 Balanced, <3.25 Growth, otherwise Dynamic); the lower one is "
                   "kept, then capped by knowledge. A horizon below 3 years always gives Conservative.")

# --- Constraints -----------------------------------------------------------------------------

defaults = {c.key: c for c in prof.default_constraints(profile)}
cons_spec: dict = {}
stock_tickers = [t for t in universe.index if universe.at[t, "vehicle"] == "Stock"]
sectors = sorted(universe.loc[universe["asset_class"] == "Equity", "sector"].unique())

with tabs[1], guarded("Constraints"):
    st.subheader(f"Client constraints — defaults of the {profile} profile")
    status_box = st.container()
    k = lambda name: f"{profile}.{name}"  # noqa: E731  (each profile keeps its own settings)
    col_a, col_b = st.columns(2, gap="large")

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

    card("max_volatility", col_a, lambda off: {"limit": pct_slider(
        "Max expected volatility", spec_p.max_volatility, 2, 25, 0.5, k("vol"), off)}, mandatory=True)
    card("min_bonds", col_a, lambda off: {"min_share": pct_slider(
        "Min share in bond funds", spec_p.min_bonds, 0, 100, 5, k("bonds"), off)})
    card("max_cash", col_a, lambda off: {"limit": pct_slider(
        "Max cash", prof.MAX_CASH, 0, 100, 5, k("cash"), off)})
    card("em_equity_cap", col_a, lambda off: {"ratio": pct_slider(
        "Max emerging markets (share of equities)", prof.EM_EQUITY_CAP, 0, 100, 1, k("em"), off)})
    card("max_stock_weight", col_b, lambda off: {"limit": pct_slider(
        "Max weight per stock", prof.MAX_STOCK_WEIGHT, 1, 30, 1, k("stock"), off)})
    card("max_etf_weight", col_b, lambda off: {"limit": pct_slider(
        "Max weight per equity / gold ETF", prof.MAX_ETF_WEIGHT, 5, 100, 5, k("etf"), off)})
    card("max_bond_etf_weight", col_a, lambda off: {"limit": pct_slider(
        "Max weight per bond ETF", prof.MAX_BOND_ETF_WEIGHT, 5, 100, 5, k("bondetf"), off)})
    card("max_bond_duration", col_a, lambda off: {"limit": st.slider(
        "Max average duration of the bonds (years)", 0.5, 15.0, float(spec_p.max_bond_duration), 0.5,
        key=k("duration"), disabled=off)})
    card("min_investment_grade", col_a, lambda off: {"min_share": pct_slider(
        "Min investment grade (share of bonds)", spec_p.min_investment_grade, 0, 100, 5, k("ig"), off)})
    card("max_high_yield", col_a, lambda off: {"limit": pct_slider(
        "Max high yield incl. EM bonds (share of portfolio)", spec_p.max_high_yield, 0, 100, 1, k("hy"), off)})
    card("max_sector_weight", col_b, lambda off: {"limit": pct_slider(
        "Max weight per equity sector", prof.MAX_SECTOR_WEIGHT, 5, 100, 5, k("sector"), off)})
    card("exclusions", col_b, lambda off: {
        "sectors": st.multiselect("Excluded sectors", sectors, default=[], key=k("excl.sectors"), disabled=off),
        "tickers": st.multiselect("Excluded companies", stock_tickers, default=[], format_func=label,
                                  key=k("excl.tickers"), disabled=off)})
    st.caption("Emerging markets = the EEM fund plus PDD (China) and MELI (Latin America). International "
               "equity funds (VGK, EWJ, EEM) count as one sector. Bond funds and gold are not subject to "
               "the sector limit. Bond durations and ratings are the funds' current fact-sheet values. "
               "High yield = HYG and the emerging-market bond fund EMB.")

# --- optimisation ----------------------------------------------------------------------------

cons_json = json.dumps(cons_spec, sort_keys=True)
try:
    res, bench_stats, spy_stats, te = optimise(profile, cons_json, confidence)
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

# --- Portfolio -------------------------------------------------------------------------------

with tabs[2], guarded("Portfolio"):
    st.subheader(f"Recommended portfolio — {profile} profile")
    if require_ok(res):
        w = res.weights
        uni = universe.loc[w.index]
        c = st.columns(4)
        c[0].metric("Expected return above T-bill", pct(res.expected_return, 2),
                    f"{(res.expected_return - bench_stats['expected_return']) * 1e4:+.0f} bps vs benchmark")
        c[1].metric("Expected volatility", pct(res.volatility, 2),
                    f"{(res.volatility - bench_stats['volatility']) * 100:+.2f} pp vs benchmark", delta_color="inverse")
        c[2].metric("Sharpe ratio", f"{res.sharpe:.2f}", f"{res.sharpe - bench_stats['sharpe']:+.2f} vs benchmark")
        c[3].metric("Tracking error vs benchmark", pct(te, 2))
        comp = pd.DataFrame({"Portfolio": [res.expected_return, res.volatility, res.sharpe],
                             bench_label(profile): [bench_stats["expected_return"], bench_stats["volatility"], bench_stats["sharpe"]],
                             "SPY": [spy_stats["expected_return"], spy_stats["volatility"], spy_stats["sharpe"]]},
                            index=["Expected excess return", "Expected volatility", "Sharpe ratio"])
        st.dataframe(comp.style.format("{:.2%}", subset=pd.IndexSlice[["Expected excess return", "Expected volatility"], :])
                     .format("{:.2f}", subset=pd.IndexSlice[["Sharpe ratio"], :]), width="stretch")
        st.caption(f"Ex-ante figures, estimated on the last {WINDOW} months. Expected returns: Black-Litterman, i.e. "
                   "market-equilibrium returns (implied by a 60% SPY / 40% AGG market portfolio) combined with each "
                   f"asset's recent average return as a view, at {confidence:.0%} confidence (sidebar). Risk: Ledoit-Wolf "
                   "covariance. The portfolio and its benchmarks are estimated with the same method; see the "
                   "Backtest tab for what this strategy actually delivered.")

        st.markdown("### Efficient frontier")
        try:
            front, free, tangency = frontiers(profile, cons_json, confidence)
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
                           line=dict(color="#c9a227", width=1.5, dash="dot")),
                go.Scatter(x=[gmv["volatility"] * 100], y=[gmv["expected_return"] * 100], name="Minimum variance",
                           mode="markers", marker=dict(color="white", size=11, symbol="circle", line=dict(color=NAVY, width=2))),
                go.Scatter(x=[tangency["volatility"] * 100], y=[tangency["expected_return"] * 100],
                           name="Maximum Sharpe (tangency)", mode="markers",
                           marker=dict(color="#c9a227", size=13, symbol="triangle-up", line=dict(color="white", width=1))),
                go.Scatter(x=[bench_stats["volatility"] * 100], y=[bench_stats["expected_return"] * 100],
                           name=bench_label(profile), mode="markers",
                           marker=dict(color=GREY, size=13, symbol="diamond", line=dict(color="white", width=1))),
                go.Scatter(x=[spy_stats["volatility"] * 100], y=[spy_stats["expected_return"] * 100], name="SPY",
                           mode="markers", marker=dict(color=ACCENT, size=12, line=dict(color="white", width=1))),
                go.Scatter(x=[res.volatility * 100], y=[res.expected_return * 100], name="Recommended portfolio",
                           mode="markers", marker=dict(color=NAVY, size=18, symbol="star", line=dict(color="white", width=1)))])
            f.add_vline(x=cap * 100, line=dict(color=NAVY, width=1, dash="dot"),
                        annotation_text=f"Volatility budget {cap:.0%}", annotation_position="top left")
            f.update_layout(title="Best expected return for each level of risk", xaxis_title="Expected volatility (% p.a.)",
                            yaxis_title="Expected return above T-bill (% p.a.)")
            f1.plotly_chart(fig_layout(f, 430), width="stretch")
            mix = front.set_index("volatility")[fr.CLASS_ORDER].rename(columns=CLASS_LABELS) * 100
            g = go.Figure([go.Scatter(x=mix.index * 100, y=mix[c], name=c, stackgroup="one", mode="none", fillcolor=clr)
                           for c, clr in zip(mix.columns, [NAVY, "#7f9cb8", "#c9a227", "#d9d9d9"])])
            g.add_vline(x=cap * 100, line=dict(color="white", width=2, dash="dot"))
            g.update_layout(title="Allocation along the frontier", xaxis_title="Expected volatility (% p.a.)",
                            yaxis_title="% of portfolio", yaxis_range=[0, 100])
            f2.plotly_chart(fig_layout(g, 430), width="stretch")
            st.caption(f"Each point of the curves is the portfolio with the highest expected return for that level of "
                       f"risk. The recommended portfolio (star) is the point at the volatility budget. At this risk, "
                       f"dropping all client constraints would add about {cost_bps:.0f} bps of expected return per "
                       "year: that is what diversification limits, the bond floor and the other rules cost on these "
                       "estimates, in exchange for a sturdier portfolio. Without constraints the curve starts at cash "
                       "(zero risk) and is almost a straight line: mixing cash with the best risky portfolio, as on "
                       "the capital market line. The benchmark and SPY sit below the curve: "
                       "for the same estimated risk, an optimised mix of the universe expects more (before "
                       "estimation error; see the Backtest tab).")
            gap = res.volatility - tangency["volatility"]
            if abs(gap) < 0.002 or tangency["sharpe"] - res.sharpe < 0.005:
                where = (f"Your portfolio is practically the maximum-Sharpe portfolio (Sharpe ratio {res.sharpe:.2f}): "
                         "at this volatility budget, no constrained mix offers a better trade-off.")
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

        b_eq = spec_p.benchmark_equity
        by_class = pd.DataFrame({"Portfolio": w.groupby(uni["asset_class"]).sum(),
                                 "Benchmark": pd.Series({"Equity": b_eq, "Bond": 1 - b_eq})}).reindex(CLASS_ORDER).fillna(0)
        by_class.index = by_class.index.map(CLASS_LABELS)
        is_bond = uni["asset_class"] == "Bond"
        is_eq = uni["asset_class"] == "Equity"
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

        st.markdown("### Bond portfolio")
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
            st.caption("Gain in expected return (bps p.a.) if the bound were relaxed by 1 percentage point.")
            st.dataframe(act[["constraint", "label", "bps_per_pp"]].rename(columns={
                "constraint": "Constraint", "label": "Bound", "bps_per_pp": "bps per +1pp"}).style.format(
                {"bps per +1pp": "{:.1f}"}), hide_index=True, width="stretch")

with tabs[3], guarded("Backtest"):
    st.subheader(f"Backtest — {profile} profile with the client constraints")
    c1, _ = st.columns([1, 3])
    freq = c1.selectbox("Rebalancing", list(btm.FREQ_MONTHS), key="bt.freq")
    try:
        bt, ew, series = run_backtest(profile, cons_json, freq, confidence)
    except EXPECTED_ERRORS as exc:
        st.error(f"The backtest could not run: {exc}")
        bt = None
    if bt is not None:
        start, end = series.index[0], series.index[-1]
        with st.container(border=True):
            st.markdown("#### How to read these results")
            st.markdown(
                f"This is a simulation of the past, {start:%B %Y} – {end:%B %Y} ({len(series) / 12:.1f} years): every "
                f"{btm.FREQ_MONTHS[freq]} months the portfolio is rebuilt using only the {WINDOW} months of data "
                "available at that date, and trading costs are deducted. "
                "The stock list is today's largest US companies, i.e. the ones that succeeded, so any strategy "
                "picking from it looks better than it really could have been. "
                f"The **{EW_LABEL}** portfolio buys the same companies naively, in equal amounts: it shares that "
                "bias, so the gap between it and our strategy measures what the optimisation adds. "
                "SPY and the profile benchmark are real index funds and are not affected; past results do not "
                "guarantee future returns.")
        for ev in bt.events:
            st.warning(ev)
        metrics = pd.DataFrame({k: risk.performance_metrics(series[k], rf) for k in series}).T
        metrics["Final wealth (USD)"] = [wealth * (1 + series[k]).prod() for k in series]
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

        growth = wealth * (1 + series).cumprod()
        styles = (("Strategy", NAVY, 3, "solid"), (EW_LABEL, "#7f9cb8", 2, "dash"), (bench_label(profile), GREY, 2, "solid"),
                  ("SPY", ACCENT, 2, "dot"))
        f = go.Figure([go.Scatter(x=growth.index, y=growth[k], name=k, line=dict(color=clr, width=w_, dash=dash))
                       for k, clr, w_, dash in styles])
        f.update_layout(title=f"Growth of {usd(wealth)}", yaxis_title="USD", yaxis_type="log")
        st.plotly_chart(fig_layout(f, 430), width="stretch")

        reb = bt.rebalances[bt.rebalances["status"] == "ok"]
        left, right = st.columns(2, gap="large")
        f = go.Figure([go.Scatter(x=reb.index, y=reb["ex_ante_vol"] * 100, name="Ex-ante (promised)",
                                  line=dict(color=NAVY, width=3)),
                       go.Scatter(x=reb.index, y=reb["realised_vol"] * 100, name="Realised (next 12 months)",
                                  line=dict(color=ACCENT, width=2))])
        f.update_layout(title="Volatility promised vs realised at each rebalancing", yaxis_title="% p.a.")
        left.plotly_chart(fig_layout(f, 380), width="stretch")
        left.caption(f"Average ex-ante volatility {reb['ex_ante_vol'].mean():.1%}; average realised over the next "
                     f"12 months {reb['realised_vol'].mean():.1%}; realised over the whole backtest "
                     f"{metrics.loc['Strategy', 'volatility']:.1%}.")
        dd = pd.DataFrame({k: risk.drawdowns(series[k]) for k in series}) * 100
        f = go.Figure([go.Scatter(x=dd.index, y=dd[k], name=k, line=dict(color=clr, dash=dash))
                       for k, clr, _, dash in styles])
        f.update_layout(title="Drawdowns", yaxis_title="%")
        right.plotly_chart(fig_layout(f, 380), width="stretch")

        alloc = bt.weights.T.groupby(universe.loc[bt.weights.columns, "asset_class"]).sum().T.reindex(columns=CLASS_ORDER)
        alloc.columns = alloc.columns.map(CLASS_LABELS)
        f = go.Figure([go.Scatter(x=alloc.index, y=alloc[c] * 100, name=c, stackgroup="one", mode="none",
                                  fillcolor=clr) for c, clr in zip(alloc.columns, [NAVY, "#7f9cb8", "#c9a227", "#d9d9d9"])])
        f.update_layout(title="Allocation by asset class over time (month-end, after drift)", yaxis_title="%",
                        yaxis_range=[0, 100])
        st.plotly_chart(fig_layout(f, 360), width="stretch")
with tabs[4], guarded("Simulation"):
    st.subheader(f"Wealth simulation — {profile} profile, starting with {usd(wealth)}")
    if require_ok(res):
        years = st.slider("Horizon (years)", 5, 20, 10, key="sim.years")
        loss_limit = prof.MAX_LOSS_LEVELS[answers["max_loss"]]
        try:
            sim, h_start, h_end, n_hist = simulate(profile, cons_json, st.session_state.get("bt.freq", "Quarterly"),
                                                   confidence, years, float(wealth), loss_limit)
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
            for name, clr, fill in (("Benchmark", GREY, "rgba(163,173,184,0.25)"), ("Portfolio", NAVY, "rgba(31,59,87,0.20)")):
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
                       "portfolio was chosen using only data available at that date, and trading costs are "
                       "deducted. Management fees and taxes are ignored. The simulation still inherits the "
                       "survivorship bias of the stock list (today's winners, picked with hindsight), so the "
                       "portfolio's outcomes are somewhat optimistic; the benchmark's are not.")

# --- About Us --------------------------------------------------------------------------------

def avatar_html(member: team.Member, size: int = 112) -> str:
    """Round photo, or a round avatar with the initials when the photo is missing."""
    uri = team.photo_data_uri(member)
    style = f"width:{size}px;height:{size}px;border-radius:50%;margin:0.25rem auto 0.75rem auto;display:block;"
    if uri:
        return f'<img src="{uri}" alt="{html.escape(member.name)}" style="{style}object-fit:cover;">'
    return (f'<div style="{style}background:{NAVY};color:#ffffff;display:flex;align-items:center;'
            f'justify-content:center;font-size:{size // 3}px;font-weight:600;letter-spacing:1px;">'
            f'{html.escape(team.initials(member.name))}</div>')


with tabs[5], guarded("About Us"):
    st.subheader("About this project")
    st.markdown(
        "We play the role of an **international private bank** advising clients whose reference currency is "
        "the US dollar. The app turns a short suitability questionnaire into a risk profile, then builds the "
        "portfolio with the highest expected return that stays within the client's volatility budget and "
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
                            f'<div style="text-align:center;color:#4a5560;font-size:0.9rem;">{html.escape(m.description)}</div>',
                            unsafe_allow_html=True)
                if m.linkedin:
                    st.markdown(f'<div style="text-align:center;margin-top:0.5rem;"><a href="{html.escape(m.linkedin)}" '
                                f'target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
    if repository:
        st.markdown(f"Source code: [{repository}]({repository})")
