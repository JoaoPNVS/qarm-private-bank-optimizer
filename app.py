"""Private Client Portfolio Optimizer — Streamlit UI only (no heavy logic here)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest as btm, benchmarks as bm, data, estimators as est, optimizer as opt, profiling as prof, risk

st.set_page_config(page_title="Private Client Portfolio Optimizer", layout="wide")

NAVY, GREY, ACCENT = "#1f3b57", "#a3adb8", "#c0392b"
CLASS_ORDER = ["Equity", "Bond", "Commodity", "Cash"]
BOND_TYPES = ["Short-term", "Treasury 7-10y", "Corporate IG", "High Yield", "TIPS"]
WINDOW = est.DEFAULT_WINDOW
EXPECTED_ERRORS = (data.DataError, est.EstimationError, btm.BacktestError, ValueError)


# --- helpers ---------------------------------------------------------------------------------

def usd(x: float) -> str:
    return f"USD {x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def bench_label(profile: str) -> str:
    e = prof.SPECS[profile].benchmark_equity
    return "Benchmark (100% SPY)" if e >= 1 else f"Benchmark ({e:.0%} SPY / {1 - e:.0%} AGG)"


def fig_layout(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(size=13))
    return fig


@st.cache_data(show_spinner=False)
def load_data():
    return data.load_universe(), data.load_returns(), data.load_rf(), data.load_benchmark_returns()


@st.cache_data(show_spinner=False)
def get_estimates() -> est.Estimates:
    universe, returns, rf, bench = load_data()
    window = est.estimation_window(returns, returns.index[-1], WINDOW)
    return est.estimate(window, rf, universe["asset_class"], benchmarks=bench, benchmark_class=bm.BENCHMARK_CLASS)


def build_constraints(profile: str, spec: dict) -> list[opt.Constraint]:
    cons = [c for c in prof.default_constraints(profile) if c.key in spec]
    for c in cons:
        c.params, c.active = spec[c.key]["params"], spec[c.key]["active"]
    return cons


@st.cache_data(show_spinner="Optimising the portfolio...")
def optimise(profile: str, cons_json: str):
    e = get_estimates()
    d = prof.problem_data(profile, e, load_data()[0])
    res = opt.solve(d, build_constraints(profile, json.loads(cons_json)))
    eq = prof.SPECS[profile].benchmark_equity
    te = bm.tracking_error(res.weights, eq, e) if res.ok else np.nan
    return res, bm.ex_ante_stats(eq, e), bm.ex_ante_stats(1.0, e), te


@st.cache_data(show_spinner=False)
def backtest_estimates() -> dict:
    """Estimates at every quarter-end of the backtest (semi-annual/annual dates are a subset)."""
    universe, returns, rf, _ = load_data()
    dates = btm.rebalance_dates(returns.index, btm.DEFAULT_START, "Quarterly")
    return btm.estimates_by_date(returns, rf, universe["asset_class"], dates)


@st.cache_data(show_spinner="Running the backtest (re-optimising at every rebalancing)...")
def run_backtest(profile: str, cons_json: str, freq: str, tc: float):
    universe, returns, rf, bench = load_data()
    res = btm.run_backtest(returns, rf, universe, build_constraints(profile, json.loads(cons_json)),
                           freq=freq, tc=tc, estimates=backtest_estimates())
    idx = res.returns.index
    series = pd.DataFrame({"Strategy": res.returns,
                           bench_label(profile): bm.profile_benchmark_returns(prof.SPECS[profile].benchmark_equity,
                                                                             bench).reindex(idx),
                           "SPY": bench["SPY"].reindex(idx)})
    return res, series


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
    st.caption("Highest expected return within the client's risk budget. All figures in USD.")
    wealth = st.number_input("Client wealth (USD)", min_value=100_000, max_value=1_000_000_000,
                             value=5_000_000, step=250_000, format="%d")
    choice = st.selectbox("Risk profile", ["auto"] + prof.PROFILES, key="profile",
                          format_func=lambda p: f"From questionnaire ({assessment.profile})" if p == "auto" else p)
    profile = assessment.profile if choice == "auto" else choice
    spec_p = prof.SPECS[profile]
    st.markdown(f"**{profile}** — {spec_p.description}")
    st.markdown(f"Risk budget: **{spec_p.max_volatility:.0%}** volatility · bonds at least "
                f"**{spec_p.min_bonds:.0%}** · benchmark **{spec_p.benchmark_equity:.0%} SPY / "
                f"{1 - spec_p.benchmark_equity:.0%} AGG**")
    st.divider()
    st.caption(f"Data: monthly total returns in USD up to {returns.index[-1]:%B %Y} (yfinance). "
               f"Estimates use the last {WINDOW} months. Risk-free: 3-month T-bill, currently "
               f"{rf.iloc[-1] * 12:.2%} p.a.")

tabs = st.tabs(["Client Profile", "Constraints", "Portfolio", "Backtest", "Simulation"])

# --- Client Profile --------------------------------------------------------------------------

with tabs[0]:
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
            c1.metric("Risk capacity", f"{a.dimensions['capacity']:.2f} / 4", a.levels["capacity"], delta_color="off")
            c2.metric("Risk willingness", f"{a.dimensions['willingness']:.2f} / 4", a.levels["willingness"],
                      delta_color="off")
            c3.metric("Knowledge", f"{a.dimensions['knowledge']:.0f} / 4")
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

with tabs[1]:
    st.subheader(f"Client constraints — defaults of the {profile} profile")
    status_box = st.container()
    k = lambda name: f"{profile}.{name}"  # noqa: E731  (each profile keeps its own settings)
    col_a, col_b = st.columns(2, gap="large")

    def card(key, column, render, mandatory=False):
        c = defaults[key]
        with column.container(border=True):
            if mandatory:
                st.markdown(f"**{c.name}** (risk budget of the profile)")
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
        "Max weight per ETF", prof.MAX_ETF_WEIGHT, 5, 100, 5, k("etf"), off)})
    card("max_sector_weight", col_b, lambda off: {"limit": pct_slider(
        "Max weight per equity sector", prof.MAX_SECTOR_WEIGHT, 5, 100, 5, k("sector"), off)})
    card("exclusions", col_b, lambda off: {
        "sectors": st.multiselect("Excluded sectors", sectors, default=[], key=k("excl.sectors"), disabled=off),
        "tickers": st.multiselect("Excluded companies", stock_tickers, default=[], format_func=label,
                                  key=k("excl.tickers"), disabled=off)})
    st.caption("Emerging markets = the EEM fund plus PDD (China) and MELI (Latin America). International "
               "equity funds (VGK, EWJ, EEM) count as one sector. Bond funds and gold are not subject to "
               "the sector limit.")

# --- optimisation ----------------------------------------------------------------------------

cons_json = json.dumps(cons_spec, sort_keys=True)
try:
    res, bench_stats, spy_stats, te = optimise(profile, cons_json)
except EXPECTED_ERRORS as exc:
    st.error(f"The portfolio could not be computed: {exc}")
    st.stop()

with status_box:
    if res.ok:
        binding = res.duals.loc[res.duals["active"], "constraint"].unique().tolist()
        st.success("All active constraints are satisfied. Binding: " + (", ".join(binding) or "none") + ".")
    else:
        st.error(res.message)

# --- Portfolio -------------------------------------------------------------------------------

with tabs[2]:
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
        st.caption(f"Ex-ante figures estimated on the last {WINDOW} months: historical means shrunk 80% toward "
                   "their asset-class average, Ledoit-Wolf covariance. Past winners still dominate the estimates; "
                   "see the Backtest tab for what such portfolios actually delivered.")

        b_eq = spec_p.benchmark_equity
        by_class = pd.DataFrame({"Portfolio": w.groupby(uni["asset_class"]).sum(),
                                 "Benchmark": pd.Series({"Equity": b_eq, "Bond": 1 - b_eq})}).reindex(CLASS_ORDER).fillna(0)
        is_bond = uni["asset_class"] == "Bond"
        by_type = w[is_bond].groupby(uni.loc[is_bond, "bond_type"]).sum().reindex(BOND_TYPES).fillna(0)
        is_eq = uni["asset_class"] == "Equity"
        by_sector = w[is_eq].groupby(uni.loc[is_eq, "sector"]).sum().sort_values(ascending=False)
        b1, b2, b3 = st.columns(3)
        b1.plotly_chart(fig_layout(grouped_bars(by_class, "By asset class"), 340), width="stretch")
        f = go.Figure(go.Bar(x=by_type.index, y=by_type * 100, marker_color=NAVY))
        f.update_layout(title=f"Bonds by type ({by_type.sum():.0%} of the portfolio)", yaxis_title="% of portfolio")
        b2.plotly_chart(fig_layout(f, 340), width="stretch")
        f = go.Figure(go.Bar(x=by_sector.index, y=by_sector * 100, marker_color=NAVY))
        f.update_layout(title=f"Equities by sector ({by_sector.sum():.0%} of the portfolio)", yaxis_title="% of portfolio")
        b3.plotly_chart(fig_layout(f, 340), width="stretch")
        em = w[is_eq & uni["is_EM"]].sum()
        st.markdown(f"**Emerging markets:** {pct(em)} of the portfolio, {pct(em / max(w[is_eq].sum(), 1e-12))} of equities. "
                    f"**Cash:** {pct(w[uni['asset_class'] == 'Cash'].sum())}. **Positions above 0.5%:** {(w > 0.005).sum()}.")

        held = w[w > 0.005].sort_values()
        f = go.Figure(go.Bar(y=[label(t) for t in held.index], x=held * 100, orientation="h",
                             marker_color=[GREY if universe.at[t, "vehicle"] != "Stock" else NAVY for t in held.index]))
        f.update_layout(xaxis_title="Weight (%)", title="Holdings (funds and cash in grey)")
        st.plotly_chart(fig_layout(f, max(400, 24 * len(held))), width="stretch")

        with st.expander("Full allocation in USD"):
            table = pd.DataFrame({"Asset": [label(t) for t in w.index], "Class": uni["asset_class"].values,
                                  "Sector / type": np.where(is_bond, uni["bond_type"], uni["sector"]),
                                  "Weight": w.values, "Amount (USD)": w.values * wealth}, index=w.index)
            st.dataframe(table[table["Weight"] > 1e-4].sort_values("Weight", ascending=False).style.format(
                {"Weight": "{:.2%}", "Amount (USD)": lambda x: usd(x)}), hide_index=True, width="stretch")
        with st.expander("Binding constraints"):
            act = res.duals[res.duals["active"]]
            st.caption("Gain in expected return (bps p.a.) if the bound were relaxed by 1 percentage point.")
            st.dataframe(act[["constraint", "label", "bps_per_pp"]].rename(columns={
                "constraint": "Constraint", "label": "Bound", "bps_per_pp": "bps per +1pp"}).style.format(
                {"bps per +1pp": "{:.1f}"}), hide_index=True, width="stretch")

with tabs[3]:
    st.subheader(f"Backtest — {profile} profile with the client constraints")
    c1, c2, _ = st.columns([1, 1, 2])
    freq = c1.selectbox("Rebalancing", list(btm.FREQ_MONTHS), key="bt.freq")
    tc_bps = c2.number_input("Transaction costs (bps per unit traded)", 0, 100, 10, step=5, key="bt.tc")
    try:
        bt, series = run_backtest(profile, cons_json, freq, tc_bps / 1e4)
    except EXPECTED_ERRORS as exc:
        st.error(f"The backtest could not run: {exc}")
        bt = None
    if bt is not None:
        start, end = series.index[0], series.index[-1]
        st.caption(f"{start:%B %Y} – {end:%B %Y} ({len(series) / 12:.1f} years). At each rebalancing the portfolio "
                   f"is re-optimised on the previous {WINDOW} months only (no look-ahead); weights drift in between. "
                   "**Survivorship bias:** the stock universe is today's index membership, so the strategy's "
                   "results are an upper bound; SPY and the benchmark are not affected.")
        for ev in bt.events:
            st.warning(ev)
        metrics = pd.DataFrame({k: risk.performance_metrics(series[k], rf) for k in series}).T
        metrics["Final wealth (USD)"] = [wealth * (1 + series[k]).prod() for k in series]
        metrics["Turnover p.a."] = [btm.annual_turnover(bt), np.nan, np.nan]
        st.dataframe(metrics.rename(columns={"annual_return": "Annual return", "volatility": "Volatility",
                                             "sharpe": "Sharpe ratio", "max_drawdown": "Max drawdown"}).style.format(
            {"Annual return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe ratio": "{:.2f}", "Max drawdown": "{:.1%}",
             "Final wealth (USD)": lambda x: usd(x), "Turnover p.a.": lambda x: "" if pd.isna(x) else f"{x:.0%}"}),
            width="stretch")
        if spec_p.benchmark_equity >= 1:
            st.caption("The Dynamic profile's benchmark is 100% SPY, so its row is identical to SPY.")

        growth = wealth * (1 + series).cumprod()
        f = go.Figure([go.Scatter(x=growth.index, y=growth[k], name=k, line=dict(color=clr, width=w_, dash=dash))
                       for k, clr, w_, dash in (("Strategy", NAVY, 3, "solid"), (bench_label(profile), GREY, 2, "solid"),
                                                ("SPY", ACCENT, 2, "dot"))])
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
        f = go.Figure([go.Scatter(x=dd.index, y=dd[k], name=k, line=dict(color=clr))
                       for k, clr in (("Strategy", NAVY), (bench_label(profile), GREY), ("SPY", ACCENT))])
        f.update_layout(title="Drawdowns", yaxis_title="%")
        right.plotly_chart(fig_layout(f, 380), width="stretch")

        alloc = bt.weights.T.groupby(universe.loc[bt.weights.columns, "asset_class"]).sum().T.reindex(columns=CLASS_ORDER)
        f = go.Figure([go.Scatter(x=alloc.index, y=alloc[c] * 100, name=c, stackgroup="one", mode="none",
                                  fillcolor=clr) for c, clr in zip(CLASS_ORDER, [NAVY, "#7f9cb8", "#c9a227", "#d9d9d9"])])
        f.update_layout(title="Allocation by asset class over time (month-end, after drift)", yaxis_title="%",
                        yaxis_range=[0, 100])
        st.plotly_chart(fig_layout(f, 360), width="stretch")
with tabs[4]:
    st.info("The wealth simulation will be added in the next step.")
