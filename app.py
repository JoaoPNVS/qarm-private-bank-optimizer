"""Swiss Private Client Optimizer — Streamlit UI only (no heavy logic here)."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data, estimators as est, optimizer as opt, profiling as prof

st.set_page_config(page_title="Swiss Private Client Optimizer", layout="wide")

NAVY, GREY, ACCENT = "#1f3b57", "#a3adb8", "#c0392b"
CLASS_ORDER = ["Equity", "Bond", "Alternative", "Real Estate", "Cash"]
UI_CONSTRAINTS = ["max_weight", "asset_class_bands", "region_bands", "em_equity_cap", "home_bias",
                  "max_non_chf", "exclusions", "max_volatility", "max_tracking_error"]
EXPECTED_ERRORS = (data.DataError, est.EstimationError, opt.OptimizationError, ValueError)


# --- helpers ---------------------------------------------------------------------------------

def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def fig_layout(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(size=13))
    return fig


@st.cache_data(show_spinner=False)
def load_data():
    return data.load_universe(), data.load_returns(), data.load_rf()


@st.cache_data(show_spinner=False)
def get_estimates(months: int, views_json: str) -> est.Estimates:
    universe, returns, rf = load_data()
    views = [est.View(**v) for v in json.loads(views_json)]
    window = est.estimation_window(returns, returns.index[-1], months)
    return est.estimate(window, rf, views=views, universe=universe)


def build_constraints(profile: str, spec: dict) -> list[opt.Constraint]:
    cons = [c for c in prof.default_constraints(profile) if c.key in spec]
    for c in cons:
        c.params, c.active = spec[c.key]["params"], spec[c.key]["active"]
    return cons


def problem(profile: str, months: int, views_json: str) -> opt.ProblemData:
    return prof.problem_data(profile, get_estimates(months, views_json), load_data()[0])


@st.cache_data(show_spinner="Optimising the portfolio...")
def optimise(profile, months, views_json, cons_json):
    d = problem(profile, months, views_json)
    res = opt.solve(d, build_constraints(profile, json.loads(cons_json)))
    return res, opt.portfolio_stats(d.benchmark, d), d.benchmark, d.risk_aversion


@st.cache_data(show_spinner="Computing the cost of each constraint...")
def constraint_costs(profile, months, views_json, cons_json, wealth):
    d = problem(profile, months, views_json)
    return opt.cost_of_constraints(d, build_constraints(profile, json.loads(cons_json)), wealth)


@st.cache_data(show_spinner="Computing efficient frontiers...")
def frontiers(profile, months, views_json, cons_json):
    d = problem(profile, months, views_json)
    return (opt.efficient_frontier(d, build_constraints(profile, json.loads(cons_json))),
            opt.efficient_frontier(d, []))


def pct_slider(label, default, lo, hi, step, key, disabled=False):
    fmt = "%.1f%%" if step < 1 else "%d%%"
    cast = float if step < 1 else int
    value = cast(round(default * 100, 1))
    return st.slider(label, cast(lo), cast(hi), value, cast(step), format=fmt, key=key, disabled=disabled) / 100


def band_slider(label, lo, hi, key, disabled=False):
    a, b = st.slider(label, 0, 100, (int(round(lo * 100)), int(round(hi * 100))), format="%d%%",
                     key=key, disabled=disabled)
    return [a / 100, b / 100]


def require_ok(res: opt.OptimizationResult) -> bool:
    if not res.ok:
        st.error(res.message)
        st.info("Adjust the constraints in the **Constraints** tab.")
    return res.ok


# --- data & sidebar --------------------------------------------------------------------------

try:
    universe, returns, rf = load_data()
except data.DataError as exc:
    st.error(f"Market data unavailable: {exc}")
    st.stop()

names = universe["name"].to_dict()
label = lambda t: f"{names.get(t, t)} ({t})"  # noqa: E731

answers = {q.key: st.session_state.get(f"q.{q.key}", 1) for q in prof.QUESTIONS}
assessment = prof.assess(answers)

with st.sidebar:
    st.title("Swiss Private Client Optimizer")
    st.caption("Constrained portfolio construction for a CHF-based private client.")
    wealth = st.number_input("Client wealth (CHF)", min_value=100_000, max_value=1_000_000_000,
                             value=5_000_000, step=250_000, format="%d")
    months = st.slider("Estimation window (months)", 36, 84, 60, step=6,
                       help="Months of history used to estimate volatilities and correlations.")
    choice = st.selectbox("Risk profile", ["auto"] + prof.PROFILES, key="profile",
                          format_func=lambda p: f"From questionnaire ({assessment.profile})" if p == "auto" else p)
    profile = assessment.profile if choice == "auto" else choice
    st.markdown(f"**{profile}** — {prof.SPECS[profile].description}")
    st.divider()
    st.caption(f"Data: monthly total returns in CHF up to {returns.index[-1]:%B %Y} (yfinance). "
               f"Risk-free: SARON (SNB), currently {rf.iloc[-1] * 12:.2%} p.a.")

tabs = st.tabs(["Client Profile", "Constraints", "House Views", "Optimal Portfolio",
                "Cost of Constraints", "Efficient Frontier"])

# --- Client Profile --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Suitability questionnaire (LSFin / FIDLEG)")
    left, right = st.columns([3, 2], gap="large")
    with left:
        for q in prof.QUESTIONS:
            st.radio(q.text, list(range(len(q.options))), format_func=lambda i, q=q: q.options[i],
                     key=f"q.{q.key}", index=1)
    with right:
        a = prof.assess({q.key: st.session_state[f"q.{q.key}"] for q in prof.QUESTIONS})
        with st.container(border=True):
            st.markdown("#### Resulting profile")
            st.markdown(f"## {a.profile}")
            st.caption(prof.SPECS[a.profile].description)
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk capacity", f"{a.dimensions['capacity']:.2f} / 4", a.levels["capacity"], delta_color="off")
            c2.metric("Risk willingness", f"{a.dimensions['willingness']:.2f} / 4", a.levels["willingness"],
                      delta_color="off")
            c3.metric("Knowledge", f"{a.dimensions['knowledge']:.0f} / 4")
            st.markdown("**Why this profile**")
            for r in a.reasons:
                st.markdown(f"- {r}")
        if choice != "auto" and profile != a.profile:
            st.warning(f"The sidebar overrides the questionnaire: the **{profile}** profile is used "
                       f"instead of {a.profile}.")
        st.caption("Scoring: each answer scores 1-4. Capacity and willingness averages map to a profile "
                   "(<1.75 Conservative, <2.50 Balanced, <3.25 Growth, otherwise Dynamic); the lower one is "
                   "kept, then capped by knowledge. A horizon below 3 years always gives Conservative.")

# --- Constraints -----------------------------------------------------------------------------

spec_p = prof.SPECS[profile]
defaults = {c.key: c for c in prof.default_constraints(profile)}
cons_spec: dict = {}

with tabs[1]:
    st.subheader(f"Client constraints — defaults of the {profile} profile")
    status_box = st.container()
    k = lambda name: f"{profile}.{name}"  # noqa: E731  (each profile keeps its own settings)
    col_a, col_b = st.columns(2, gap="large")

    def card(key, column, render):
        c = defaults[key]
        with column.container(border=True):
            active = st.toggle(c.name, value=True, key=k(f"{key}.on"))
            params = render(not active)
            cons_spec[key] = {"active": active, "params": params}
            st.caption(c.describe(params))

    card("max_weight", col_a, lambda off: {"limit": pct_slider(
        "Max weight per asset", spec_p.max_weight, 5, 50, 1, k("max_weight"), off)})
    card("asset_class_bands", col_a, lambda off: {"bands": {
        cls: band_slider(cls, lo, hi, k(f"band.{cls}"), off) for cls, (lo, hi) in spec_p.asset_bands.items()}})
    card("region_bands", col_a, lambda off: {"bands": {
        r: band_slider(f"{r} (share of equities)", lo, hi, k(f"region.{r}"), off)
        for r, (lo, hi) in prof.REGION_BANDS.items()}})
    card("em_equity_cap", col_b, lambda off: {"ratio": pct_slider(
        "Max EM share of equities", prof.EM_EQUITY_CAP, 0, 100, 1, k("em"), off)})
    card("home_bias", col_b, lambda off: {"min_share": pct_slider(
        "Min Swiss share of equities", spec_p.home_bias, 0, 100, 1, k("home"), off)})
    card("max_non_chf", col_b, lambda off: {"limit": pct_slider(
        "Max non-CHF exposure", spec_p.max_non_chf, 0, 100, 1, k("fx"), off)})
    card("exclusions", col_b, lambda off: {"tickers": st.multiselect(
        "Excluded assets", [t for t in universe.index if t != data.CASH_TICKER], default=[],
        format_func=label, key=k("excl"), disabled=off)})
    card("max_volatility", col_b, lambda off: {"limit": pct_slider(
        "Max volatility", spec_p.max_volatility, 2, 25, 0.5, k("vol"), off)})
    card("max_tracking_error", col_b, lambda off: {"limit": pct_slider(
        "Max tracking error vs benchmark", spec_p.max_tracking_error, 0.5, 10, 0.5, k("te"), off)})
    st.caption("Turnover limits and transaction costs apply once current holdings are entered (next version).")

# --- House Views -----------------------------------------------------------------------------

with tabs[2]:
    st.subheader("CIO house views (Black-Litterman)")
    use_views = st.toggle("Apply house views", value=True, key="views.on")
    views = []
    for i, v in enumerate(est.HOUSE_VIEWS):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1.2, 1.2])
            on = c1.checkbox(v.name, value=True, key=f"view.{i}.on", disabled=not use_views)
            kind = "Relative view (long - short)" if v.short else "Absolute view (excess over SARON)"
            c1.caption(kind)
            value = c2.slider("View (% p.a.)", -5.0, 5.0, v.value * 100, 0.25, key=f"view.{i}.value",
                              disabled=not (use_views and on)) / 100
            conf = c3.slider("Confidence", 5, 100, int(v.confidence * 100), 5, format="%d%%",
                             key=f"view.{i}.conf", disabled=not (use_views and on)) / 100
            views.append(replace(v, value=value, confidence=conf, active=use_views and on))
    views_json = json.dumps([asdict(v) for v in views], sort_keys=True)
    st.caption("Posterior = equilibrium blended with the views. Confidence 50% is the standard He-Litterman "
               "uncertainty, 100% imposes the view exactly. Without views the equilibrium returns are used.")

try:
    estimates = get_estimates(months, views_json)
except EXPECTED_ERRORS as exc:
    st.error(f"Could not estimate expected returns and risks: {exc}")
    st.stop()

with tabs[2]:
    if not estimates.views.empty:
        st.markdown("#### Effect on each view")
        vt = estimates.views.rename(columns={"view": "View", "confidence": "Confidence",
                                             "prior": "Equilibrium", "posterior": "Posterior"})
        st.dataframe(vt.style.format("{:.2%}"), width="stretch")
    mu = pd.DataFrame({"Asset": [label(t) for t in estimates.mu.index],
                       "Asset class": universe.loc[estimates.mu.index, "asset_class"].values,
                       "Equilibrium": estimates.prior.values, "Black-Litterman": estimates.mu.values},
                      index=estimates.mu.index).drop(data.CASH_TICKER)
    mu["Change"] = mu["Black-Litterman"] - mu["Equilibrium"]
    left, right = st.columns([3, 2], gap="large")
    left.markdown("#### Expected excess returns (% p.a.)")
    left.dataframe(mu.sort_values(["Asset class", "Equilibrium"], ascending=[True, False]).style.format(
        {"Equilibrium": "{:.2%}", "Black-Litterman": "{:.2%}", "Change": "{:+.2%}"}),
        hide_index=True, width="stretch", height=520)
    ch = mu.sort_values("Change")
    fig = go.Figure(go.Bar(x=ch["Change"] * 100, y=ch.index, orientation="h",
                           marker_color=[NAVY if x >= 0 else ACCENT for x in ch["Change"]]))
    right.markdown("#### Change vs equilibrium (pp)")
    right.plotly_chart(fig_layout(fig, 560), width="stretch")

# --- optimisation ----------------------------------------------------------------------------

cons_json = json.dumps(cons_spec, sort_keys=True)
try:
    res, bench_stats, bench_w, lam = optimise(profile, months, views_json, cons_json)
except EXPECTED_ERRORS as exc:
    st.error(f"Optimisation failed: {exc}")
    st.stop()

with status_box:
    if res.ok:
        n_active = res.duals.loc[res.duals["active"], "key"].nunique()
        st.success(f"All active constraints are satisfied. {n_active} of them are binding "
                   "(see the Cost of Constraints tab).")
    else:
        st.error(res.message)
        if res.diagnosis:
            st.markdown("**Constraints in conflict:** " + ", ".join(defaults[k_].name for k_ in res.diagnosis.conflicting))

# --- Optimal Portfolio -----------------------------------------------------------------------

with tabs[3]:
    st.subheader(f"Optimal portfolio — {profile} profile")
    if require_ok(res):
        w = res.weights
        c = st.columns(5)
        c[0].metric("Expected excess return", pct(res.expected_return, 2),
                    f"{(res.expected_return - bench_stats['expected_return']) * 1e4:+.0f} bps vs benchmark")
        c[1].metric("Volatility", pct(res.volatility, 2),
                    f"{(res.volatility - bench_stats['volatility']) * 100:+.2f} pp vs benchmark", delta_color="inverse")
        c[2].metric("Sharpe ratio", f"{res.sharpe:.2f}", f"{res.sharpe - bench_stats['sharpe']:+.2f} vs benchmark")
        c[3].metric("Tracking error", pct(res.tracking_error, 2))
        c[4].metric("Risk aversion λ", f"{lam:.2f}")
        st.caption("Returns are expected excess returns over SARON, in CHF, annualised.")

        both = pd.DataFrame({"Portfolio": w, "Benchmark": bench_w})
        shown = both[(both > 0.005).any(axis=1)].sort_values("Portfolio")
        fig = go.Figure([go.Bar(name="Portfolio", y=[label(t) for t in shown.index], x=shown["Portfolio"] * 100,
                                orientation="h", marker_color=NAVY),
                         go.Bar(name="Benchmark", y=[label(t) for t in shown.index], x=shown["Benchmark"] * 100,
                                orientation="h", marker_color=GREY)])
        fig.update_layout(barmode="group", xaxis_title="Weight (%)")
        st.markdown("#### Weights vs profile benchmark")
        st.plotly_chart(fig_layout(fig, max(400, 26 * len(shown))), width="stretch")

        def breakdown(col, title, order=None):
            t = both.groupby(universe.loc[both.index, col]).sum()
            t = t.reindex(order).dropna(how="all") if order else t.sort_values("Portfolio", ascending=False)
            f = go.Figure([go.Bar(name=n, x=t.index, y=t[n] * 100, marker_color=clr)
                           for n, clr in (("Portfolio", NAVY), ("Benchmark", GREY))])
            f.update_layout(barmode="group", yaxis_title="%", title=title)
            return f

        b1, b2, b3 = st.columns(3)
        b1.plotly_chart(fig_layout(breakdown("asset_class", "By asset class", CLASS_ORDER), 340), width="stretch")
        b2.plotly_chart(fig_layout(breakdown("region", "By region"), 340), width="stretch")
        b3.plotly_chart(fig_layout(breakdown("currency", "By currency exposure"), 340), width="stretch")
        nonchf = both[universe.loc[both.index, "non_chf"].values].sum()
        st.markdown(f"**Non-CHF exposure:** portfolio {pct(nonchf['Portfolio'])} — benchmark {pct(nonchf['Benchmark'])}")

        with st.expander("Full allocation in CHF"):
            table = pd.DataFrame({"Asset": [label(t) for t in w.index],
                                  "Asset class": universe.loc[w.index, "asset_class"].values,
                                  "Weight": w.values, "Amount (CHF)": w.values * wealth,
                                  "Benchmark": bench_w.values}, index=w.index)
            st.dataframe(table[table["Weight"] > 1e-4].sort_values("Weight", ascending=False).style.format(
                {"Weight": "{:.2%}", "Benchmark": "{:.2%}", "Amount (CHF)": lambda x: chf(x)}),
                hide_index=True, width="stretch")

# --- Cost of Constraints ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("What each constraint costs the client")
    if require_ok(res):
        try:
            costs = constraint_costs(profile, months, views_json, cons_json, float(wealth))
        except EXPECTED_ERRORS as exc:
            st.error(f"Could not compute constraint costs: {exc}")
            costs = None
        if costs is not None:
            st.caption(f"Each active constraint is removed in turn and the portfolio re-optimised. The cost is the "
                       f"certainty-equivalent return given up, in basis points and in CHF per year on {chf(wealth)}.")
            view = costs[["constraint", "description", "utility_cost_bps", "utility_cost_chf",
                          "sharpe_with", "sharpe_without"]].rename(columns={
                "constraint": "Constraint", "description": "Rule", "utility_cost_bps": "Cost (bps p.a.)",
                "utility_cost_chf": "Cost (CHF p.a.)", "sharpe_with": "Sharpe with", "sharpe_without": "Sharpe without"})
            st.dataframe(view.style.format({"Cost (bps p.a.)": "{:.2f}", "Cost (CHF p.a.)": lambda x: chf(x),
                                            "Sharpe with": "{:.3f}", "Sharpe without": "{:.3f}"}),
                         hide_index=True, width="stretch")
            srt = costs.sort_values("utility_cost_chf")
            fig = go.Figure(go.Bar(x=srt["utility_cost_chf"], y=srt["constraint"], orientation="h",
                                   marker_color=NAVY, text=[chf(x) for x in srt["utility_cost_chf"]],
                                   textposition="outside"))
            fig.update_layout(xaxis_title="CHF per year")
            st.plotly_chart(fig_layout(fig, 60 + 38 * len(srt)), width="stretch")
        active = res.duals[res.duals["active"]]
        st.markdown("#### Binding constraints (shadow prices)")
        st.caption("Gain in certainty-equivalent return (bps p.a.) if the bound were relaxed by 1 percentage point.")
        st.dataframe(active[["constraint", "label", "bps_per_pp"]].rename(columns={
            "constraint": "Constraint", "label": "Bound", "bps_per_pp": "bps per +1pp"}).style.format(
            {"bps per +1pp": "{:.3f}"}), hide_index=True, width="stretch")

# --- Efficient Frontier ----------------------------------------------------------------------

with tabs[5]:
    st.subheader("Efficient frontier: client constraints vs long-only")
    if require_ok(res):
        fr_c, fr_u = frontiers(profile, months, views_json, cons_json)
        fig = go.Figure()
        for fr, name, clr, dash in ((fr_u, "Long-only (no client constraints)", GREY, "dash"),
                                    (fr_c, "With client constraints", NAVY, "solid")):
            if not fr.points.empty:
                fig.add_trace(go.Scatter(x=fr.points["volatility"] * 100, y=fr.points["expected_return"] * 100,
                                         mode="lines", name=name, line=dict(color=clr, dash=dash, width=3)))
        fig.add_trace(go.Scatter(x=[res.volatility * 100], y=[res.expected_return * 100], mode="markers",
                                 name="Client portfolio", marker=dict(color=ACCENT, size=14, symbol="star")))
        fig.add_trace(go.Scatter(x=[bench_stats["volatility"] * 100], y=[bench_stats["expected_return"] * 100],
                                 mode="markers", name="Profile benchmark", marker=dict(color=GREY, size=12)))
        fig.update_layout(xaxis_title="Volatility (% p.a.)", yaxis_title="Expected excess return (% p.a.)")
        st.plotly_chart(fig_layout(fig, 480), width="stretch")
        if not fr_c.weights.empty:
            st.markdown("#### Allocation along the constrained frontier")
            by_class = fr_c.weights.T.groupby(universe.loc[fr_c.weights.columns, "asset_class"]).sum().T
            by_class.index = (fr_c.points["volatility"] * 100).round(2)
            long = by_class.reset_index(names="vol").melt(id_vars="vol", var_name="Asset class", value_name="w")
            area = px.area(long, x="vol", y="w", color="Asset class", category_orders={"Asset class": CLASS_ORDER},
                           color_discrete_sequence=[NAVY, "#4f7396", "#8fa9c2", "#c9b79c", "#d9d9d9"])
            area.update_layout(xaxis_title="Volatility (% p.a.)", yaxis_title="Weight", yaxis_tickformat=".0%")
            st.plotly_chart(fig_layout(area, 380), width="stretch")
