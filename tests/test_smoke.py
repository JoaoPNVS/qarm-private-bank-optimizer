"""App scenarios with Streamlit's AppTest (slow: each one runs the whole app, backtest included)."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 300   # a new profile or constraint re-runs the backtest (slow machines: minutes)


def run_app(widgets: dict | None = None) -> AppTest:
    """Run the app, then set each {(widget_kind, key): value} in order, rerunning after each one
    (widgets of a profile only exist once that profile is selected)."""
    at = AppTest.from_file(APP_PATH).run(timeout=TIMEOUT)
    for (kind, key), value in (widgets or {}).items():
        getattr(at, kind)(key=key).set_value(value)
        at.run(timeout=TIMEOUT)
    return at


def test_default_app_runs_cleanly():
    at = run_app()
    assert not at.exception and not at.error
    assert not at.warning  # also catches Streamlit deprecation notices
    assert "All active constraints are satisfied" in at.success[0].value
    assert any(m.label == "Sharpe ratio" for m in at.metric)


@pytest.mark.parametrize("profile", ["Conservative", "Balanced", "Growth", "Dynamic"])
def test_every_profile_runs(profile):
    at = run_app({("selectbox", "profile"): profile})
    assert not at.exception and not at.error


def test_incompatible_constraints_show_message():
    # 70% in bonds at most 5% per bond fund: the 15 bond funds give 75% at best, and the duration
    # and quality limits leave far fewer of them usable
    at = run_app({("selectbox", "profile"): "Conservative", ("slider", "Conservative.bondetf"): 5})
    assert not at.exception
    assert any("cannot all be satisfied" in e.value for e in at.error)


def test_backtest_tab_annual_rebalancing():
    at = run_app({("selectbox", "bt.freq"): "Annual"})
    assert not at.exception and not at.error and not at.warning
    assert any("Average estimated volatility at rebalancing" in c.value for c in at.caption)


def backtest_metrics(at):
    return next(df.value for df in at.dataframe if "Final wealth (USD)" in df.value.columns)


def backtest_trading(at):
    return next(df.value for df in at.dataframe if "Turnover p.a." in df.value.columns)


def test_backtest_benchmark_follows_the_profile():
    at = run_app()
    rows = {}
    for p in ["Conservative", "Balanced", "Growth", "Dynamic"]:
        at.selectbox(key="profile").set_value(p)
        at.run(timeout=TIMEOUT)
        m = backtest_metrics(at)
        assert list(m.index)[0] == "Strategy" and list(m.index)[-1] == "SPY"
        rows[p] = m.iloc[2]
        spy = m.loc["SPY"]
    vols = [rows[p]["Volatility"] for p in rows]
    rets = [rows[p]["Annual return"] for p in rows]
    assert vols == sorted(vols) and len(set(vols)) == 4 and len(set(rets)) == 4
    assert np.isclose(rows["Dynamic"]["Volatility"], spy["Volatility"])   # 100% SPY by definition
    assert rows["Balanced"]["Annual return"] < spy["Annual return"]


def test_app_defaults_match_the_engine_exactly():
    # untouched sliders must give the profile defaults exactly (e.g. EM <= 1/3, not 33%)
    # the default house view is the insider view at its default confidence
    from src import backtest as btm, data, estimators as est, insiders as ins, profiling as prof
    uni, ret, rf = data.load_universe(), data.load_returns(), data.load_rf()
    dates = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Quarterly")
    e = btm.estimates_by_date(ret, rf, est.shrinkage_groups(uni), dates, benchmarks=data.load_benchmark_returns())
    signal = ins.market_signal(data.load_insider_market())
    engine = btm.run_backtest(ret, rf, uni, prof.default_constraints("Balanced"),
                              estimates=btm.with_insider_view(e, signal, ins.DEFAULT_CONFIDENCE))
    at = run_app()
    app_final = backtest_metrics(at).loc["Strategy", "Final wealth (USD)"] / 5_000_000
    assert np.isclose(app_final, (1 + engine.returns).prod(), rtol=1e-10)


def test_view_confidence_slider():
    at = run_app()
    r50 = metric(at, "Expected return above T-bill")
    at.slider(key="views_insider").set_value(0)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert metric(at, "Expected return above T-bill") != r50
    at.slider(key="views_insider").set_value(90)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error


def test_unexpected_error_in_a_tab_shows_a_clear_message(monkeypatch):
    from src import risk

    def boom(*args, **kwargs):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(risk, "performance_metrics", boom)  # only used by the Backtest tab
    at = run_app()
    assert not at.exception                                   # no traceback on screen
    assert any("Something went wrong while building the Backtest view" in e.value for e in at.error)
    assert any(m.label == "Sharpe ratio" for m in at.metric)  # the Portfolio tab still works


def test_unexpected_error_in_the_optimisation(monkeypatch):
    from src import optimizer

    def boom(*args, **kwargs):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(optimizer, "solve", boom)
    at = run_app({("slider", "Balanced.stock"): 7})          # new inputs, so the cache is bypassed
    assert not at.exception
    assert any("unexpected error" in e.value for e in at.error)


# --- QA regression suite: edge cases, no stale values, cross-tab consistency -------------------

OPTIONAL = ["min_bonds", "max_cash", "em_equity_cap", "max_stock_weight", "max_etf_weight",
            "max_bond_etf_weight", "max_bond_duration", "min_investment_grade", "max_high_yield",
            "max_sector_weight", "exclusions"]


def holdings(at):
    return next(df.value for df in at.dataframe if "Amount (USD)" in df.value.columns)


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def clean_or_clear(at):
    """No traceback, and every error is a user-facing explanation."""
    assert not at.exception
    for e in at.error:
        assert "cannot all be satisfied" in e.value or "could not run" in e.value, e.value


@pytest.mark.parametrize("profile", ["Conservative", "Balanced", "Growth", "Dynamic"])
def test_all_optional_constraints_off(profile):
    widgets = {("selectbox", "profile"): profile}
    widgets.update({("toggle", f"{profile}.{key}.on"): False for key in OPTIONAL})
    at = run_app(widgets)
    assert not at.exception and not at.error
    assert np.isclose(holdings(at)["Weight"].sum(), 1)


@pytest.mark.parametrize("widgets", [
    {("slider", "Balanced.stock"): 1},
    {("slider", "Balanced.sector"): 5},
    {("slider", "Balanced.etf"): 5},
    {("slider", "Balanced.bondetf"): 5},       # 40% bonds over fewer eligible funds in 2012
    {("slider", "Balanced.duration"): 0.5},
    {("slider", "Balanced.ig"): 100},
    {("slider", "Balanced.hy"): 0},
    {("slider", "Balanced.vol"): 2.0},
    {("slider", "Balanced.vol"): 25.0},
    {("slider", "Balanced.bonds"): 100},       # feasible today, not in 2008 (four bond funds)
    {("slider", "Balanced.cash"): 0},
    {("slider", "Balanced.em"): 0},
], ids=lambda w: "-".join(f"{k[1]}={v}" for k, v in w.items()))
def test_edge_cases_never_crash(widgets):
    clean_or_clear(run_app(widgets))


def test_exclude_all_tech():
    at = run_app({("multiselect", "Balanced.excl.sectors"): ["Information Technology"]})
    assert not at.exception and not at.error
    h = holdings(at)
    assert h.loc[h["Sector / type"] == "Information Technology", "Weight"].sum() == 0


def test_exclude_all_stocks():
    from src import data
    uni = data.load_universe()
    at = run_app({("multiselect", "Balanced.excl.tickers"): list(uni.index[uni["vehicle"] == "Stock"])})
    clean_or_clear(at)
    assert set(holdings(at).index) - {"_other"} <= set(uni.index[uni["vehicle"] != "Stock"])


def test_holdings_consistent_with_wealth():
    at = run_app()
    h = holdings(at)
    assert np.isclose(h["Weight"].sum(), 1) and np.isclose(h["Amount (USD)"].sum(), 5_000_000)


@pytest.mark.parametrize("wealth", [100_000, 1_000_000_000])
def test_wealth_extremes_scale_everything(wealth):
    at = run_app()
    base = backtest_metrics(at)["Final wealth (USD)"] / 5_000_000
    at.number_input[0].set_value(wealth)   # client wealth (Client settings)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert np.allclose(backtest_metrics(at)["Final wealth (USD)"] / wealth, base)
    assert np.isclose(holdings(at)["Amount (USD)"].sum(), wealth)


def test_changes_propagate_no_stale_cache():
    at = run_app()
    r0, m0 = metric(at, "Expected return above T-bill"), backtest_metrics(at)
    at.slider(key="Balanced.vol").set_value(8.0)   # the volatility budget always binds
    at.run(timeout=TIMEOUT)
    r1, m1, t1 = metric(at, "Expected return above T-bill"), backtest_metrics(at), backtest_trading(at)
    assert r1 != r0 and m1.loc["Strategy", "Annual return"] != m0.loc["Strategy", "Annual return"]
    assert np.isclose(m1.iloc[2]["Annual return"], m0.iloc[2]["Annual return"])  # benchmark unaffected
    at.selectbox(key="bt.freq").set_value("Annual")
    at.run(timeout=TIMEOUT)
    assert backtest_trading(at).loc["Strategy", "Turnover p.a."] < t1.loc["Strategy", "Turnover p.a."]
    assert not at.exception and not at.error


def test_bond_portfolio_section():
    at = run_app({("selectbox", "profile"): "Conservative"})
    labels = {m.label: m.value for m in at.metric}
    assert labels["Bonds in the portfolio"].startswith("7")             # 70% floor
    years = float(labels["Average duration"].split()[0])
    assert 0 < years <= 5.0                                               # Conservative duration limit
    assert any("Share of bonds" in df.value.columns for df in at.dataframe)


def test_backtest_shows_equal_weight_and_how_to_read():
    at = run_app()
    m = backtest_metrics(at)
    assert list(m.index) == ["Strategy", "Equal weight (same universe)", "Benchmark (60% SPY / 40% AGG)", "SPY"]
    t = backtest_trading(at)
    assert list(t.index) == ["Strategy", "Equal weight (same universe)"]
    assert t.loc["Equal weight (same universe)", "Turnover p.a."] > 0 and t.loc["Strategy", "Trading costs p.a."] > 0
    assert any("How to read these results" in md.value for md in at.markdown)


def sim_table(at):
    """The simulation's two tables (wealth, then risk measures) joined by row."""
    money = next(df.value for df in at.dataframe if "Optimistic (95%)" in df.value.columns)
    return money.join(next(df.value for df in at.dataframe if "1-year VaR (95%)" in df.value.columns))


def test_simulation_tab():
    at = run_app()
    labels = {m.label: m.value for m in at.metric}
    # default answers: max one-year loss "10%"
    assert "Median wealth in 10 years" in labels and "Chance of a loss above 10% in a year" in labels
    assert "Worst fall within 10 years (typical)" in labels
    sim = sim_table(at)
    assert list(sim.index) == ["Our strategy, 2012–2026 replayed", "Benchmark (60% SPY / 40% AGG)"]
    assert (sim["Pessimistic (5%)"] < sim["Median"]).all() and (sim["Median"] < sim["Optimistic (95%)"]).all()
    assert sim["Loss above 10% in a given year"].between(0, 1).all()
    assert (sim["Worst fall, bad case (5%)"] <= sim["Worst fall, typical"]).all() and (sim["Worst fall, typical"] < 0).all()
    assert (sim["1-year expected shortfall (95%)"] <= sim["1-year VaR (95%)"]).all()
    assert "Average of the worst 5% of years" in labels
    at.slider(key="sim.years").set_value(20)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert "Median wealth in 20 years" in {m.label for m in at.metric}
    sim20 = sim_table(at)
    at.number_input[0].set_value(1_000_000)
    at.run(timeout=TIMEOUT)
    sim20_1m = sim_table(at)
    money = ["Median", "Pessimistic (5%)", "Optimistic (95%)"]
    assert np.allclose(sim20_1m[money], sim20[money] / 5)  # same scenarios, wealth scales linearly
    probs = ["Loss above 10% in a given year", "Worst fall, typical", "Worst fall, bad case (5%)",
             "1-year VaR (95%)", "1-year expected shortfall (95%)"]
    assert np.allclose(sim20_1m[probs].astype(float), sim20[probs].astype(float))


def test_simulation_follows_the_stated_loss_limit():
    at = run_app()
    at.button(key="btn.max_loss").click()   # the questionnaire is an accordion: open the question first
    at.run(timeout=TIMEOUT)
    at.radio(key="q.max_loss").set_value(0)  # "5%"
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert "Chance of a loss above 5% in a year" in {m.label for m in at.metric}


def about_names(at):
    return [md.value for md in at.markdown if "font-weight:600;font-size:1.1rem" in md.value]


def open_about(at):
    """About Us is a dialog opened by the top-right button."""
    at.button(key="btn-about").click()
    return at.run(timeout=TIMEOUT)


def test_about_us_with_team_photos():
    at = open_about(run_app())
    assert not at.exception and not at.error
    cards = about_names(at)
    assert len(cards) == 5 and any("Joao Neves" in c for c in cards)
    initials = [md.value for md in at.markdown if "border-radius:50%" in md.value and "<img" not in md.value]
    assert not initials                       # every member's photo is in assets/team/, so no initials avatar


def test_about_us_with_a_photo(tmp_path, monkeypatch):
    import base64
    from src import team
    # a test team file: one member with a real (1-pixel) photo, one whose photo file is missing
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    (team.ROOT / "assets" / "team" / "_test_photo.png").write_bytes(png)
    try:
        test_file = tmp_path / "team.json"
        test_file.write_text('{"members": [{"name": "Test Person", "role": "Role", "description": "d", '
                             '"photo": "assets/team/_test_photo.png", "linkedin": "https://www.linkedin.com/in/x"}, '
                             '{"name": "No Photo", "role": "Role", "description": "d", "photo": "assets/team/none.png"}]}',
                             encoding="utf-8")
        monkeypatch.setattr(team, "TEAM_FILE", test_file)
        monkeypatch.setattr(team.load_team, "__defaults__", (test_file,))
        at = open_about(run_app())
        assert not at.exception and not at.error
        avatars = [md.value for md in at.markdown if "border-radius:50%" in md.value]
        assert sum("<img" in a for a in avatars) == 1 and sum("<div" in a for a in avatars) == 1
        assert any("LinkedIn" in md.value for md in at.markdown)
    finally:
        (team.ROOT / "assets" / "team" / "_test_photo.png").unlink()


def test_efficient_frontier_section():
    at = run_app()
    assert not at.exception and not at.error
    assert any(md.value == "### Efficient frontier" for md in at.markdown)
    assert any("theoretical allocation line" in c.value for c in at.caption)
    assert any("maximum-Sharpe (tangency) portfolio" in c.value for c in at.caption)
    at = run_app({("selectbox", "profile"): "Dynamic", ("slider", "views_insider"): 0})
    assert not at.exception and not at.error


def test_risk_contributions_section():
    at = run_app()
    assert not at.exception and not at.error
    assert any(md.value == "### Where the risk comes from" for md in at.markdown)
    labels = {m.label: m.value for m in at.metric}
    assert {"Effective number of positions", "Effective number (entropy)", "Gini coefficient",
            "Diversification ratio", "Risk from the top 10"} <= set(labels)
    assert float(labels["Diversification ratio"]) > 1
    assert any("Euler's theorem" in c.value for c in at.caption)
    at = run_app({("selectbox", "profile"): "Dynamic"})    # benchmark = 100% SPY, no bonds held
    assert not at.exception and not at.error
    assert any("SPY carries 100% of the risk" in c.value for c in at.caption)


# --- house view: insider trading only; past performance as a naive backtest comparison ---------

def test_insider_signal_tab():
    at = run_app()
    assert not at.exception and not at.error
    labels = {m.label for m in at.metric}
    assert {"Insider buying breadth", "Signal", "Expected premium of equities over bonds",
            "Equities in the portfolio"} <= labels
    assert any("vs our stocks" in df.value.columns for df in at.dataframe)          # event study
    assert any("Why we do not bet on individual stocks" in md.value for md in at.markdown)
    assert any("insider view" in c.value for c in at.caption)                       # Portfolio caption


def test_insider_is_the_only_house_view():
    """No house-view choice in the client settings: the client sets only the confidence in the insider view."""
    at = run_app()
    assert not any(r.key == "view_source" for r in at.radio)
    assert not any(s.key == "views" for s in at.slider)                              # no past-performance slider
    r50 = metric(at, "Expected return above T-bill")
    at.slider(key="views_insider").set_value(0)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert metric(at, "Expected return above T-bill") != r50


def test_compare_house_views():
    at = run_app({("toggle", "bt.compare"): True})
    assert not at.exception and not at.error
    table = next(df.value for df in at.dataframe if "Estimated volatility (ex ante)" in df.value.columns)
    assert len(table) == 3 and table["Sharpe ratio"].notna().all()
    assert any(name.startswith("Naive: past performance") for name in table.index)


def test_missing_insider_data_falls_back(monkeypatch):
    import streamlit as st
    from src import data

    def missing(*args, **kwargs):
        raise data.DataError("Data file 'insider_market.parquet' not found. Run `python scripts/download_insiders.py`.")

    st.cache_data.clear()
    monkeypatch.setattr(data, "load_insider_market", missing)
    try:
        at = run_app()
        assert not at.exception and not at.error
        assert at.slider(key="views_insider_off").disabled                             # equilibrium, no view
        assert any("insider data is not available" in i.value for i in at.info)
    finally:
        st.cache_data.clear()


def test_home_tab():
    at = run_app()
    assert not at.exception and not at.error
    home = at.tabs[0]
    assert home.label == "Home"
    labels = {m.label: m.value for m in home.metric}
    # only what is true before the client answers: no profile or portfolio figures on Home
    assert set(labels) == {"Insider signal", "Insider buying breadth", "Investment universe", "Market data up to"}
    assert labels["Investment universe"] == "170 assets" and labels["Market data up to"] == "Sep 2026"
    assert any('<div class="home-title">Private Client Portfolio Optimizer</div>' in md.value
               for md in home.markdown)                                            # the ribbon carries the name
    assert not home.title and at.tabs[1].title[0].value == "Private Client Portfolio Optimizer"   # no header on Home
    steps = next(md.value for md in home.markdown if '<div class="home-steps">' in md.value)
    assert all(t in steps for t in ["Client Profile", "Portfolio", "Insider Signal", "Backtest", "Simulation"])

