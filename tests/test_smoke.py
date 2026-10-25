"""App scenarios with Streamlit's AppTest (slow: each one runs the whole app, backtest included)."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 180


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
    assert any("Average ex-ante volatility" in c.value for c in at.caption)


def backtest_metrics(at):
    return next(df.value for df in at.dataframe if "Final wealth (USD)" in df.value.columns)


def test_backtest_benchmark_follows_the_profile():
    at = run_app()
    rows = {}
    for p in ["Conservative", "Balanced", "Growth", "Dynamic"]:
        at.selectbox(key="profile").set_value(p)
        at.run(timeout=TIMEOUT)
        m = backtest_metrics(at)
        assert list(m.index)[0] == "Strategy" and list(m.index)[-1] == "SPY"
        rows[p] = m.iloc[1]
        spy = m.loc["SPY"]
    vols = [rows[p]["Volatility"] for p in rows]
    rets = [rows[p]["Annual return"] for p in rows]
    assert vols == sorted(vols) and len(set(vols)) == 4 and len(set(rets)) == 4
    assert np.isclose(rows["Dynamic"]["Volatility"], spy["Volatility"])   # 100% SPY by definition
    assert rows["Balanced"]["Annual return"] < spy["Annual return"]


def test_app_defaults_match_the_engine_exactly():
    # untouched sliders must give the profile defaults exactly (e.g. EM <= 1/3, not 33%)
    from src import backtest as btm, data, profiling as prof
    uni, ret, rf = data.load_universe(), data.load_returns(), data.load_rf()
    engine = btm.run_backtest(ret, rf, uni, prof.default_constraints("Balanced"))
    at = run_app()
    app_final = backtest_metrics(at).loc["Strategy", "Final wealth (USD)"] / 5_000_000
    assert np.isclose(app_final, (1 + engine.returns).prod(), rtol=1e-10)


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
    at.sidebar.number_input[0].set_value(wealth)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert np.allclose(backtest_metrics(at)["Final wealth (USD)"] / wealth, base)
    assert np.isclose(holdings(at)["Amount (USD)"].sum(), wealth)


def test_changes_propagate_no_stale_cache():
    at = run_app()
    r0, m0 = metric(at, "Expected return above T-bill"), backtest_metrics(at)
    at.slider(key="Balanced.stock").set_value(3)
    at.run(timeout=TIMEOUT)
    r1, m1 = metric(at, "Expected return above T-bill"), backtest_metrics(at)
    assert r1 != r0 and m1.loc["Strategy", "Annual return"] != m0.loc["Strategy", "Annual return"]
    assert np.isclose(m1.iloc[1]["Annual return"], m0.iloc[1]["Annual return"])  # benchmark unaffected
    at.selectbox(key="bt.freq").set_value("Annual")
    at.run(timeout=TIMEOUT)
    assert backtest_metrics(at).loc["Strategy", "Turnover p.a."] < m1.loc["Strategy", "Turnover p.a."]
    at.number_input(key="bt.tc").set_value(50)
    at.run(timeout=TIMEOUT)
    assert not at.exception and not at.error
