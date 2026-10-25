import importlib
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
MODULES = ["data", "estimators", "optimizer", "benchmarks", "backtest", "risk", "profiling", "report"]
TIMEOUT = 180


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"src.{name}")


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
    # 70% in bonds needs at least four bond funds at 20%; at 10% per fund the five funds give 50%
    at = run_app({("selectbox", "profile"): "Conservative", ("slider", "Conservative.etf"): 10})
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
