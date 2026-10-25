import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src import backtest as btm, data, optimizer as opt, profiling as prof, risk
from tests.helpers import satisfied


@pytest.fixture(scope="module")
def inputs():
    return data.load_universe(), data.load_returns(), data.load_rf()


@pytest.fixture(scope="module")
def estimates(inputs):
    uni, ret, rf = inputs
    return btm.estimates_by_date(ret, rf, uni["asset_class"], btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Quarterly"))


@pytest.fixture(scope="module")
def balanced(inputs, estimates):
    uni, ret, rf = inputs
    return btm.run_backtest(ret, rf, uni, prof.default_constraints("Balanced"), estimates=estimates)


def test_rebalance_dates(inputs):
    _, ret, _ = inputs
    q = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Quarterly")
    s = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Semi-annual")
    a = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Annual")
    assert q[0] == pd.Timestamp("2008-12-31") and {d.month for d in q} == {3, 6, 9, 12}
    assert set(a) <= set(s) <= set(q) and {d.month for d in a} == {12}
    with pytest.raises(btm.BacktestError):
        btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Weekly")


def test_weights_valid_every_month(balanced):
    w = balanced.weights
    assert np.allclose(w.sum(axis=1), 1) and (w >= -1e-9).all().all()
    assert balanced.returns.index[0] == pd.Timestamp("2009-01-31")
    assert len(balanced.returns) == len(w) - 1


def test_constraints_hold_at_each_rebalancing(inputs, balanced, estimates):
    uni, _, _ = inputs
    cons = prof.default_constraints("Balanced")
    for t in balanced.rebalances.index:
        e = estimates[t]
        d = opt.ProblemData(e.mu, e.cov, uni)
        target = balanced.weights.loc[t].reindex(e.mu.index).fillna(0)
        assert all(satisfied(c, target, d) for c in cons), t
    assert (balanced.rebalances["ex_ante_vol"] <= prof.SPECS["Balanced"].max_volatility + 1e-6).all()


def test_returns_drift_and_costs(inputs, balanced):
    _, ret, _ = inputs
    reb = balanced.rebalances
    t = reb.index[3]
    nxt = ret.index[ret.index.get_loc(t) + 1]
    w = balanced.weights.loc[t]
    r = ret.loc[nxt].reindex(w.index).fillna(0)
    gross = w @ r
    assert np.isclose(balanced.returns[nxt], (1 + gross) * (1 - reb.loc[t, "cost"]) - 1)
    if nxt not in reb.index:  # weights drift with returns between rebalancings
        assert np.allclose(balanced.weights.loc[nxt], w * (1 + r) / (1 + gross))
    assert np.isclose(reb["cost"].sum(), btm.DEFAULT_TC * reb["turnover"].sum())
    assert reb.loc[reb.index[0], "initial"] and np.isclose(reb["turnover"].iloc[0], 1.0)


def test_realised_volatility_horizon(balanced):
    reb = balanced.rebalances
    assert reb["realised_vol"].iloc[:-4].notna().all() and reb["realised_vol"].iloc[-4:].isna().all()
    assert 0 < btm.annual_turnover(balanced) < 3


def test_infeasible_rebalancing_keeps_weights(inputs, estimates):
    uni, ret, rf = inputs

    def build(w, d, p):  # becomes impossible once more than 140 assets are eligible
        return [opt.Part("test", cp.sum(w) >= 2)] if len(d.tickers) > 140 else []

    trap = opt.Constraint("trap", "Test trap", {}, build, lambda p: "test")
    res = btm.run_backtest(ret, rf, uni, prof.default_constraints("Balanced") + [trap], estimates=estimates)
    skipped = res.rebalances[res.rebalances["status"] == "skipped"]
    assert len(skipped) > 0 and len(res.events) == len(skipped)
    t = skipped.index[0]
    prev = res.weights.index[res.weights.index.get_loc(t) - 1]
    r = ret.loc[t].reindex(res.weights.columns).fillna(0)
    expected = res.weights.loc[prev] * (1 + r) / (1 + res.weights.loc[prev] @ r)
    assert np.allclose(res.weights.loc[t], expected)  # no trade at the skipped date


def test_infeasible_start_is_a_clear_error(inputs, estimates):
    uni, ret, rf = inputs
    with pytest.raises(btm.BacktestError, match="first rebalancing"):
        btm.run_backtest(ret, rf, uni, [opt.max_volatility(0.10), opt.min_bonds(0.95), opt.max_etf_weight(0.2)],
                         estimates=estimates)


def test_performance_metrics():
    r = pd.Series([0.10, -0.05, 0.02, -0.08, 0.04], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    rf = pd.Series(0.0, index=r.index)
    m = risk.performance_metrics(r, rf)
    wealth = (1 + r).cumprod()
    assert np.isclose(m["max_drawdown"], (wealth / wealth.cummax() - 1).min())
    assert np.isclose(m["annual_return"], wealth.iloc[-1] ** (12 / 5) - 1)
    assert np.isclose(m["sharpe"], r.mean() * 12 / (r.std() * np.sqrt(12)))
