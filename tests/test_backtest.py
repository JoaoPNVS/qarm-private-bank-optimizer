import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src import estimators as est, backtest as btm, data, optimizer as opt, profiling as prof, risk
from tests.helpers import satisfied


@pytest.fixture(scope="module")
def inputs():
    return data.load_universe(), data.load_returns(), data.load_rf()


@pytest.fixture(scope="module")
def estimates(inputs):
    uni, ret, rf = inputs
    return btm.estimates_by_date(ret, rf, est.shrinkage_groups(uni), btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Quarterly"))


@pytest.fixture(scope="module")
def balanced(inputs, estimates):
    uni, ret, rf = inputs
    return btm.run_backtest(ret, rf, uni, prof.default_constraints("Balanced"), estimates=estimates)


def test_rebalance_dates(inputs):
    _, ret, _ = inputs
    q = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Quarterly")
    s = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Semi-annual")
    a = btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Annual")
    assert q[0] == s[0] == a[0] == pd.Timestamp("2012-01-31")
    assert {d.month for d in q} == {1, 4, 7, 10} and {d.month for d in a} == {1}
    assert set(a) <= set(s) <= set(q)
    with pytest.raises(btm.BacktestError):
        btm.rebalance_dates(ret.index, btm.DEFAULT_START, "Weekly")


def test_weights_valid_every_month(balanced):
    w = balanced.weights
    assert np.allclose(w.sum(axis=1), 1) and (w >= -1e-9).all().all()
    assert balanced.returns.index[0] == pd.Timestamp("2012-02-29")
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
    uni_tc = (data.load_universe()["tc_bps"] / 1e4).reindex(balanced.weights.columns).fillna(0)
    expected = []
    for t in reb.index:  # cost = sum_i c_i |target_i - drifted_i|, drifted from the previous month-end
        pos = balanced.weights.index.get_loc(t)
        if pos == 0:
            before = pd.Series(0.0, index=uni_tc.index)  # first rebalancing: bought from cash
        else:
            w0 = balanced.weights.iloc[pos - 1]
            r_ = ret.loc[t].reindex(w0.index).fillna(0)
            before = w0 * (1 + r_) / (1 + w0 @ r_)
        expected.append(float(uni_tc @ (balanced.weights.loc[t] - before).abs()))
    assert np.allclose(reb["cost"], expected)
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
    # 95% in bonds at most 10% per fund needs ten bond funds; only seven existed in 2012-01
    with pytest.raises(btm.BacktestError, match="first rebalancing") as err:
        btm.run_backtest(ret, rf, uni, [opt.max_volatility(0.10), opt.min_bonds(0.95), opt.max_bond_etf_weight(0.10)],
                         estimates=estimates)
    assert "including 7 bond funds" in str(err.value) and "Minimum bonds" in str(err.value)


def test_performance_metrics():
    r = pd.Series([0.10, -0.05, 0.02, -0.08, 0.04], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    rf = pd.Series(0.0, index=r.index)
    m = risk.performance_metrics(r, rf)
    wealth = (1 + r).cumprod()
    assert np.isclose(m["max_drawdown"], (wealth / wealth.cummax() - 1).min())
    assert np.isclose(m["annual_return"], wealth.iloc[-1] ** (12 / 5) - 1)
    assert np.isclose(m["sharpe"], r.mean() * 12 / (r.std() * np.sqrt(12)))


@pytest.mark.parametrize("profile", prof.PROFILES)
def test_default_constraints_feasible_at_every_rebalancing(inputs, estimates, profile):
    # bond floor without BIL/FLOT (cash equivalents), 15% per bond fund, duration and IG limits;
    # quarterly dates contain the semi-annual and annual ones and feasibility does not depend on w_prev
    uni, ret, rf = inputs
    res = btm.run_backtest(ret, rf, uni, prof.default_constraints(profile), estimates=estimates)
    assert not res.events and (res.rebalances["status"] == "ok").all()
    bond = uni["asset_class"] == "Bond"
    floor = prof.SPECS[profile].min_bonds
    assert (res.weights.loc[res.rebalances.index, bond[res.weights.columns].values].sum(axis=1) >= floor - 1e-6).all()


def test_cost_penalty_reduces_trading(inputs, estimates):
    uni, ret, rf = inputs
    cons = prof.default_constraints("Balanced")
    with_penalty = btm.run_backtest(ret, rf, uni, cons, estimates=estimates)
    without = btm.run_backtest(ret, rf, uni, cons, estimates=estimates, penalize_costs=False)
    assert btm.annual_turnover(with_penalty) < btm.annual_turnover(without)
    assert btm.annual_cost(with_penalty) < btm.annual_cost(without)
