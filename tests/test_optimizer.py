import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est, optimizer as opt
from tests.helpers import satisfied


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def estimates(universe):
    ret = data.load_returns()
    return est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), universe["asset_class"])


@pytest.fixture(scope="module")
def pdata(estimates, universe):
    return opt.ProblemData(estimates.mu, estimates.cov, universe)


def vol_cap(limit=0.10):
    return opt.max_volatility(limit)


def free_solution(pdata, limit=0.10):
    return opt.solve(pdata, [vol_cap(limit)]).weights


def test_volatility_cap_is_mandatory(pdata):
    res = opt.solve(pdata, [opt.max_stock_weight()])
    assert res.status == "invalid" and "volatility" in res.message


@pytest.mark.parametrize("limit", [0.04, 0.08, 0.12])
def test_max_return_hits_the_volatility_cap(pdata, limit):
    res = opt.solve(pdata, [vol_cap(limit)])
    assert res.ok
    assert np.isclose(res.volatility, limit, atol=1e-4)  # the risk budget is fully used
    assert np.isclose(res.weights.sum(), 1) and (res.weights >= 0).all()


def test_higher_cap_gives_higher_expected_return(pdata):
    r = [opt.solve(pdata, [vol_cap(v)]).expected_return for v in (0.04, 0.08, 0.12)]
    assert r == sorted(r)


def binding_cases(pdata):
    """Each constraint at a level that the unconstrained solution violates."""
    w = free_solution(pdata)
    uni = pdata.universe
    eq = uni["asset_class"] == "Equity"
    stocks, etfs = uni["vehicle"] == "Stock", uni["vehicle"] == "ETF"
    top_stock = w[stocks].idxmax()
    top_sector = w[eq].groupby(uni.loc[eq, "sector"]).sum()
    em_share = w[eq & uni["is_EM"]].sum() / w[eq].sum()
    assert em_share > 0 and w[etfs].max() > 0  # otherwise the EM / ETF cases below could not bind
    return [
        opt.max_stock_weight(w[stocks].max() / 2),
        opt.max_etf_weight(w[etfs].max() / 2),
        opt.em_equity_cap(em_share / 2),
        opt.max_sector_weight(top_sector.max() / 2),
        opt.exclusions(tickers=[top_stock]),
        opt.exclusions(sectors=[top_sector.idxmax()]),
        opt.min_bonds(min(0.9, w[uni["asset_class"] == "Bond"].sum() + 0.3)),
    ]


def test_each_constraint_respected_and_binding(pdata):
    free = free_solution(pdata)
    for c in binding_cases(pdata):
        assert not satisfied(c, free, pdata), c.key  # the test constraint really binds
        res = opt.solve(pdata, [vol_cap(), c])
        assert res.ok, c.key
        assert satisfied(c, res.weights, pdata) and satisfied(vol_cap(), res.weights, pdata), c.key
        assert res.duals["active"].any()


def test_max_cash(pdata, universe):
    cash = universe["asset_class"] == "Cash"
    free = opt.solve(pdata, [vol_cap(0.02)]).weights.reindex(universe.index).fillna(0)
    assert free[cash].sum() > 0.2  # a low risk budget pushes the free solution into cash
    c = opt.max_cash(0.10)
    res = opt.solve(pdata, [vol_cap(0.02), c])
    assert res.ok and satisfied(c, res.weights, pdata) and satisfied(vol_cap(0.02), res.weights, pdata)


def test_exclusions_by_sector_and_ticker(pdata, universe):
    c = opt.exclusions(tickers=["AAPL"], sectors=["Energy"])
    res = opt.solve(pdata, [vol_cap(), c])
    out = ["AAPL"] + [t for t in universe.index[universe["sector"] == "Energy"] if t in res.weights]
    assert res.weights[out].sum() < 1e-6
    assert "Energy sector" in c.description and "AAPL" in c.description


def test_transaction_costs_reduce_trading(estimates, universe):
    w_prev = pd.Series({"AAPL": 0.5, data.CASH_TICKER: 0.5})
    free = opt.ProblemData(estimates.mu, estimates.cov, universe, w_prev=w_prev, tc=0.0)
    costly = opt.ProblemData(estimates.mu, estimates.cov, universe, w_prev=w_prev, tc=0.05)
    assert opt.solve(costly, [vol_cap()]).turnover < opt.solve(free, [vol_cap()]).turnover


def test_inactive_constraint_is_ignored(pdata):
    c = opt.min_bonds(0.9)
    c.active = False
    assert np.allclose(opt.solve(pdata, [vol_cap(), c]).weights, free_solution(pdata), atol=1e-6)


def test_infeasibility_is_diagnosed(pdata):
    # no bonds or cash allowed and a 2% volatility cap: impossible with equities and gold only
    cons = [vol_cap(0.02), opt.exclusions(sectors=["Bonds", "Cash"])]
    res = opt.solve(pdata, cons)
    assert not res.ok and res.weights is None
    assert set(res.diagnosis.conflicting) == {"exclusions", "max_volatility"}
    assert "cannot all be satisfied" in res.message


def test_single_constraint_infeasible(pdata):
    res = opt.solve(pdata, [vol_cap(), opt.exclusions(tickers=list(pdata.tickers))])
    assert res.status == "infeasible"
    assert res.diagnosis.conflicting == ["exclusions"]
