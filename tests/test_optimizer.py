import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est, optimizer as opt
from tests.helpers import TOL, satisfied


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


def test_volatility_cap_is_mandatory(pdata):
    res = opt.solve(pdata, [opt.max_weight(0.15)])
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


def test_constraints_respected(pdata):
    cons = [vol_cap(0.10), opt.max_weight(0.10), opt.exclusions(["SPY"])]
    res = opt.solve(pdata, cons)
    assert res.ok and all(satisfied(c, res.weights, pdata) for c in cons)
    assert res.duals["active"].any()


def test_transaction_costs_reduce_trading(estimates, universe):
    w_prev = pd.Series({"CSSMI.SW": 0.5, "CSBGC7.SW": 0.5})
    free = opt.ProblemData(estimates.mu, estimates.cov, universe, w_prev=w_prev, tc=0.0)
    costly = opt.ProblemData(estimates.mu, estimates.cov, universe, w_prev=w_prev, tc=0.05)
    assert opt.solve(costly, [vol_cap()]).turnover < opt.solve(free, [vol_cap()]).turnover


def test_infeasibility_is_diagnosed(pdata):
    cons = [vol_cap(0.04), opt.asset_class_bands({"Equity": (0.60, 1.0)})]
    res = opt.solve(pdata, cons)
    assert not res.ok and res.weights is None
    assert set(res.diagnosis.conflicting) == {"asset_class_bands", "max_volatility"}
    assert "cannot all be satisfied" in res.message


def test_single_constraint_infeasible(pdata):
    res = opt.solve(pdata, [vol_cap(), opt.max_weight(0.02)])  # 35 assets * 2% < 100%
    assert res.status == "infeasible"
    assert res.diagnosis.conflicting == ["max_weight"]
