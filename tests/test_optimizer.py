import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est, optimizer as opt
from tests.helpers import TOL, satisfied


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def estimates():
    ret = data.load_returns()
    return est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf())


@pytest.fixture(scope="module")
def pdata(estimates, universe):
    return opt.ProblemData(estimates.mu, estimates.cov, universe, estimates.delta)


def balanced_constraints():
    return [
        opt.max_weight(0.15),
        opt.asset_class_bands({"Equity": (0.35, 0.55), "Bond": (0.30, 0.55), "Alternative": (0.0, 0.15),
                               "Real Estate": (0.0, 0.10), "Cash": (0.0, 0.10)}),
        opt.em_equity_cap(1 / 3), opt.home_bias(0.30), opt.max_non_chf(0.40),
        opt.exclusions(["DBC"]), opt.max_volatility(0.09),
    ]


def test_unconstrained_equilibrium_recovers_reference(pdata, estimates):
    res = opt.solve(pdata, [])
    assert res.ok
    assert np.allclose(res.weights, estimates.w_ref, atol=1e-3)


def test_default_portfolio_is_diversified(pdata):
    res = opt.solve(pdata, balanced_constraints())
    assert res.ok
    assert res.weights.max() <= 0.15 + TOL
    assert (res.weights > 0.01).sum() >= 10


def test_raw_means_would_concentrate(estimates, universe):
    ret = data.load_returns()
    raw = est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), "shrunk_mean", 0.0)
    res = opt.solve(opt.ProblemData(raw.mu, raw.cov, universe, estimates.delta), [])
    assert res.weights.nlargest(3).sum() > 0.8


TIGHT = [
    opt.max_weight(0.08),
    opt.asset_class_bands({"Equity": (0.60, 0.70), "Bond": (0.10, 0.20)}),
    opt.em_equity_cap(0.05),
    opt.home_bias(0.50),
    opt.max_non_chf(0.20),
    opt.exclusions(["SPY", "CSSMI.SW"]),
    opt.max_volatility(0.04),
    opt.region_bands({"Europe": (0.20, 0.40), "Japan": (0.10, 0.20), "US": (0.0, 0.30)}),
]


@pytest.mark.parametrize("constraint", TIGHT, ids=lambda c: c.key)
def test_each_constraint_respected_and_binding(pdata, constraint):
    free = opt.solve(pdata, [])
    res = opt.solve(pdata, [constraint])
    assert res.ok
    assert satisfied(constraint, res.weights, pdata)
    assert not satisfied(constraint, free.weights, pdata)  # the test constraint really binds
    assert res.duals["active"].any()


def test_turnover_constraint(estimates, universe):
    w_prev = pd.Series({data.CASH_TICKER: 1.0})
    d = opt.ProblemData(estimates.mu, estimates.cov, universe, estimates.delta, w_prev=w_prev, tc=0.001)
    c = opt.max_turnover(0.5)
    res = opt.solve(d, [c])
    assert res.ok and satisfied(c, res.weights, d)
    assert np.isclose(res.turnover, 0.5, atol=1e-4)


def test_tracking_error_constraint(estimates, universe):
    bench = pd.Series({"CSSMI.SW": 0.5, "CSBGC7.SW": 0.5})
    d = opt.ProblemData(estimates.mu, estimates.cov, universe, estimates.delta, benchmark=bench)
    c = opt.max_tracking_error(0.02)
    free, res = opt.solve(d, []), opt.solve(d, [c])
    assert res.ok and satisfied(c, res.weights, d) and not satisfied(c, free.weights, d)


def test_tracking_error_ignored_without_benchmark(pdata):
    assert np.allclose(opt.solve(pdata, [opt.max_tracking_error(0.001)]).weights,
                       opt.solve(pdata, []).weights, atol=1e-6)


def test_cost_in_chf(pdata):
    cost = opt.cost_of_constraints(pdata, balanced_constraints(), wealth=5e6)
    assert np.allclose(cost["utility_cost_chf"], cost["utility_cost_bps"] * 500)


def test_all_balanced_constraints_respected(pdata):
    cons = balanced_constraints()
    res = opt.solve(pdata, cons)
    assert all(satisfied(c, res.weights, pdata) for c in cons)
    assert np.isclose(res.weights.sum(), 1) and (res.weights >= 0).all()


def test_inactive_constraint_is_ignored(pdata):
    c = opt.max_weight(0.05)
    c.active = False
    assert np.allclose(opt.solve(pdata, [c]).weights, opt.solve(pdata, []).weights, atol=1e-6)


def test_shadow_prices(pdata):
    res = opt.solve(pdata, [opt.max_weight(0.15), opt.max_non_chf(0.95)])
    d = res.duals.set_index("label")
    assert d.loc["Max weight [SPY]", "active"]          # SPY (22% in reference) hits the cap
    assert not d.loc["Non-CHF <= 95%", "active"]        # slack constraint has zero price


def test_infeasibility_is_diagnosed(pdata):
    cons = [opt.max_weight(0.15), opt.asset_class_bands({"Equity": (0.50, 1.0)}), opt.max_volatility(0.04)]
    res = opt.solve(pdata, cons)
    assert not res.ok and res.weights is None
    assert set(res.diagnosis.conflicting) == {"asset_class_bands", "max_volatility"}
    assert set(res.diagnosis.single_fixes) == {"asset_class_bands", "max_volatility"}
    assert "Maximum volatility" in res.message


def test_overlapping_conflicts(pdata):
    # A 2% volatility cap conflicts both with 50% equities and with a 15% cap per asset
    # (not enough cash and short bonds): only dropping the volatility cap fixes everything.
    cons = [opt.max_weight(0.15), opt.asset_class_bands({"Equity": (0.50, 1.0)}), opt.max_volatility(0.02)]
    diag = opt.solve(pdata, cons).diagnosis
    assert "max_volatility" in diag.conflicting and len(diag.conflicting) == 2
    assert diag.single_fixes == ["max_volatility"]


def test_single_constraint_infeasible(pdata):
    res = opt.solve(pdata, [opt.max_weight(0.02)])  # 35 assets * 2% < 100%
    assert res.status == "infeasible"
    assert res.diagnosis.conflicting == ["max_weight"]


def test_cost_of_constraints(pdata):
    cost = opt.cost_of_constraints(pdata, balanced_constraints()).set_index("key")
    assert (cost["utility_cost_bps"] >= -1e-3).all()
    assert cost.loc["max_non_chf", "utility_cost_bps"] > 0


def test_cost_of_constraints_requires_feasible_base(pdata):
    with pytest.raises(opt.OptimizationError):
        opt.cost_of_constraints(pdata, [opt.max_weight(0.02)])


def test_frontiers(pdata):
    free = opt.efficient_frontier(pdata, [], n_points=12)
    cons = opt.efficient_frontier(pdata, balanced_constraints(), n_points=12)
    for f in (free, cons):
        assert len(f.points) >= 10
        assert f.points["volatility"].is_monotonic_increasing
        assert (f.points["expected_return"].diff().dropna() >= -1e-6).all()
        assert np.allclose(f.weights.sum(axis=1), 1)
    assert cons.points["sharpe"].max() <= free.points["sharpe"].max() + 1e-4


def test_frontier_empty_when_infeasible(pdata):
    f = opt.efficient_frontier(pdata, [opt.max_weight(0.02)])
    assert f.points.empty and "No feasible" in f.message
