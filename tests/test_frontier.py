import numpy as np
import pytest

from src import benchmarks as bm, data, estimators as est, frontier as fr, optimizer as opt, profiling as prof


@pytest.fixture(scope="module")
def setup():
    uni, ret = data.load_universe(), data.load_returns()
    base = est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), est.shrinkage_groups(uni),
                        benchmarks=data.load_benchmark_returns(), benchmark_class=bm.BENCHMARK_CLASS)
    return prof.problem_data("Balanced", est.with_view_confidence(base, est.DEFAULT_VIEW_CONFIDENCE), uni)


def test_frontier_is_increasing_and_complete(setup):
    f = fr.efficient_frontier(setup, prof.default_constraints("Balanced"), max_vol=0.25)
    assert len(f) >= 10 and f["volatility"].is_monotonic_increasing and f["expected_return"].is_monotonic_increasing
    assert np.allclose(f[fr.CLASS_ORDER].sum(axis=1), 1)
    assert f["volatility"].max() <= 0.25 + 1e-6
    assert (f["Bond"] >= 0.40 - 1e-5).all()          # the client's bond floor holds all along the curve


def test_recommended_portfolio_lies_on_the_frontier(setup):
    cons = prof.default_constraints("Balanced")
    f = fr.efficient_frontier(setup, cons, max_vol=0.25)
    res = opt.solve(opt.ProblemData(setup.mu, setup.cov, setup.universe), cons)
    on_curve = np.interp(res.volatility, f["volatility"], f["expected_return"])
    assert abs(on_curve - res.expected_return) < 0.002   # linear interpolation of a concave curve


def test_constraints_cost_return(setup):
    with_rules = fr.efficient_frontier(setup, prof.default_constraints("Balanced"), max_vol=0.25)
    free = fr.efficient_frontier(setup, [], max_vol=0.25)
    at = with_rules["volatility"]
    assert (np.interp(at, free["volatility"], free["expected_return"]) >= with_rules["expected_return"] - 1e-4).all()
    assert free["volatility"].iloc[0] < 1e-3            # unconstrained: 100% cash is allowed


def test_infeasible_constraints_have_no_frontier(setup):
    cons = [opt.exclusions(tickers=list(setup.tickers))]
    with pytest.raises(fr.FrontierError):
        fr.efficient_frontier(setup, cons)
