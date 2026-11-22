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


def test_max_sharpe_portfolio_beats_every_point(setup):
    cons = prof.default_constraints("Balanced")
    f = fr.efficient_frontier(setup, cons, max_vol=0.25)
    t = fr.max_sharpe_portfolio(setup, cons, f)
    sharpe = f["expected_return"] / f["volatility"]
    assert t["sharpe"] >= sharpe.max() - 1e-9
    assert np.isclose(t["sharpe"], t["expected_return"] / t["volatility"])
    res = opt.solve(opt.ProblemData(setup.mu, setup.cov, setup.universe), cons)
    assert t["sharpe"] >= res.sharpe - 1e-4                       # nothing the client may hold does better
    assert f["volatility"].min() <= t["volatility"] <= f["volatility"].max()
    assert t["Bond"] >= 0.40 - 1e-5                                # it respects the client's constraints
    # the peak is genuine: a little less or a little more risk gives a lower Sharpe ratio
    others = [c for c in cons if c.key != opt.VOL_KEY]
    for v in (t["volatility"] - 0.005, t["volatility"] + 0.005):
        r = opt.solve(opt.ProblemData(setup.mu, setup.cov, setup.universe), others + [opt.max_volatility(v)],
                      diagnose=False)
        assert r.sharpe <= t["sharpe"] + 1e-6


def test_minimum_variance_is_the_left_end(setup):
    cons = prof.default_constraints("Balanced")
    f = fr.efficient_frontier(setup, cons, max_vol=0.25)
    gmv_vol = f["volatility"].iloc[0]
    others = [c for c in cons if c.key != opt.VOL_KEY]
    data_ = opt.ProblemData(setup.mu, setup.cov, setup.universe)
    assert not opt.solve(data_, others + [opt.max_volatility(gmv_vol - 1e-3)], diagnose=False).ok
    assert opt.solve(data_, others + [opt.max_volatility(gmv_vol + 1e-4)], diagnose=False).ok


def test_infeasible_constraints_have_no_frontier(setup):
    cons = [opt.exclusions(tickers=list(setup.tickers))]
    with pytest.raises(fr.FrontierError):
        fr.efficient_frontier(setup, cons)
