import numpy as np
import pandas as pd
import pytest

from src import benchmarks as bm, data, diversification as div, estimators as est, optimizer as opt, profiling as prof


def lecture_cov() -> pd.DataFrame:
    """Course example (risk budgeting, 3 assets): volatilities 20 / 30 / 15%, correlation 60% between
    assets 1 and 2, 10% between each of them and asset 3."""
    vol = np.array([0.20, 0.30, 0.15])
    corr = np.array([[1.0, 0.6, 0.1], [0.6, 1.0, 0.1], [0.1, 0.1, 1.0]])
    t = ["a1", "a2", "a3"]
    return pd.DataFrame(np.outer(vol, vol) * corr, index=t, columns=t)


def test_risk_contributions_match_the_lecture_example():
    rc = div.risk_contributions(pd.Series([0.50, 0.25, 0.25], index=["a1", "a2", "a3"]), lecture_cov())
    assert np.allclose(rc["marginal"], [0.1799, 0.2517, 0.0499], atol=5e-5)
    assert np.allclose(rc["absolute"], [0.0900, 0.0629, 0.0125], atol=5e-5)
    assert np.allclose(rc["relative"], [0.5440, 0.3806, 0.0754], atol=5e-5)
    assert np.isclose(rc["absolute"].sum(), 0.1654, atol=5e-5)        # Euler: contributions add up to sigma
    assert np.isclose(rc["relative"].sum(), 1.0)


def test_erc_portfolio_of_the_lecture_has_equal_contributions():
    rc = div.risk_contributions(pd.Series([0.3041, 0.2028, 0.4931], index=["a1", "a2", "a3"]), lecture_cov())
    assert np.allclose(rc["relative"], 1 / 3, atol=1e-3)


def test_riskless_portfolio_has_nothing_to_split():
    cov = pd.DataFrame(np.zeros((2, 2)), index=["c", "d"], columns=["c", "d"])
    rc = div.risk_contributions(pd.Series([0.5, 0.5], index=["c", "d"]), cov)
    assert (rc[["marginal", "absolute", "relative"]] == 0).all().all()


def test_group_contributions_are_sums():
    rc = div.risk_contributions(pd.Series([0.50, 0.25, 0.25], index=["a1", "a2", "a3"]), lecture_cov())
    g = div.by_group(rc, pd.Series({"a1": "Equity", "a2": "Equity", "a3": "Bond"}))
    assert np.isclose(g.loc["Equity", "relative"], 0.5440 + 0.3806, atol=1e-4)
    assert np.isclose(g["weight"].sum(), 1) and np.isclose(g["relative"].sum(), 1)


def test_concentration_measures_on_simple_cases():
    n = 4
    t = list("abcd")
    eq = div.diversification_measures(pd.Series(1 / n, index=t), pd.DataFrame(np.eye(n) * 0.04, index=t, columns=t))
    assert np.isclose(eq["herfindahl"], 1 / n) and np.isclose(eq["effective_n"], n) and np.isclose(eq["entropy_n"], n)
    assert np.isclose(eq["gini"], 0)
    assert np.isclose(eq["diversification_ratio"], np.sqrt(n))         # independent, same volatility
    one = div.diversification_measures(pd.Series([1.0, 0, 0, 0], index=t), pd.DataFrame(np.eye(n) * 0.04, index=t, columns=t))
    assert np.isclose(one["herfindahl"], 1) and np.isclose(one["entropy_n"], 1)
    assert np.isclose(one["gini"], (n - 1) / n) and np.isclose(one["diversification_ratio"], 1)


def test_perfectly_correlated_assets_give_no_diversification():
    vol = np.array([0.1, 0.2, 0.3])
    t = ["x", "y", "z"]
    cov = pd.DataFrame(np.outer(vol, vol), index=t, columns=t)
    m = div.diversification_measures(pd.Series([0.2, 0.3, 0.5], index=t), cov)
    assert np.isclose(m["diversification_ratio"], 1)


@pytest.fixture(scope="module")
def balanced():
    uni, ret = data.load_universe(), data.load_returns()
    e = est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), est.shrinkage_groups(uni),
                     benchmarks=data.load_benchmark_returns(), benchmark_class=bm.BENCHMARK_CLASS)
    e = est.with_view_confidence(e, est.DEFAULT_VIEW_CONFIDENCE)
    res = opt.solve(prof.problem_data("Balanced", e, uni), prof.default_constraints("Balanced"))
    return res, e, uni


def test_recommended_portfolio_risk_adds_up(balanced):
    res, e, uni = balanced
    rc = div.risk_contributions(res.weights, e.cov)
    assert np.isclose(rc["absolute"].sum(), res.volatility, rtol=1e-6)
    by_class = div.by_group(rc, uni["asset_class"])
    # equities carry more of the risk than of the money; cash carries none
    assert by_class.loc["Equity", "relative"] > by_class.loc["Equity", "weight"]
    assert abs(by_class.loc["Cash", "relative"]) < 1e-3
    m = div.diversification_measures(res.weights, e.cov)
    assert 1 < m["effective_n"] <= m["entropy_n"] + 1e-9 <= (res.weights > 0).sum() + 1e-6
    assert m["diversification_ratio"] > 1 and 0 < m["gini"] < 1 and 0 < m["top10_risk_share"] <= 1.5


def test_benchmark_risk_is_mostly_equity(balanced):
    _, e, _ = balanced
    rc = div.risk_contributions(pd.Series({bm.EQUITY_LEG: 0.6, bm.BOND_LEG: 0.4}), e.joint_cov)
    assert rc.loc["SPY", "relative"] > 0.8                             # 60% of the money, most of the risk
