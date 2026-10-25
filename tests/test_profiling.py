import numpy as np
import pytest

from src import data, estimators as est, optimizer as opt, profiling as prof
from tests.helpers import satisfied

ALL_LOW = {q.key: 0 for q in prof.QUESTIONS}
ALL_HIGH = {q.key: len(q.options) - 1 for q in prof.QUESTIONS}


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def estimates(universe):
    ret = data.load_returns()
    return est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), est.shrinkage_groups(universe))


def test_extreme_answers():
    assert prof.assess(ALL_LOW).profile == "Conservative"
    assert prof.assess(ALL_HIGH).profile == "Dynamic"


def test_lower_of_capacity_and_willingness():
    a = prof.assess({**ALL_HIGH, "income": 0, "wealth_share": 0})  # capacity (4+1+1)/3 = 2.0
    assert a.levels == {"capacity": "Balanced", "willingness": "Dynamic"}
    assert a.profile == "Balanced"


def test_knowledge_cap_and_horizon_knockout():
    assert prof.assess({**ALL_HIGH, "knowledge": 0}).profile == "Balanced"
    assert prof.assess({**ALL_HIGH, "knowledge": 1}).profile == "Growth"
    a = prof.assess({**ALL_HIGH, "horizon": 0})
    assert a.profile == "Conservative" and any("3 years" in r for r in a.reasons)


def test_invalid_answers():
    with pytest.raises(ValueError, match="Missing"):
        prof.assess({"horizon": 1})
    with pytest.raises(ValueError, match="Invalid"):
        prof.assess({**ALL_LOW, "max_loss": 7})


@pytest.mark.parametrize("profile", prof.PROFILES)
def test_default_portfolio_feasible(profile, universe, estimates):
    d = prof.problem_data(profile, estimates, universe)
    cons = prof.default_constraints(profile)
    res = opt.solve(d, cons)
    assert res.ok, res.message
    assert all(c.active for c in cons) and all(satisfied(c, res.weights, d) for c in cons)


def test_bond_floors_and_default_caps():
    floors = {p: next(c for c in prof.default_constraints(p) if c.key == "min_bonds").params["min_share"]
              for p in prof.PROFILES}
    assert floors == {"Conservative": 0.70, "Balanced": 0.40, "Growth": 0.20, "Dynamic": 0.0}
    params = {c.key: c.params for c in prof.default_constraints("Balanced")}
    assert params["max_cash"]["limit"] == 0.10 and params["max_stock_weight"]["limit"] == 0.08 and params["max_etf_weight"]["limit"] == 0.20
    assert params["max_bond_etf_weight"]["limit"] == 0.15 and params["max_bond_duration"]["limit"] == 6.0
    assert params["min_investment_grade"]["min_share"] == 0.80 and params["max_high_yield"]["limit"] == 0.10
    durations = [next(c for c in prof.default_constraints(p) if c.key == "max_bond_duration").params["limit"]
                 for p in prof.PROFILES]
    assert durations == sorted(durations)
    assert params["max_sector_weight"]["limit"] == 0.40 and abs(params["em_equity_cap"]["ratio"] - 1 / 3) < 1e-12


def test_risk_and_return_increase_with_profile(universe, estimates):
    caps = [prof.SPECS[p].max_volatility for p in prof.PROFILES]
    assert caps == sorted(caps)
    res = [opt.solve(prof.problem_data(p, estimates, universe), prof.default_constraints(p)) for p in prof.PROFILES]
    vols, rets = [r.volatility for r in res], [r.expected_return for r in res]
    assert vols == sorted(vols) and rets == sorted(rets)
