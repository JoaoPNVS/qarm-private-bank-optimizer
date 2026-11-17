import numpy as np
import pandas as pd
import pytest

from src import data, risk


def test_portfolio_history_uses_common_months():
    ret = data.load_returns()
    w = pd.Series({"AAPL": 0.5, "PLTR": 0.5, "MSFT": 1e-6})  # dust ignored
    h = risk.portfolio_history(w, ret)
    assert h.index[0] == ret["PLTR"].first_valid_index()
    t = h.index[5]
    assert np.isclose(h[t], 0.5 * ret.loc[t, "AAPL"] + 0.5 * ret.loc[t, "PLTR"], rtol=1e-3)


def test_block_bootstrap_draws_contiguous_shared_blocks():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    hist = pd.DataFrame({"a": np.arange(60) / 1000, "b": -np.arange(60) / 1000}, index=idx)
    sims = risk.block_bootstrap(hist, horizon_months=30, n_sims=50, block=12, seed=1)
    a, b = sims["a"], sims["b"]
    assert a.shape == (50, 30) and np.allclose(a, -b)            # same blocks for every column
    steps = np.diff(a[:, :12] * 1000, axis=1)
    assert np.allclose(steps, 1)                                 # 12 consecutive months per block
    again = risk.block_bootstrap(hist, 30, 50, 12, seed=1)["a"]
    assert np.array_equal(a, again)                              # reproducible


def test_wealth_simulation_statistics():
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(0)
    hist = pd.DataFrame({"risky": rng.normal(0.01, 0.05, 48), "safe": np.full(48, 0.002)}, index=idx)
    out = risk.wealth_simulation(hist, 1_000_000, years=5, n_sims=2000)
    r, s = out["risky"], out["safe"]
    assert r["p5"] < r["median"] < r["p95"] and 0 < r["prob_loss"] < 1
    assert s["prob_loss"] == 0 and np.isclose(s["median"], 1_000_000 * 1.002 ** 60)
    b = r["bands"]
    assert len(b) == 61 and (b["p5"] <= b["p50"]).all() and (b["p50"] <= b["p95"]).all()
    assert np.allclose(b.iloc[0], 1_000_000)


def test_questionnaire_risk_measures():
    # 12-month blocks of a known pattern: year A = 12 x -2% (about -21.5%), year B = 12 x +3%
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    hist = pd.DataFrame({"x": [-0.02] * 12 + [0.03] * 12, "flat": np.full(24, 0.001)}, index=idx)
    out = risk.wealth_simulation(hist, 1.0, years=3, n_sims=4000, block=12, seed=0, year_loss_limit=0.10)
    x, flat = out["x"], out["flat"]
    # only blocks starting at month 0 are pure "A" years; a year loses more than 10% only if it is mostly A
    yearly = np.array([np.prod(1 + hist["x"].values[s:s + 12]) - 1 for s in range(13)])
    assert np.isclose(x["prob_year_loss"], (yearly < -0.10).mean(), atol=0.03)
    assert flat["prob_year_loss"] == 0 and flat["dd_median"] == 0 and flat["dd_p5"] == 0
    # worst fall: never deeper than three consecutive A years, and the bad case is at least as deep as the typical
    assert -(1 - 0.98 ** 36) <= x["dd_p5"] <= x["dd_median"] < 0
    assert np.isclose(x["dd_p5"], -(1 - 0.98 ** 12), atol=0.05) or x["dd_p5"] < -(1 - 0.98 ** 12)


def test_too_short_history_is_a_clear_error():
    hist = pd.DataFrame({"a": np.zeros(20)}, index=pd.date_range("2000-01-31", periods=20, freq="ME"))
    with pytest.raises(ValueError, match="24 months"):
        risk.block_bootstrap(hist, 60)
