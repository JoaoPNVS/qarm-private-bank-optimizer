import numpy as np
import pandas as pd
import pytest

from src import benchmarks as bm, data, estimators as est, profiling as prof


@pytest.fixture(scope="module")
def series():
    return data.load_returns(), data.load_benchmark_returns()


@pytest.fixture(scope="module")
def estimates(series):
    ret, bench = series
    return est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(),
                        data.load_universe()["asset_class"], benchmarks=bench, benchmark_class=bm.BENCHMARK_CLASS)


def test_profile_benchmark_is_the_stated_mix(series):
    _, bench = series
    b = bm.profile_benchmark_returns(0.6, bench)
    assert b.index[0] == pd.Timestamp("2003-10-31")
    assert np.allclose(b, (0.6 * bench["SPY"] + 0.4 * bench["AGG"]).loc[b.index])
    assert bm.profile_benchmark_returns(1.0, bench).equals(bench["SPY"].dropna().rename("Benchmark"))


@pytest.mark.parametrize("profile", prof.PROFILES)
def test_volatility_cap_calibrated_on_benchmark(series, profile):
    _, bench = series
    s = prof.SPECS[profile]
    vol = bm.profile_benchmark_returns(s.benchmark_equity, bench).loc["2003-10-31":].std() * np.sqrt(12)
    assert np.isclose(s.max_volatility, np.ceil(vol * 100) / 100)  # rounded up to the next percent


def test_benchmark_equity_increases_with_profile():
    shares = [prof.SPECS[p].benchmark_equity for p in prof.PROFILES]
    assert shares == sorted(shares) and shares[-1] == 1.0
    assert np.isclose(prof.SPECS["Balanced"].min_bonds, 0.40)


def test_ex_ante_stats(estimates, series):
    _, bench = series
    dyn = bm.ex_ante_stats(1.0, estimates)
    spy_ex = (bench["SPY"] - data.load_rf()).loc[estimates.start:estimates.end]
    assert np.isclose(dyn["volatility"], spy_ex.std() * np.sqrt(12))  # Ledoit-Wolf keeps volatilities
    bal = bm.ex_ante_stats(0.6, estimates)
    assert np.isclose(bal["expected_return"], 0.6 * estimates.benchmark_mu["SPY"] + 0.4 * estimates.benchmark_mu["AGG"])
    assert bal["volatility"] < dyn["volatility"]


def test_tracking_error(estimates):
    jc = estimates.joint_cov
    spy_only = pd.Series(0.0, index=jc.index)  # a portfolio of investable assets only
    expected = np.sqrt(jc.loc["SPY", "SPY"])
    assert np.isclose(bm.tracking_error(spy_only, 1.0, estimates), expected)
    cash = pd.Series({data.CASH_TICKER: 1.0})
    assert np.isclose(bm.tracking_error(cash, 0.0, estimates), np.sqrt(jc.loc["AGG", "AGG"]))


def test_benchmarks_do_not_change_investable_estimates(series, estimates):
    ret, _ = series
    plain = est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(),
                         data.load_universe()["asset_class"])
    assert plain.mu.equals(estimates.mu) and plain.cov.equals(estimates.cov)
