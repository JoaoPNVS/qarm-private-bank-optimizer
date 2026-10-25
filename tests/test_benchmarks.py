import numpy as np
import pandas as pd
import pytest

from src import benchmarks as bm, data, estimators as est, profiling as prof

CALIBRATION_START = "2008-02-29"


@pytest.fixture(scope="module")
def series():
    return data.load_returns(), data.load_benchmark_returns()


@pytest.fixture(scope="module")
def estimates(series):
    ret, spy = series
    uni = data.load_universe()
    return est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(), uni["asset_class"],
                        benchmark=spy)


def test_profile_benchmark_is_the_stated_mix(series):
    ret, spy = series
    b = bm.profile_benchmark_returns(0.6, spy, ret)
    assert b.index[0] == pd.Timestamp(CALIBRATION_START)
    assert np.allclose(b, (0.6 * spy + 0.4 * ret[bm.BOND_LEG]).loc[b.index])
    assert bm.profile_benchmark_returns(1.0, spy, ret).equals(spy.dropna().rename("Benchmark"))


@pytest.mark.parametrize("profile", prof.PROFILES)
def test_volatility_cap_calibrated_on_benchmark(series, profile):
    ret, spy = series
    s = prof.SPECS[profile]
    vol = bm.profile_benchmark_returns(s.benchmark_equity, spy, ret).loc[CALIBRATION_START:].std() * np.sqrt(12)
    assert vol <= s.max_volatility < vol + 0.01  # historical benchmark vol rounded up to the percent


def test_benchmark_equity_increases_with_profile():
    shares = [prof.SPECS[p].benchmark_equity for p in prof.PROFILES]
    assert shares == sorted(shares) and shares[-1] == 1.0
    assert np.isclose(prof.SPECS["Balanced"].min_defensive, 0.40)


def test_ex_ante_stats(estimates, series):
    ret, spy = series
    dyn = bm.ex_ante_stats(1.0, estimates)
    window_spy = (spy - data.load_rf()).loc[estimates.start:estimates.end]  # excess returns
    assert np.isclose(dyn["volatility"], window_spy.std() * np.sqrt(12))  # Ledoit-Wolf keeps volatilities
    bal = bm.ex_ante_stats(0.6, estimates)
    assert np.isclose(bal["expected_return"], 0.6 * estimates.benchmark_mu + 0.4 * estimates.mu[bm.BOND_LEG])
    assert bal["volatility"] < dyn["volatility"]


def test_tracking_error(estimates):
    jc = estimates.joint_cov
    spy = jc.columns[-1]
    bonds_only = pd.Series({bm.BOND_LEG: 1.0})
    expected = np.sqrt(jc.loc[spy, spy] + jc.loc[bm.BOND_LEG, bm.BOND_LEG] - 2 * jc.loc[spy, bm.BOND_LEG])
    assert np.isclose(bm.tracking_error(bonds_only, 1.0, estimates), expected)
    assert np.isclose(bm.tracking_error(bonds_only, 0.0, estimates), 0.0)


def test_benchmark_does_not_change_investable_estimates(series, estimates):
    ret, _ = series
    plain = est.estimate(est.estimation_window(ret, ret.index[-1], 60), data.load_rf(),
                         data.load_universe()["asset_class"])
    assert plain.mu.equals(estimates.mu) and plain.cov.equals(estimates.cov)
