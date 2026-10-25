import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est


@pytest.fixture(scope="module")
def window():
    ret = data.load_returns()
    return est.estimation_window(ret, ret.index[-1], 60), data.load_rf()


@pytest.fixture(scope="module")
def estimates(window):
    return est.estimate(*window)


def test_reference_weights_valid():
    uni = data.load_universe()
    assert np.isclose(sum(est.REFERENCE_WEIGHTS.values()), 1.0)
    assert set(est.REFERENCE_WEIGHTS) <= set(uni.index)
    by_class = pd.Series(est.REFERENCE_WEIGHTS).groupby(uni["asset_class"]).sum()
    assert np.isclose(by_class["Equity"], 0.50) and np.isclose(by_class["Bond"], 0.40)


def test_cov_psd_and_preserves_volatility(window, estimates):
    cov = estimates.cov
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-10
    risky = [c for c in cov if c != data.CASH_TICKER]
    ex = est.excess_returns(*window)
    assert np.allclose(np.diag(cov.loc[risky, risky]), ex[risky].var() * 12)
    assert (cov.loc[data.CASH_TICKER] == 0).all()
    assert 0 < estimates.shrinkage < 1


def test_equilibrium_prices_reference_at_target_sharpe(estimates):
    w, mu, cov = estimates.w_ref, estimates.mu, estimates.cov
    sharpe = w @ mu / np.sqrt(w @ cov @ w)
    assert np.isclose(sharpe, est.REFERENCE_SHARPE)
    assert estimates.mu[data.CASH_TICKER] == 0
    assert (estimates.mu.drop(data.CASH_TICKER) > 0).all()


def test_shrunk_mean_limits(window):
    ex = est.excess_returns(*window)
    raw = est.shrunk_mean(ex, 0.0)
    assert np.allclose(raw.drop(data.CASH_TICKER), ex.drop(columns=data.CASH_TICKER).mean() * 12)
    full = est.shrunk_mean(ex, 1.0).drop(data.CASH_TICKER)
    assert np.allclose(full, full.iloc[0])


def test_window_errors():
    ret = data.load_returns()
    with pytest.raises(est.EstimationError, match="Incomplete"):
        est.estimation_window(ret, pd.Timestamp("2011-06-30"), 36)
    with pytest.raises(est.EstimationError, match="Unknown"):
        est.estimate(ret.iloc[-60:], data.load_rf(), method="magic")
