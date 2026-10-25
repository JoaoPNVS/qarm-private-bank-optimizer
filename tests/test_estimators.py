import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def window():
    ret = data.load_returns()
    return est.estimation_window(ret, ret.index[-1], 60), data.load_rf()


@pytest.fixture(scope="module")
def estimates(window, universe):
    return est.estimate(*window, est.shrinkage_groups(universe))


def test_cov_psd_and_preserves_volatility(window, estimates):
    cov = estimates.cov
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-10
    risky = [c for c in cov if c != data.CASH_TICKER]
    ex = est.excess_returns(*window)
    assert np.allclose(np.diag(cov.loc[risky, risky]), ex[risky].var() * 12)
    assert (cov.loc[data.CASH_TICKER] == 0).all()
    assert 0 < estimates.cov_shrinkage < 1


def test_shrinkage_groups(universe):
    g = est.shrinkage_groups(universe)
    assert g["AAPL"] == "Equity" and g["GLD"] == "Commodity" and g["BIL"] == "Cash" and g["FLOT"] == "Cash"
    assert g["SHY"] == "Bond Short" and g["IEF"] == "Bond Intermediate" and g["TLT"] == "Bond Long"


def test_shrinkage_toward_group_mean(window, universe):
    ex = est.excess_returns(*window)
    cls = est.shrinkage_groups(universe)
    raw = est.shrunk_mean(ex, cls, 0.0)
    assert np.allclose(raw.drop(data.CASH_TICKER), ex.drop(columns=data.CASH_TICKER).mean() * 12)
    full = est.shrunk_mean(ex, cls, 1.0)
    for c in set(cls.reindex(full.index)) - {"Cash"}:
        members = full[cls.reindex(full.index) == c]
        assert np.allclose(members, members.iloc[0])  # every asset gets its class average
    half = est.shrunk_mean(ex, cls, 0.5)
    assert np.allclose(half, (raw + full) / 2)
    assert half[data.CASH_TICKER] == 0


def test_short_bonds_not_dragged_by_long_bonds(window, universe):
    # whole-class shrinkage pulled short funds toward the 2022 losses of long bonds
    ex = est.excess_returns(*window)
    by_bucket = est.shrunk_mean(ex, est.shrinkage_groups(universe))
    whole_class = est.shrunk_mean(ex, universe["asset_class"])
    assert by_bucket["SHY"] > whole_class["SHY"] and by_bucket["TLT"] < whole_class["TLT"]


def test_shrinkage_keeps_bonds_below_equities(estimates, universe):
    cls = universe["asset_class"].reindex(estimates.mu.index)
    assert estimates.mu[cls == "Bond"].mean() < estimates.mu[cls == "Equity"].mean()


def test_eligibility_drops_short_histories():
    ret = data.load_returns()
    early = est.estimation_window(ret, pd.Timestamp("2015-06-30"), 60)
    later = est.estimation_window(ret, pd.Timestamp("2018-06-30"), 60)
    assert "META" not in early.columns and "META" in later.columns  # IPO 2012-05
    assert "AAPL" in early.columns and not early.isna().any().any()


def test_window_errors():
    ret = data.load_returns()
    with pytest.raises(est.EstimationError, match="Only"):
        est.estimation_window(ret, pd.Timestamp("1996-06-30"), 60)
    with pytest.raises(est.EstimationError, match="Shrinkage"):
        est.shrunk_mean(ret.iloc[-60:].fillna(0), est.shrinkage_groups(data.load_universe()), 1.5)
